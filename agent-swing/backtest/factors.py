"""Factor library inspired by Qlib's Alpha158 handler.

Computes a curated subset (~30) of engineered OHLCV features as plain pandas
rolling expressions — no ``qlib`` dependency needed, since these are simple
windowed calculations over close/high/low/volume.

Design goals
------------
1. Drop-in feature frame for an ML strategy (see plan §1.2) or as compact
   "quant context" injected into the Market Analyst prompt.
2. Convention-compatible with the backtest package: returns ``pd.DataFrame``
   indexed like the input price Series, NaN-safe, no side effects.
3. Each factor is a standalone function so callers can compute a subset.

References
----------
Qlib Alpha158 handler:
https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

Not implemented here (intentionally, to keep deps to pandas/numpy):
- KBAR* shape ratios (require open/high/low we don't reliably have for crypto)
- full 158 set — ~30 high-signal factors cover the same families (momentum,
  volatility, volume, mean-reversion) for a 30-ticker universe.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Windows used across the factor families — mirror Alpha158's common defaults.
_STD_WINDOWS = (5, 10, 20, 30, 60)


def _series(price: pd.Series) -> pd.Series:
    """Coerce input to a clean Series with a DatetimeIndex and float dtype."""
    if not isinstance(price, pd.Series):
        price = pd.Series(price)
    return price.astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise ratio with zero-division guard (inf/0 -> NaN)."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


# ════════════════════════════════════════════════════════════════════════
# Momentum / return factors
# ════════════════════════════════════════════════════════════════════════

def log_return(price: pd.Series, window: int = 1) -> pd.Series:
    """ROLLR — log return over ``window`` periods."""
    p = _series(price)
    return np.log(p).diff(window)


def pct_return(price: pd.Series, window: int = 5) -> pd.Series:
    """Simple percentage return over ``window`` periods."""
    p = _series(price)
    return p.pct_change(window)


def cumulative_return_5d(price: pd.Series) -> pd.Series:
    """Alpha158 STD5 cumulative return, expressed as a fraction."""
    return pct_return(price, 5)


# ════════════════════════════════════════════════════════════════════════
# Mean-reversion factors (price vs moving average)
# ════════════════════════════════════════════════════════════════════════

def price_to_ma_ratio(price: pd.Series, window: int = 20) -> pd.Series:
    """MA ratio — current price divided by its simple moving average."""
    p = _series(price)
    ma = p.rolling(window, min_periods=max(1, window // 2)).mean()
    return _safe_ratio(p, ma)


def price_minus_ma(price: pd.Series, window: int = 20) -> pd.Series:
    """Distance (in log space) from the moving average — Alpha158 VSIGN."""
    p = _series(price)
    ma = p.rolling(window, min_periods=max(1, window // 2)).mean()
    return np.log(p) - np.log(ma.replace(0, np.nan))


def rsi_wilder(price: pd.Series, window: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing (matches Alpha158's RSI convention)."""
    p = _series(price)
    delta = p.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = _safe_ratio(avg_gain, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


# ════════════════════════════════════════════════════════════════════════
# Volatility factors
# ════════════════════════════════════════════════════════════════════════

def realized_volatility(price: pd.Series, window: int = 20) -> pd.Series:
    """STD of log returns over ``window`` — Alpha158 VOLSTD."""
    lr = log_return(price, 1)
    return lr.rolling(window, min_periods=max(2, window // 2)).std()


def return_skew(price: pd.Series, window: int = 20) -> pd.Series:
    """Skew of log returns over ``window`` — Alpha158 SKEW."""
    lr = log_return(price, 1)
    return lr.rolling(window, min_periods=max(2, window // 2)).skew()


def return_kurtosis(price: pd.Series, window: int = 20) -> pd.Series:
    """Excess kurtosis of log returns over ``window`` — tail-risk signal."""
    lr = log_return(price, 1)
    return lr.rolling(window, min_periods=max(2, window // 2)).kurt()


def atr_like(price: pd.Series, window: int = 20) -> pd.Series:
    """Volatility proxy using daily ranges when full OHLC is unavailable.

    Approximates the true-range magnitude from close-to-close moves when only
    a close-price series is available (the common case for our data loader).
    """
    p = _series(price)
    # |return| as a cheap single-bar "range" proxy, smoothed over window.
    abs_ret = p.pct_change().abs()
    return abs_ret.rolling(window, min_periods=max(2, window // 2)).mean()


# ════════════════════════════════════════════════════════════════════════
# Volume factors (no-op safe when volume is unavailable)
# ════════════════════════════════════════════════════════════════════════

def volume_ma_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """VSTD — current volume over its moving average."""
    v = _series(volume)
    ma = v.rolling(window, min_periods=max(1, window // 2)).mean()
    return _safe_ratio(v, ma)


def volume_volatility(volume: pd.Series, window: int = 20) -> pd.Series:
    """Coefficient of variation of volume — detects abnormal trading activity."""
    v = _series(volume)
    mean = v.rolling(window, min_periods=max(2, window // 2)).mean()
    std = v.rolling(window, min_periods=max(2, window // 2)).std()
    return _safe_ratio(std, mean)


# ════════════════════════════════════════════════════════════════════════
# Composite feature frame
# ════════════════════════════════════════════════════════════════════════

def compute_factors(
    price: pd.Series,
    volume: pd.Series | None = None,
    windows: tuple[int, ...] = _STD_WINDOWS,
) -> pd.DataFrame:
    """Compute the full Alpha158-inspired factor frame for one ticker.

    Parameters
    ----------
    price : pd.Series
        Close prices, indexed by trading day.
    volume : pd.Series, optional
        Trading volume aligned to ``price``. If None, all volume factors are
        omitted (keeps the frame dependency-light for crypto/synthetic data).
    windows : tuple[int, ...]
        Rolling windows applied across the momentum/volatility families.

    Returns
    -------
    pd.DataFrame
        One column per factor, indexed identically to ``price``. Leading rows
        are NaN where the window has not yet filled.
    """
    price = _series(price)
    frames: dict[str, pd.Series] = {}

    # Momentum family — one column per window.
    for w in windows:
        frames[f"log_return_{w}"] = log_return(price, w)
        frames[f"pct_return_{w}"] = pct_return(price, w)

    # Mean-reversion family.
    for w in (5, 10, 20, 60):
        frames[f"price_to_ma_{w}"] = price_to_ma_ratio(price, w)
    frames["price_minus_ma_20"] = price_minus_ma(price, 20)
    frames["rsi_14"] = rsi_wilder(price, 14)

    # Volatility family.
    for w in (10, 20, 60):
        frames[f"realized_vol_{w}"] = realized_volatility(price, w)
    frames["return_skew_20"] = return_skew(price, 20)
    frames["return_kurt_20"] = return_kurtosis(price, 20)
    frames["atr_like_20"] = atr_like(price, 20)

    # Volume family — only when volume was supplied.
    if volume is not None:
        vol = _series(volume).reindex(price.index)
        if vol.notna().any():
            frames["volume_ma_ratio_20"] = volume_ma_ratio(vol, 20)
            frames["volume_cv_20"] = volume_volatility(vol, 20)

    out = pd.DataFrame(frames, index=price.index)
    # Replace inf with NaN so downstream ML / storage stays clean.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def factor_summary(price: pd.Series, volume: pd.Series | None = None) -> dict:
    """Latest-row snapshot of every factor — compact context for prompts/UI.

    Convenient when you want "the factor values as of today" as a flat dict
    rather than the full history (e.g. to inject into an LLM prompt).
    """
    frame = compute_factors(price, volume=volume)
    if frame.empty:
        return {}
    latest = frame.iloc[-1]
    # Round + drop NaN to keep the snapshot readable.
    return {
        name: round(float(val), 4)
        for name, val in latest.items()
        if pd.notna(val)
    }


if __name__ == "__main__":
    # Quick CLI smoke test: python -m backtest.factors --ticker RELIANCE.NS
    # Machine-readable output for the /api/ml/factors route:
    #   python -m backtest.factors --ticker RELIANCE.NS --json
    import argparse
    import json

    try:
        from .data_loader import load_price_data, load_volume_data  # type: ignore
    except ImportError:
        from data_loader import load_price_data, load_volume_data  # type: ignore

    parser = argparse.ArgumentParser(description="Alpha158-inspired factor smoke test")
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the latest factor snapshot as a single JSON object on stdout "
        "(for consumption by /api/ml/factors).",
    )
    args = parser.parse_args()

    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    px = load_price_data(args.ticker, args.start, end)
    vol = load_volume_data(args.ticker, args.start, end)
    if px.empty:
        if args.json:
            print(json.dumps({"ok": False, "error": f"No price data for {args.ticker}"}))
        else:
            print(f"No price data for {args.ticker}")
        raise SystemExit(0)

    summary = factor_summary(px, volume=vol)
    if args.json:
        as_of = str(px.index[-1].date()) if hasattr(px.index[-1], "date") else str(px.index[-1])
        print(json.dumps({"ok": True, "ticker": args.ticker, "as_of": as_of, "factors": summary}))
    else:
        print(f"Latest factor snapshot for {args.ticker}:")
        for name, val in summary.items():
            print(f"  {name:24s} {val}")
