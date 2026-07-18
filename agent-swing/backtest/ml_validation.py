"""ML-tab validation helpers: regime detection, cross-sectional momentum, bootstrap.

Standalone CLI — does not modify backtest runner or strategies.

Examples:
  python -m backtest.ml_validation --mode regime --ticker RELIANCE.NS --json
  python -m backtest.ml_validation --mode cross_sectional --json
  python -m backtest.ml_validation --mode bootstrap --returns 1.2,-0.5,2.1 --json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_PREFERRED = {
    "trending": ["trend", "momentum", "ml"],
    "bull": ["trend", "momentum", "ml"],
    "chop": ["mean_reversion", "composite", "ml"],
    "ranging": ["mean_reversion", "composite", "ml"],
    "bear": ["mean_reversion", "baseline"],
    "high_vol": ["mean_reversion", "baseline"],
}


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder smoothing)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def detect_regime(ticker: str, start: str = "2022-01-01", end: str | None = None) -> dict:
    """Classify market regime from ADX + realized vol ratio."""
    try:
        from .data_loader import load_ohlcv
    except ImportError:
        from data_loader import load_ohlcv

    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    ohlcv = load_ohlcv(ticker, start, end)
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 60:
        return {
            "ok": False,
            "error": f"Insufficient OHLCV for {ticker}",
            "ticker": ticker,
        }

    high = ohlcv["High"]
    low = ohlcv["Low"]
    close = ohlcv["Close"]
    adx_series = _adx(high, low, close)
    adx_val = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else None

    rets = close.pct_change().dropna()
    vol20 = float(rets.tail(20).std() * np.sqrt(252) * 100) if len(rets) >= 20 else None
    vol60 = float(rets.tail(60).std() * np.sqrt(252) * 100) if len(rets) >= 60 else None
    vol_ratio = (vol20 / vol60) if vol20 and vol60 and vol60 > 0 else None

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    price = float(close.iloc[-1])
    above50 = price > float(ma50.iloc[-1]) if pd.notna(ma50.iloc[-1]) else None
    above200 = price > float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else None

    label = "chop"
    description = "Range-bound / choppy — mean reversion strategies tend to work better."

    if adx_val is not None and adx_val >= 25 and above50 and above200:
        label = "trending"
        description = "Strong trend (ADX ≥ 25, price above MAs) — trend/momentum/ML preferred."
    elif adx_val is not None and adx_val >= 25 and not above50:
        label = "bear"
        description = "Downtrend with directional strength — defensive / mean reversion."
    elif vol_ratio is not None and vol_ratio >= 1.35:
        label = "high_vol"
        description = "Elevated short-term volatility — reduce size; mean reversion or cash."
    elif above50 and above200:
        label = "bull"
        description = "Uptrend without extreme ADX — blend trend and ML signals."
    elif adx_val is not None and adx_val < 20:
        label = "ranging"
        description = "Low ADX ranging market — RSI/Bollinger-style mean reversion."

    return {
        "ok": True,
        "ticker": ticker,
        "label": label,
        "adx": round(adx_val, 2) if adx_val is not None else None,
        "vol_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
        "vol20_ann_pct": round(vol20, 2) if vol20 is not None else None,
        "preferred_families": REGIME_PREFERRED.get(label, []),
        "description": description,
        "as_of": str(close.index[-1].date()) if hasattr(close.index[-1], "date") else str(close.index[-1]),
    }


def cross_sectional_momentum(
    tickers: list[str],
    lookback: int = 20,
    top_n: int = 5,
    start: str = "2022-01-01",
    end: str | None = None,
) -> dict:
    """Rank universe by trailing momentum (video-style cross-sectional filter)."""
    try:
        from .data_loader import load_price_data
    except ImportError:
        from data_loader import load_price_data

    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    rows: list[dict] = []
    for ticker in tickers:
        try:
            price = load_price_data(ticker, start, end)
            if price is None or len(price) < lookback + 2:
                continue
            close = price if isinstance(price, pd.Series) else price["Close"]
            mom = (float(close.iloc[-1]) / float(close.iloc[-1 - lookback]) - 1.0) * 100.0
            rows.append({"ticker": ticker, "momentum20d_pct": round(mom, 3)})
        except Exception as exc:
            logger.debug("skip %s: %s", ticker, exc)

    rows.sort(key=lambda r: r["momentum20d_pct"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    return {
        "ok": True,
        "lookback_days": lookback,
        "top_n": top_n,
        "universe_size": len(tickers),
        "ranked": rows[: max(top_n, len(rows))],
        "top_picks": rows[:top_n],
    }


def bootstrap_returns(
    returns_pct: list[float],
    iterations: int = 500,
    seed: int = 42,
) -> dict:
    """Bootstrap total compounded return distribution from per-trade returns (%)."""
    if len(returns_pct) < 3:
        return {
            "ok": True,
            "iterations": iterations,
            "sample_size": len(returns_pct),
            "median_return_pct": None,
            "p5_return_pct": None,
            "p95_return_pct": None,
            "positive_pct": None,
            "verdict": "insufficient",
        }

    rng = np.random.default_rng(seed)
    arr = np.array(returns_pct, dtype=float) / 100.0
    n = len(arr)
    totals = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sample = rng.choice(arr, size=n, replace=True)
        totals[i] = (np.prod(1.0 + sample) - 1.0) * 100.0

    positive_pct = float((totals > 0).mean() * 100.0)
    median = float(np.median(totals))
    p5 = float(np.percentile(totals, 5))
    p95 = float(np.percentile(totals, 95))

    if positive_pct >= 65 and p5 > -15:
        verdict = "robust"
    elif positive_pct >= 45:
        verdict = "fragile"
    else:
        verdict = "fragile"

    return {
        "ok": True,
        "iterations": iterations,
        "sample_size": n,
        "median_return_pct": round(median, 3),
        "p5_return_pct": round(p5, 3),
        "p95_return_pct": round(p95, 3),
        "positive_pct": round(positive_pct, 1),
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ML validation helpers")
    parser.add_argument("--mode", required=True, choices=["regime", "cross_sectional", "bootstrap"])
    parser.add_argument("--ticker", help="Ticker for regime mode")
    parser.add_argument("--tickers", help="Comma-separated tickers for cross_sectional")
    parser.add_argument("--returns", help="Comma-separated trade return %% for bootstrap")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "regime":
        if not args.ticker:
            out = {"ok": False, "error": "--ticker required for regime mode"}
        else:
            out = detect_regime(args.ticker.strip().upper())
    elif args.mode == "cross_sectional":
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        else:
            tickers = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "SBIN.NS", "WIPRO.NS", "BAJFINANCE.NS", "LT.NS", "MARUTI.NS",
                "BHARTIARTL.NS", "HINDUNILVR.NS", "ITC.NS", "SUNPHARMA.NS", "AXISBANK.NS",
            ]
        out = cross_sectional_momentum(tickers, top_n=args.top)
    else:
        if not args.returns:
            out = {"ok": False, "error": "--returns required for bootstrap mode"}
        else:
            rets = [float(x) for x in args.returns.split(",") if x.strip()]
            out = bootstrap_returns(rets)

    if args.json:
        print(json.dumps(out))
    else:
        print(out)
    return 0 if out.get("ok", False) else 1


if __name__ == "__main__":
    sys.exit(main())
