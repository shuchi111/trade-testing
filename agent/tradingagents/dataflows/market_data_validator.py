from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from .errors import StaleMarketDataError, VendorDataError
from .symbol_utils import normalize_symbol


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    vendor: str
    latest_date: str
    latest_close: float
    latest_volume: float | None
    trade_date: str | None
    stale: bool
    stale_reason: str
    age_days: int | None
    currency: str | None = None
    exchange: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _max_stale_days() -> int:
    return max(0, int(os.getenv("MAX_MARKET_DATA_STALE_DAYS", "5")))


def verified_market_snapshot(
    ticker: str,
    trade_date: str | date | None = None,
    *,
    period: str | None = None,
    max_stale_days: int | None = None,
) -> MarketSnapshot:
    """Return a real yfinance snapshot with freshness metadata."""
    import yfinance as yf

    symbol = normalize_symbol(ticker)
    hist_period = period or os.getenv("YFINANCE_HISTORY_PERIOD", "1y")
    try:
        yf_ticker = yf.Ticker(symbol)
        hist = yf_ticker.history(period=hist_period)
    except Exception as exc:
        raise VendorDataError(f"yfinance snapshot failed for {symbol}: {exc}") from exc

    if hist is None or hist.empty:
        raise VendorDataError(f"No real market history returned for {symbol}")

    row = hist.iloc[-1]
    latest_idx = hist.index[-1]
    latest_date = _as_date(getattr(latest_idx, "date", lambda: latest_idx)())
    if latest_date is None:
        latest_date = _as_date(latest_idx)
    if latest_date is None:
        raise VendorDataError(f"Could not resolve latest bar date for {symbol}")

    price_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
    latest_close = row.get(price_col)
    try:
        close = float(latest_close)
    except Exception as exc:
        raise VendorDataError(f"Invalid latest close for {symbol}: {latest_close}") from exc
    if close <= 0:
        raise VendorDataError(f"Invalid non-positive latest close for {symbol}: {close}")

    trade_dt = _as_date(trade_date)
    age_days: int | None = None
    stale = False
    stale_reason = ""
    if trade_dt is not None:
        age_days = (trade_dt - latest_date).days
        max_days = _max_stale_days() if max_stale_days is None else max_stale_days
        if age_days < 0:
            stale = True
            stale_reason = (
                f"latest market bar {latest_date.isoformat()} is after trade date "
                f"{trade_dt.isoformat()}"
            )
        elif age_days > max_days:
            stale = True
            stale_reason = (
                f"latest market bar {latest_date.isoformat()} is {age_days} days "
                f"before trade date {trade_dt.isoformat()} (max {max_days})"
            )

    info: dict[str, Any] = {}
    try:
        info = yf_ticker.fast_info or {}
    except Exception:
        info = {}

    volume = None
    try:
        volume = float(row["Volume"]) if "Volume" in hist.columns else None
    except Exception:
        volume = None

    return MarketSnapshot(
        ticker=symbol,
        vendor="yfinance",
        latest_date=latest_date.isoformat(),
        latest_close=close,
        latest_volume=volume,
        trade_date=trade_dt.isoformat() if trade_dt else None,
        stale=stale,
        stale_reason=stale_reason,
        age_days=age_days,
        currency=str(info.get("currency")) if info.get("currency") else None,
        exchange=str(info.get("exchange")) if info.get("exchange") else None,
    )


def require_fresh_market_snapshot(
    ticker: str,
    trade_date: str | date | None,
    *,
    period: str | None = None,
    max_stale_days: int | None = None,
) -> MarketSnapshot:
    snapshot = verified_market_snapshot(
        ticker, trade_date, period=period, max_stale_days=max_stale_days
    )
    if snapshot.stale:
        raise StaleMarketDataError(snapshot.stale_reason)
    return snapshot


def format_market_snapshot(snapshot: MarketSnapshot | dict[str, Any]) -> str:
    data = snapshot.to_dict() if isinstance(snapshot, MarketSnapshot) else snapshot
    volume = data.get("latest_volume")
    volume_text = f"{volume:,.0f}" if isinstance(volume, (int, float)) else "n/a"
    stale_text = "STALE" if data.get("stale") else "fresh"
    reason = data.get("stale_reason") or "none"
    return (
        f"Verified market snapshot ({data.get('vendor', 'unknown')}): "
        f"{data.get('ticker')} close {data.get('latest_close')} on "
        f"{data.get('latest_date')}, volume {volume_text}, freshness {stale_text}, "
        f"stale reason: {reason}."
    )
