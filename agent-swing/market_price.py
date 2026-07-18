from __future__ import annotations

import logging
import os

import yfinance as yf  # type: ignore[reportMissingImports]

logger = logging.getLogger("market_price")


def fetch_last_close(symbol: str, period: str | None = None) -> float | None:
    """Return last quoted close for ``symbol``: split-adjusted if available."""
    hist_period = (
        (period.strip() if period else None)
        or os.getenv("YFINANCE_HISTORY_PERIOD", "").strip()
        or "10d"
    )

    sym = symbol.upper()
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period=hist_period)
        if hist is None or hist.empty:
            return None
        adj = hist["Adj Close"] if "Adj Close" in hist.columns else hist["Close"]
        val = adj.iloc[-1]
        out = float(val) if val is not None else None
        return out if out is not None and out > 0 else None
    except Exception as exc:
        logger.error("fetch_last_close %s failed: %s", sym, exc)
        return None
