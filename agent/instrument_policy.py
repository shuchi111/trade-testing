"""Ticker-specific execution policy (fractional crypto, USD→INR quotes)."""
from __future__ import annotations

import math
import os
from functools import lru_cache

FRACTIONAL_TICKERS = frozenset({"BTC-USD", "ETH-USD"})
CRYPTO_QUANTITY_DECIMALS = 6
DEFAULT_MIN_CRYPTO_BUY_NOTIONAL_INR = 500.0
DEFAULT_USD_INR_RATE = 85.0


def is_fractional_ticker(ticker: str) -> bool:
    return (ticker or "").strip().upper() in FRACTIONAL_TICKERS


def min_crypto_buy_notional_inr() -> float:
    raw = os.getenv(
        "MIN_CRYPTO_BUY_NOTIONAL_INR", str(DEFAULT_MIN_CRYPTO_BUY_NOTIONAL_INR)
    ).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_MIN_CRYPTO_BUY_NOTIONAL_INR


def usd_inr_fallback_rate() -> float:
    raw = os.getenv("USD_INR_RATE", str(DEFAULT_USD_INR_RATE)).strip()
    try:
        rate = float(raw)
        return rate if rate > 0 else DEFAULT_USD_INR_RATE
    except ValueError:
        return DEFAULT_USD_INR_RATE


@lru_cache(maxsize=1)
def usd_inr_rate() -> float:
    """Live USD/INR from yfinance with env fallback."""
    try:
        import yfinance as yf  # type: ignore[reportMissingImports]

        ticker = yf.Ticker("USDINR=X")
        fast = getattr(ticker, "fast_info", None) or {}
        for key in ("last_price", "lastPrice"):
            value = fast.get(key)
            if value is not None:
                rate = float(value)
                if rate > 0:
                    return rate
        hist = ticker.history(period="5d")
        if hist is not None and not hist.empty:
            rate = float(hist["Close"].iloc[-1])
            if rate > 0:
                return rate
    except Exception:
        pass
    return usd_inr_fallback_rate()


def quote_to_inr(ticker: str, quote_price: float, currency: str | None = None) -> float:
    """Convert a market quote into INR for wallet math."""
    if not quote_price or quote_price <= 0:
        return 0.0
    cur = (currency or "").strip().upper()
    if is_fractional_ticker(ticker) or cur == "USD":
        return quote_price * usd_inr_rate()
    return quote_price


def size_buy_quantity(*, buy_value_inr: float, price_inr: float, fractional: bool) -> float:
    if price_inr <= 0 or buy_value_inr <= 0:
        return 0.0
    raw = buy_value_inr / price_inr
    if fractional:
        scale = 10**CRYPTO_QUANTITY_DECIMALS
        return math.floor(raw * scale) / scale
    return float(int(buy_value_inr // price_inr))


def min_buy_notional_inr(ticker: str) -> float:
    if is_fractional_ticker(ticker):
        return min_crypto_buy_notional_inr()
    return 0.0
