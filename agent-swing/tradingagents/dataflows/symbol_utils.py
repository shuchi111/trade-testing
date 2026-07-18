"""Symbol normalization and identity resolution for vendor calls."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import yfinance as yf  # type: ignore[reportMissingImports]

from .errors import NoMarketDataError as NoMarketDataError

logger = logging.getLogger(__name__)

_FOREX_CURRENCIES = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "CNY", "CNH", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN",
        "MXN", "ZAR", "TRY", "INR", "KRW", "BRL", "RUB", "THB",
    }
)

_CRYPTO_BASES = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "DOT", "AVAX", "LINK"}
)

_ALIASES = {
    "XAUUSD": "GC=F", "XAU": "GC=F", "GOLD": "GC=F",
    "XAGUSD": "SI=F", "XAG": "SI=F", "SILVER": "SI=F",
    "XPTUSD": "PL=F", "XPDUSD": "PA=F",
    "WTICOUSD": "CL=F", "USOIL": "CL=F", "WTI": "CL=F",
    "BCOUSD": "BZ=F", "UKOIL": "BZ=F", "BRENT": "BZ=F",
    "NATGAS": "NG=F", "XNGUSD": "NG=F",
    "COPPER": "HG=F", "XCUUSD": "HG=F",
    "SPX500": "^GSPC", "US500": "^GSPC", "SPX": "^GSPC",
    "NAS100": "^NDX", "US100": "^NDX", "USTEC": "^NDX",
    "US30": "^DJI", "DJI30": "^DJI", "WS30": "^DJI",
    "GER40": "^GDAXI", "GER30": "^GDAXI", "DE40": "^GDAXI",
    "UK100": "^FTSE", "JP225": "^N225", "JPN225": "^N225",
    "FRA40": "^FCHI", "EU50": "^STOXX50E", "HK50": "^HSI",
}

_YAHOO_SAFE = re.compile(r"^[A-Za-z0-9._\-\^=]+$")
_CRYPTO_QUOTES = ("USDT", "USDC", "USD")
_INDIAN_EXCHANGE_SUFFIXES = (".NS", ".BO")


def indian_equity_base(raw: str) -> str | None:
    """Return the NSE/BSE base symbol (e.g. ``SUNPHARMA`` from ``SUNPHARMA.NS``)."""
    if not isinstance(raw, str):
        return None
    symbol = raw.strip().upper()
    for suffix in _INDIAN_EXCHANGE_SUFFIXES:
        if symbol.endswith(suffix):
            base = symbol[: -len(suffix)]
            return base if base else None
    return None


def is_indian_equity(raw: str) -> bool:
    return indian_equity_base(raw) is not None


def reddit_search_term(raw: str) -> str:
    """Normalize a ticker into a Reddit search query."""
    return indian_equity_base(raw) or crypto_base(raw) or raw.strip()


def crypto_base(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    compact = raw.strip().upper().rstrip("+").replace("-", "")
    for quote in _CRYPTO_QUOTES:
        if compact.endswith(quote):
            base = compact[: -len(quote)]
            return base if base in _CRYPTO_BASES else None
    return None


def _normalize_crypto(s: str) -> str | None:
    base = crypto_base(s)
    return f"{base}-USD" if base else None


def normalize_symbol(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip().upper().rstrip("+")
    crypto = _normalize_crypto(s)
    if s in _ALIASES:
        canonical = _ALIASES[s]
    elif crypto is not None:
        canonical = crypto
    elif len(s) == 6 and s[:3] in _FOREX_CURRENCIES and s[3:] in _FOREX_CURRENCIES:
        canonical = f"{s}=X"
    else:
        canonical = s

    if canonical != raw.strip().upper():
        logger.info("Resolved symbol %r to Yahoo symbol %r", raw, canonical)
    return canonical


def is_yahoo_safe(symbol: str) -> bool:
    return bool(symbol) and _YAHOO_SAFE.fullmatch(symbol) is not None


def _clean_identity_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict[str, str]:
    symbol = normalize_symbol(ticker)
    if not symbol:
        return {}

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return {}

    identity: dict[str, str] = {}
    name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if name:
        identity["company_name"] = name

    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("currency", "currency"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value

    return identity
