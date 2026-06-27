from __future__ import annotations

from functools import lru_cache
from typing import Any


def normalize_symbol(symbol: str) -> str:
    """Normalize whitespace/case while preserving exchange suffixes like .NS."""
    return (symbol or "").strip().upper()


def _clean_identity_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict[str, str]:
    """Best-effort real identity lookup for a ticker.

    Identity is fail-open: recommendations should not hallucinate a company,
    but yfinance identity outages should not stop a run before data validation.
    """
    import yfinance as yf

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
