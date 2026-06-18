"""Normalize AI decision strings — mirrors lib/ai-recommendation/decision-bucket.ts."""

from __future__ import annotations

import re

_BUY = frozenset({"BUY", "OVERWEIGHT"})
_SELL = frozenset({"SELL", "UNDERWEIGHT"})
_OVERWEIGHT_RE = re.compile(r"\bOVERWEIGHT\b", re.IGNORECASE)


def recommendation_bucket(decision: str | None) -> str:
    """Map a rating token (or short prefixed label) to buy / sell / hold / unknown."""
    if not decision:
        return "unknown"
    u = decision.strip().upper()
    if not u:
        return "unknown"
    if u in _BUY:
        return "buy"
    if u in _SELL:
        return "sell"
    if u == "HOLD":
        return "hold"

    first = u.split()[0] if u else ""
    if first in _BUY:
        return "buy"
    if first in _SELL:
        return "sell"
    if first == "HOLD":
        return "hold"

    # Ambiguous prose defaults to hold (matches TypeScript — no substring BUY/SELL scan).
    return "hold"


def is_overweight(decision: str | None) -> bool:
    if not decision:
        return False
    u = decision.strip().upper()
    if u == "OVERWEIGHT":
        return True
    first = u.split()[0] if u else ""
    if first == "OVERWEIGHT":
        return True
    return bool(_OVERWEIGHT_RE.search(u))
