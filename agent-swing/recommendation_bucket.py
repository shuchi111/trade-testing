"""Normalize AI decision strings — mirrors lib/ai-recommendation/decision-bucket.ts."""

from __future__ import annotations

_BUY = frozenset({"BUY", "OVERWEIGHT"})
_SELL = frozenset({"SELL", "UNDERWEIGHT"})


def recommendation_bucket(decision: str | None) -> str:
    if not decision:
        return "unknown"
    u = decision.strip().upper()
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
    for kw in _BUY:
        if kw in u:
            return "buy"
    for kw in _SELL:
        if kw in u:
            return "sell"
    return "hold"


def is_overweight(decision: str | None) -> bool:
    if not decision:
        return False
    u = decision.strip().upper()
    return u == "OVERWEIGHT" or "OVERWEIGHT" in u
