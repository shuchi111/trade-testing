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


_CANONICAL = _BUY | _SELL | frozenset({"HOLD"})


def _canonical_token(decision: str | None) -> str:
    if not decision:
        return "HOLD"
    u = decision.strip().upper()
    if u in _CANONICAL:
        return u
    first = u.split()[0] if u else ""
    if first in _CANONICAL:
        return first
    return u


def is_sell_rating(decision: str | None) -> bool:
    """True when the rating token is SELL or UNDERWEIGHT."""
    token = _canonical_token(decision)
    return token in _SELL


def coerce_decision_for_holdings(decision: str | None, holding_quantity: float) -> str:
    """SELL/UNDERWEIGHT require an open position; otherwise downgrade to HOLD."""
    token = _canonical_token(decision)
    if float(holding_quantity or 0) <= 0 and token in _SELL:
        return "HOLD"
    return token


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
