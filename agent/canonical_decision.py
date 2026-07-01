"""Single source for the rating token used by signals, DB, and executor."""

from __future__ import annotations

from recommendation_bucket import recommendation_bucket
from tradingagents.graph.signal_processing import coerce_decision_token

_QUANTITY_EPSILON = 1e-6


def coerce_decision_for_holdings(
    decision: str | None,
    holding_quantity: float,
    *,
    quantity_epsilon: float = _QUANTITY_EPSILON,
) -> str:
    """
    Downgrade exit ratings when there is no open position to trim or sell.

    SELL and UNDERWEIGHT require an existing holding; bearish no-position views
    should surface as HOLD in cache and UI.
    """
    token = (decision or "").strip().upper() or "HOLD"
    if holding_quantity > quantity_epsilon:
        return token
    if recommendation_bucket(token) == "sell":
        return "HOLD"
    return token


def resolve_canonical_decision(
    decision: str | None,
    final_trade_decision: str | None = None,
) -> str:
    """
    One rating token for signal + trade.

    Always uses ``coerce_decision_token``: ``Rating:`` in the portfolio-manager
    report wins over a short extractor token (so Hold beats stray BUY in prose).
    """
    return coerce_decision_token(
        (decision or "").strip(),
        (final_trade_decision or "").strip(),
    )
