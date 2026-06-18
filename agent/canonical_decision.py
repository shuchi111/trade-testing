"""Single source for the rating token used by signals, DB, and executor."""

from __future__ import annotations

from tradingagents.graph.signal_processing import coerce_decision_token


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
