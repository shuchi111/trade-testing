"""Single source for the rating token used by signals, DB, and executor."""

from __future__ import annotations

from tradingagents.graph.signal_processing import coerce_decision_token


def resolve_canonical_decision(
    decision: str | None,
    final_trade_decision: str | None = None,
) -> str:
    """
    One rating token for signal + trade.

    Prefers ``Rating:`` in the portfolio-manager report over a short extractor
    token (mirrors UI ``resolveCanonicalDecision``).
    """
    return coerce_decision_token(
        (decision or "").strip(),
        (final_trade_decision or "").strip(),
    )
