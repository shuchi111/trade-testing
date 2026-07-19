"""Tests for canonical decision resolution (DB + UI parity)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "agent"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canonical_decision import resolve_canonical_decision
from tradingagents.graph.report_builder import build_complete_report
from tradingagents.graph.signal_processing import coerce_decision_token


def test_rating_line_wins_over_extractor_token():
    report = "Trader mentioned BUY earlier. Rating: Hold\n\nExecutive summary: maintain position."
    assert coerce_decision_token("BUY", report) == "HOLD"
    assert resolve_canonical_decision("BUY", report) == "HOLD"


def test_report_builder_includes_decision_header():
    state = {
        "company_of_interest": "RELIANCE.NS",
        "trade_date": "2026-07-08",
        "market_report": "neutral",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "final_trade_decision": "Rating: Overweight\n\nAdd gradually.",
    }
    md = build_complete_report(state, canonical_decision="OVERWEIGHT")
    assert "**Decision:** OVERWEIGHT" in md
