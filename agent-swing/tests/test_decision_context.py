"""Tests for holdings-aware decision guards and report context."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.graph.decision_context import (
    enforce_holdings_decision,
    parse_ticker_holding,
)
from tradingagents.graph.report_builder import build_complete_report


def _ctx_for_ticker(ticker: str, holding_line: str) -> str:
    return "\n".join(
        [
            "=== LIVE PORTFOLIO ===",
            "Wallet cash: ₹10,000.00",
            "=== CURRENT TICKER FOCUS ===",
            holding_line,
            "=== CLAUDE SKILLS PACK (observe BEFORE any Buy/Sell/Hold signal) ===",
            "Consensus score=55/100 gate=caution",
            "=== END CLAUDE SKILLS PACK ===",
            f"Ticker marker: {ticker}",
        ]
    )


def test_parse_flat_holding():
    ctx = _ctx_for_ticker("TCS.NS", "No open position in TCS.NS.")
    status = parse_ticker_holding(ctx, "TCS.NS")
    assert status.is_holding is False
    assert status.quantity == 0.0


def test_parse_open_holding():
    ctx = _ctx_for_ticker("TCS.NS", "Hold: 5 TCS.NS @ ₹3,500.00 (purchased 2026-01-01, 30 days held)")
    status = parse_ticker_holding(ctx, "TCS.NS")
    assert status.is_holding is True
    assert status.quantity == 5.0


def test_enforce_clamps_sell_when_flat():
    ctx = _ctx_for_ticker("TCS.NS", "No open position in TCS.NS.")
    decision, note = enforce_holdings_decision("SELL", ctx, "TCS.NS")
    assert decision == "HOLD"
    assert note and "flat" in note.lower()


def test_enforce_clamps_underweight_when_flat():
    ctx = _ctx_for_ticker("RELIANCE.NS", "No open position in RELIANCE.NS.")
    decision, note = enforce_holdings_decision("UNDERWEIGHT", ctx, "RELIANCE.NS")
    assert decision == "HOLD"
    assert note


def test_enforce_allows_sell_when_holding():
    ctx = _ctx_for_ticker("TCS.NS", "Hold: 2 TCS.NS @ ₹3,500.00")
    decision, note = enforce_holdings_decision("SELL", ctx, "TCS.NS")
    assert decision == "SELL"
    assert note is None


def test_report_includes_ground_truth_context():
    ctx = _ctx_for_ticker("TCS.NS", "No open position in TCS.NS.")
    report = build_complete_report(
        {
            "company_of_interest": "TCS.NS",
            "trade_date": "2026-07-20",
            "market_report": "Market view.",
            "sentiment_report": "",
            "news_report": "",
            "fundamentals_report": "",
            "final_trade_decision": "Rating: Hold",
        },
        portfolio_context=ctx,
        canonical_decision="HOLD",
        guard_note="Decision guard: clamped",
    )
    assert "## 0. Ground Truth Context" in report
    assert "### Holdings Status" in report
    assert "### Claude Skills Pack & Screeners" in report
    assert "### Full Portfolio & History Context" in report
    assert "Wallet cash:" in report
    assert "CLAUDE SKILLS PACK" in report
    assert "### Decision Guard" in report
    assert "## I. Analyst Team Reports" in report
