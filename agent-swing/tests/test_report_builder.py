"""Unit tests for swing complete_report builder (CLI agent-output shape)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.graph.report_builder import build_complete_report
from tradingagents.graph.reporting import save_run_reports


def test_build_complete_report_matches_cli_agent_sections():
    ctx = "\n".join(
        [
            "=== LIVE PORTFOLIO ===",
            "Wallet cash: ₹10,000.00",
            "=== CLAUDE SKILLS PACK ===",
            "Consensus score=62/100 gate=caution",
            "=== END CLAUDE SKILLS PACK ===",
        ]
    )
    report = build_complete_report(
        {
            "company_of_interest": "TCS.NS",
            "trade_date": "2026-07-20",
            "market_report": "Market says weekly structure intact.",
            "sentiment_report": "Social sentiment mixed.\nStockTwits Cashtag Stream: n/a",
            "news_report": "News calm.\nFRED Macro Data: n/a\nPolymarket Prediction Markets: n/a",
            "fundamentals_report": "Fundamentals solid.",
            "investment_debate_state": {
                "bull_history": "Bull: higher highs.",
                "bear_history": "Bear: valuation rich.",
                "judge_decision": "Research Manager: HOLD.",
            },
            "trader_investment_plan": "Trader: wait for pullback.",
            "risk_debate_state": {
                "aggressive_history": "Aggressive: buy dips.",
                "conservative_history": "Conservative: stay flat.",
                "neutral_history": "Neutral: patience.",
            },
            "final_trade_decision": "HOLD with GTT target price 3500",
            "portfolio_context": ctx,
        },
        portfolio_context=ctx,
        canonical_decision="HOLD",
    )

    assert "# Trading Analysis Report: TCS.NS" in report
    assert "**Decision:** HOLD" in report
    assert "## I. Analyst Team Reports" in report
    assert "### Social Analyst" in report
    assert "Social sentiment mixed." in report
    assert "## II. Research Team Decision" in report
    assert "### Bull Researcher" in report
    assert "### Bear Researcher" in report
    assert "### Research Manager" in report
    assert "## III. Trading Team Plan" in report
    assert "### Trader" in report
    assert "## IV. Risk Management Team Decision" in report
    assert "### Aggressive Analyst" in report
    assert "### Conservative Analyst" in report
    assert "### Neutral Analyst" in report
    assert "## V. Portfolio Manager Decision" in report
    assert "HOLD with GTT target price 3500" in report
    assert "## 0. Ground Truth Context" in report
    assert "Wallet cash:" in report
    assert "CLAUDE SKILLS PACK" in report


def test_save_run_reports_writes_cli_style_files(tmp_path):
    state = {
        "company_of_interest": "ETH-USD",
        "trade_date": "2026-07-07",
        "market_report": "Market body.",
        "sentiment_report": "Sentiment body.",
        "news_report": "News body.",
        "fundamentals_report": "Fundamentals body.",
        "investment_debate_state": {
            "bull_history": "Bull body.",
            "bear_history": "Bear body.",
            "judge_decision": "Manager body.",
        },
        "trader_investment_plan": "Trader body.",
        "risk_debate_state": {
            "aggressive_history": "Aggressive body.",
            "conservative_history": "Conservative body.",
            "neutral_history": "Neutral body.",
        },
        "final_trade_decision": "FINAL TRANSACTION PROPOSAL: **SELL**",
    }
    paths = save_run_reports(
        state,
        ticker="ETH-USD",
        trade_date="2026-07-07",
        project_dir=tmp_path,
        canonical_decision="SELL",
    )
    flat = tmp_path / "results" / "ETH-USD" / "2026-07-07" / "reports"
    assert (flat / "market_report.md").read_text(encoding="utf-8") == "Market body.\n"
    assert (flat / "sentiment_report.md").read_text(encoding="utf-8") == "Sentiment body.\n"
    assert (flat / "investment_plan.md").read_text(encoding="utf-8") == "Manager body.\n"
    assert (flat / "final_trade_decision.md").read_text(encoding="utf-8").startswith("FINAL TRANSACTION")
    tree = Path(paths["report_tree_dir"])
    assert (tree / "1_analysts" / "market.md").read_text(encoding="utf-8") == "Market body.\n"
    assert (tree / "2_research" / "bull.md").read_text(encoding="utf-8") == "Bull body.\n"
    assert (tree / "4_risk" / "neutral.md").read_text(encoding="utf-8") == "Neutral body.\n"
    complete = (tree / "complete_report.md").read_text(encoding="utf-8")
    assert "## I. Analyst Team Reports" in complete
    assert "## V. Portfolio Manager Decision" in complete
    assert "FINAL TRANSACTION PROPOSAL" in complete
