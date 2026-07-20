"""Write per-agent markdown files like the TradingAgents CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

def _debate_field(state: Mapping[str, Any], key: str, field: str) -> str:
    block = state.get(key) or {}
    if isinstance(block, dict):
        return str(block.get(field) or "").strip()
    return ""


def _research_manager_text(state: Mapping[str, Any]) -> str:
    return (
        _debate_field(state, "investment_debate_state", "judge_decision")
        or str(state.get("investment_plan") or "").strip()
    )


def agent_report_sections(state: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Return (section_id, title, body) for every agent in CLI order."""
    return [
        ("market", "Market Analyst", str(state.get("market_report") or "")),
        ("social", "Social Analyst", str(state.get("sentiment_report") or "")),
        ("news", "News Analyst", str(state.get("news_report") or "")),
        ("fundamentals", "Fundamentals Analyst", str(state.get("fundamentals_report") or "")),
        ("bull", "Bull Researcher", _debate_field(state, "investment_debate_state", "bull_history")),
        ("bear", "Bear Researcher", _debate_field(state, "investment_debate_state", "bear_history")),
        ("research_manager", "Research Manager", _research_manager_text(state)),
        ("trader", "Trader", str(state.get("trader_investment_plan") or "")),
        ("aggressive", "Aggressive Analyst", _debate_field(state, "risk_debate_state", "aggressive_history")),
        ("conservative", "Conservative Analyst", _debate_field(state, "risk_debate_state", "conservative_history")),
        ("neutral", "Neutral Analyst", _debate_field(state, "risk_debate_state", "neutral_history")),
        ("portfolio_manager", "Portfolio Manager", str(state.get("final_trade_decision") or "")),
    ]


def save_run_reports(
    final_state: Mapping[str, Any],
    *,
    ticker: str,
    trade_date: str,
    project_dir: str | Path | None = None,
    canonical_decision: str = "",
    portfolio_context: str = "",
    guard_note: str | None = None,
) -> dict[str, str]:
    """Persist CLI-style report files and return written paths.

    Writes:
    - ``results/<TICKER>/<trade_date>/reports/*.md`` (flat per-agent files)
    - ``reports/<TICKER>_<timestamp>/`` (folder tree + complete_report.md)
    """
    base = Path(project_dir) if project_dir else Path(__file__).resolve().parent.parent
    ticker = (ticker or final_state.get("company_of_interest") or "UNKNOWN").strip()
    trade_date = (trade_date or final_state.get("trade_date") or "").strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    flat_dir = base / "results" / ticker / trade_date / "reports"
    tree_dir = base / "reports" / f"{ticker}_{stamp}"
    flat_dir.mkdir(parents=True, exist_ok=True)
    (tree_dir / "1_analysts").mkdir(parents=True, exist_ok=True)
    (tree_dir / "2_research").mkdir(parents=True, exist_ok=True)
    (tree_dir / "3_trading").mkdir(parents=True, exist_ok=True)
    (tree_dir / "4_risk").mkdir(parents=True, exist_ok=True)
    (tree_dir / "5_portfolio").mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    sections = agent_report_sections(final_state)

    flat_map = {
        "market": "market_report.md",
        "social": "sentiment_report.md",
        "news": "news_report.md",
        "fundamentals": "fundamentals_report.md",
        "research_manager": "investment_plan.md",
        "trader": "trader_investment_plan.md",
        "portfolio_manager": "final_trade_decision.md",
    }
    tree_map = {
        "market": "1_analysts/market.md",
        "social": "1_analysts/sentiment.md",
        "news": "1_analysts/news.md",
        "fundamentals": "1_analysts/fundamentals.md",
        "bull": "2_research/bull.md",
        "bear": "2_research/bear.md",
        "research_manager": "2_research/manager.md",
        "trader": "3_trading/trader.md",
        "aggressive": "4_risk/aggressive.md",
        "conservative": "4_risk/conservative.md",
        "neutral": "4_risk/neutral.md",
        "portfolio_manager": "5_portfolio/decision.md",
    }

    for section_id, _title, body in sections:
        text = (body or "").strip()
        if section_id in flat_map:
            path = flat_dir / flat_map[section_id]
            path.write_text(text + ("\n" if text else ""), encoding="utf-8")
            written[f"flat:{section_id}"] = str(path)
        if section_id in tree_map:
            path = tree_dir / tree_map[section_id]
            path.write_text(text + ("\n" if text else ""), encoding="utf-8")
            written[f"tree:{section_id}"] = str(path)

    from tradingagents.graph.report_builder import build_complete_report

    complete = build_complete_report(
        final_state,
        portfolio_context=portfolio_context,
        canonical_decision=canonical_decision,
        guard_note=guard_note,
    )
    complete_path = tree_dir / "complete_report.md"
    complete_path.write_text(complete, encoding="utf-8")
    written["complete_report"] = str(complete_path)
    written["results_dir"] = str(flat_dir)
    written["report_tree_dir"] = str(tree_dir)
    return written
