"""Build complete markdown reports: ground truth context + all agent outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from tradingagents.graph.decision_context import build_ground_truth_report_section
from tradingagents.graph.reporting import agent_report_sections
from tradingagents.graph.signal_processing import coerce_decision_token


def _section(title: str, body: str) -> str:
    text = (body or "").strip()
    if not text:
        return f"### {title}\n\n_(No report generated.)_\n"
    return f"### {title}\n\n{text}\n"


def build_complete_report(
    final_state: Mapping[str, Any],
    *,
    portfolio_context: str = "",
    canonical_decision: str = "",
    guard_note: str | None = None,
) -> str:
    """Render ground-truth context plus all agent outputs (CLI sections 0 + I–V)."""
    ticker = final_state.get("company_of_interest") or "UNKNOWN"
    trade_date = final_state.get("trade_date") or ""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    final_pm = str(final_state.get("final_trade_decision") or "")
    decision = (canonical_decision or "").strip() or coerce_decision_token("", final_pm)
    sections = {sid: (title, body) for sid, title, body in agent_report_sections(final_state)}

    ground_truth = build_ground_truth_report_section(
        portfolio_context,
        str(ticker),
        guard_note=guard_note,
    )

    parts: list[str] = [
        f"# Trading Analysis Report: {ticker}",
        "",
        f"**Decision:** {decision}",
        f"Generated: {generated} UTC",
        f"Trade date: {trade_date}",
        "",
        ground_truth,
        "",
        "## I. Analyst Team Reports",
        "",
        _section(sections["market"][0], sections["market"][1]),
        _section(sections["social"][0], sections["social"][1]),
        _section(sections["news"][0], sections["news"][1]),
        _section(sections["fundamentals"][0], sections["fundamentals"][1]),
        "## II. Research Team Decision",
        "",
        _section(sections["bull"][0], sections["bull"][1]),
        _section(sections["bear"][0], sections["bear"][1]),
        _section(sections["research_manager"][0], sections["research_manager"][1]),
        "## III. Trading Team Plan",
        "",
        _section(sections["trader"][0], sections["trader"][1]),
        "## IV. Risk Management Team Decision",
        "",
        _section(sections["aggressive"][0], sections["aggressive"][1]),
        _section(sections["conservative"][0], sections["conservative"][1]),
        _section(sections["neutral"][0], sections["neutral"][1]),
        "## V. Portfolio Manager Decision",
        "",
        _section(sections["portfolio_manager"][0], sections["portfolio_manager"][1] or final_pm),
    ]
    return "\n".join(parts).strip() + "\n"
