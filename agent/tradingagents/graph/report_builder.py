"""Build CLI-style complete markdown reports from graph final state."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _section(title: str, body: str) -> str:
    text = (body or "").strip()
    if not text:
        return f"### {title}\n\n_(No report generated.)_\n"
    return f"### {title}\n{text}\n"


def _debate_field(state: Mapping[str, Any], key: str, field: str) -> str:
    block = state.get(key) or {}
    if isinstance(block, dict):
        return str(block.get(field) or "").strip()
    return ""


def build_complete_report(
    final_state: Mapping[str, Any],
    *,
    portfolio_context: str = "",
) -> str:
    """Render the same multi-section report shape as the TradingAgents CLI."""
    ticker = final_state.get("company_of_interest") or "UNKNOWN"
    trade_date = final_state.get("trade_date") or ""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    parts = [
        f"# Trading Analysis Report: {ticker}",
        "",
        f"Generated: {generated} UTC",
        f"Trade date: {trade_date}",
        "",
    ]

    ctx = (portfolio_context or final_state.get("portfolio_context") or "").strip()
    if ctx:
        parts.extend([
            "## Portfolio, Wallet, Backtest & Strategy Context",
            "",
            ctx,
            "",
        ])

    research_manager = (
        _debate_field(final_state, "investment_debate_state", "judge_decision")
        or str(final_state.get("investment_plan") or "")
    )

    parts.extend([
        "## I. Analyst Team Reports",
        "",
        _section("Market Analyst", str(final_state.get("market_report") or "")),
        _section("Sentiment / Social Analyst", str(final_state.get("sentiment_report") or "")),
        _section("News Analyst", str(final_state.get("news_report") or "")),
        _section("Fundamentals Analyst", str(final_state.get("fundamentals_report") or "")),
        "## II. Research Team Decision",
        "",
        _section("Bull Researcher", _debate_field(final_state, "investment_debate_state", "bull_history")),
        _section("Bear Researcher", _debate_field(final_state, "investment_debate_state", "bear_history")),
        _section("Research Manager", research_manager),
        "## III. Trading Team Plan",
        "",
        _section("Trader", str(final_state.get("trader_investment_plan") or "")),
        "## IV. Risk Management Team Decision",
        "",
        _section("Aggressive Analyst", _debate_field(final_state, "risk_debate_state", "aggressive_history")),
        _section("Conservative Analyst", _debate_field(final_state, "risk_debate_state", "conservative_history")),
        _section("Neutral Analyst", _debate_field(final_state, "risk_debate_state", "neutral_history")),
        "## V. Portfolio Manager Decision",
        "",
        _section("Portfolio Manager", str(final_state.get("final_trade_decision") or "")),
    ])
    return "\n".join(parts).strip() + "\n"
