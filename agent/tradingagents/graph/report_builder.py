"""Build CLI-style complete markdown reports from graph final state."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from tradingagents.agents.utils.prefetch_context import (
    format_fred_appendix,
    format_polymarket_appendix,
    format_reddit_appendix,
    format_stocktwits_appendix,
    polymarket_topics_for,
    prefetch_fred_blocks,
    prefetch_polymarket_blocks,
)
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.graph.signal_processing import coerce_decision_token


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


def _has_source_section(text: str, section_title: str) -> bool:
    return section_title.lower() in (text or "").lower()


def _fallback_source_appendices(final_state: Mapping[str, Any]) -> str:
    """Fetch raw source appendices when analyst reports did not persist them.

    The analyst nodes normally append these blocks themselves. This safeguard
    keeps cron ``complete_report.md`` self-contained even if a node falls back
    to a legacy/plain report shape.
    """
    ticker = str(final_state.get("company_of_interest") or "").strip()
    if not ticker:
        return ""

    asset_type = str(final_state.get("asset_type") or "equity").strip() or "equity"
    combined_reports = "\n\n".join([
        str(final_state.get("sentiment_report") or ""),
        str(final_state.get("news_report") or ""),
    ])

    appendices: list[str] = []
    if not _has_source_section(combined_reports, "StockTwits Cashtag Stream"):
        appendices.append(format_stocktwits_appendix(fetch_stocktwits_messages(ticker, limit=30)))
    if not _has_source_section(combined_reports, "Reddit Posts"):
        appendices.append(format_reddit_appendix(fetch_reddit_posts(ticker)))
    if not _has_source_section(combined_reports, "FRED Macro Data"):
        appendices.append(format_fred_appendix(prefetch_fred_blocks(str(final_state.get("trade_date") or ""))))
    if not _has_source_section(combined_reports, "Polymarket Prediction Markets"):
        topics = polymarket_topics_for(asset_type, ticker)
        appendices.append(format_polymarket_appendix(prefetch_polymarket_blocks(topics)))

    if not appendices:
        return ""

    return "\n\n".join([
        "## VI. Raw Source Data Appendices",
        "",
        "\n\n".join(appendices),
    ])


def build_complete_report(
    final_state: Mapping[str, Any],
    *,
    portfolio_context: str = "",
    canonical_decision: str = "",
) -> str:
    """Render the same multi-section report shape as the TradingAgents CLI."""
    ticker = final_state.get("company_of_interest") or "UNKNOWN"
    trade_date = final_state.get("trade_date") or ""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    final_pm = str(final_state.get("final_trade_decision") or "")
    decision = (canonical_decision or "").strip() or coerce_decision_token("", final_pm)

    parts = [
        f"# Trading Analysis Report: {ticker}",
        "",
        f"**Decision:** {decision}",
        f"Generated: {generated} UTC",
        f"Trade date: {trade_date}",
        "",
    ]

    ctx = (portfolio_context or final_state.get("portfolio_context") or "").strip()
    parts.extend([
        "## Portfolio, Wallet, Backtest & Strategy Context",
        "",
        ctx or "_No portfolio, wallet, holdings, or strategy context was supplied to this run._",
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
        _section("Portfolio Manager", final_pm),
    ])

    fallback_appendices = _fallback_source_appendices(final_state)
    if fallback_appendices:
        parts.extend(["", fallback_appendices])

    return "\n".join(parts).strip() + "\n"
