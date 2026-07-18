"""Pre-fetch vendor data blocks for analyst prompts and report appendices.

Analysts inject these blocks into the LLM prompt and append them verbatim to
the saved report so FRED, Polymarket, Reddit, and StockTwits data is always
present even when the model summarizes instead of citing sources.
"""

from __future__ import annotations

from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.symbol_utils import crypto_base

FRED_INDICATORS = ("cpi", "unemployment", "fed_funds_rate", "10y_treasury")
POLYMARKET_TOPICS_DEFAULT = ("Fed rate cut", "recession 2026")


def polymarket_topics_for(asset_type: str, ticker: str) -> tuple[str, ...]:
    """Return Polymarket search topics for the instrument type."""
    topics = list(POLYMARKET_TOPICS_DEFAULT)
    is_crypto = asset_type in ("crypto", "cryptocurrency") or "-USD" in ticker.upper()
    if is_crypto:
        base = crypto_base(ticker)
        if base:
            topics.append(base.lower())
    return tuple(dict.fromkeys(topics))


def prefetch_fred_blocks(curr_date: str, look_back_days: int = 180) -> dict[str, str]:
    """Fetch the standard FRED macro bundle (degrades gracefully without API key)."""
    blocks: dict[str, str] = {}
    for indicator in FRED_INDICATORS:
        blocks[indicator] = route_to_vendor(
            "get_macro_indicators",
            indicator,
            curr_date,
            look_back_days,
        )
    return blocks


def prefetch_polymarket_blocks(topics: tuple[str, ...], limit: int = 6) -> dict[str, str]:
    """Fetch Polymarket odds for each topic (no API key required)."""
    blocks: dict[str, str] = {}
    for topic in topics:
        blocks[topic] = route_to_vendor("get_prediction_markets", topic, limit)
    return blocks


def format_fred_appendix(blocks: dict[str, str]) -> str:
    """Render pre-fetched FRED blocks as a markdown appendix section."""
    parts = ["## FRED Macro Data (pre-fetched)", ""]
    labels = {
        "cpi": "Consumer Price Index (CPI)",
        "unemployment": "Unemployment Rate",
        "fed_funds_rate": "Federal Funds Rate",
        "10y_treasury": "10-Year Treasury Yield",
    }
    for key, content in blocks.items():
        parts.append(f"### {labels.get(key, key.replace('_', ' ').title())}")
        parts.append(content)
        parts.append("")
    return "\n".join(parts).strip()


def format_polymarket_appendix(blocks: dict[str, str]) -> str:
    """Render pre-fetched Polymarket blocks as a markdown appendix section."""
    parts = ["## Polymarket Prediction Markets (pre-fetched)", ""]
    for topic, content in blocks.items():
        parts.append(f'### Topic: "{topic}"')
        parts.append(content)
        parts.append("")
    return "\n".join(parts).strip()


def format_stocktwits_appendix(block: str) -> str:
    return "\n".join(["## StockTwits Cashtag Stream (pre-fetched)", "", block])


def format_reddit_appendix(block: str) -> str:
    return "\n".join(["## Reddit Posts (pre-fetched)", "", block])


def format_news_appendix(*, ticker: str, news_block: str, global_news_block: str) -> str:
    parts = [
        "## Yahoo Finance News (pre-fetched)",
        "",
        f"### Ticker news — {ticker}",
        news_block,
        "",
        "### Global macro headlines",
        global_news_block,
    ]
    return "\n".join(parts)
