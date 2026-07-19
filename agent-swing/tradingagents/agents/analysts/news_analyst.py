"""News analyst — multi-source macro and headline research.

Pre-fetches Yahoo news, FRED macro series, and Polymarket crowd odds before
the LLM runs, then appends the raw vendor blocks to the saved report so
CPI/unemployment/yields and prediction-market probabilities are always present.
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.claude_skills_pack import (
    _SKILLS_OBSERVE_PREAMBLE,
    skills_observe_excerpt,
)
from tradingagents.agents.utils.prefetch_context import (
    format_fred_appendix,
    format_news_appendix,
    format_polymarket_appendix,
    polymarket_topics_for,
    prefetch_fred_blocks,
    prefetch_polymarket_blocks,
)


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_news_analyst(llm):
    def news_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        start_date = _seven_days_back(current_date)
        asset_type = state.get("asset_type", "equity")
        asset_label = "company" if asset_type in ("equity", "stock") else "asset"
        instrument_context = get_instrument_context_from_state(state)
        skills_block = skills_observe_excerpt(state.get("portfolio_context", ""))

        news_block = get_news.func(ticker, start_date, current_date)
        global_news_block = get_global_news.func(current_date, 7, 10)
        fred_blocks = prefetch_fred_blocks(current_date)
        polymarket_topics = polymarket_topics_for(asset_type, ticker)
        polymarket_blocks = prefetch_polymarket_blocks(polymarket_topics)

        fred_appendix = format_fred_appendix(fred_blocks)
        polymarket_appendix = format_polymarket_appendix(polymarket_blocks)
        news_appendix = format_news_appendix(
            ticker=ticker,
            news_block=news_block,
            global_news_block=global_news_block,
        )

        system_message = f"""You are a news researcher analyzing recent news and macro trends for trading {ticker}.

All source data for the past week has been pre-fetched and is included below. Write a comprehensive report covering:
1. {asset_label.capitalize()}-specific developments from the Yahoo news block
2. Broader macro context using the FRED numeric data (cite latest CPI, unemployment, Fed funds, 10Y yield values)
3. Forward-looking risks and catalysts using Polymarket crowd-implied probabilities (cite specific % odds)
4. Actionable trading insights with a markdown summary table at the end

When FRED shows DATA_UNAVAILABLE, note that FRED_API_KEY is not configured and rely on Yahoo headlines for macro narrative.
When Polymarket or Reddit/StockTwits blocks show unavailable placeholders, say so explicitly — do not invent odds or posts.

### Yahoo Finance — ticker news ({start_date} to {current_date})
<start_of_ticker_news>
{news_block}
<end_of_ticker_news>

### Yahoo Finance — global headlines (past 7 days)
<start_of_global_news>
{global_news_block}
<end_of_global_news>

### FRED macro indicators (past ~6 months)
<start_of_fred>
{fred_appendix}
<end_of_fred>

### Polymarket prediction markets
<start_of_polymarket>
{polymarket_appendix}
<end_of_polymarket>

{_SKILLS_OBSERVE_PREAMBLE}{skills_block}
=== END SKILLS PACK EXCERPT ===

{get_language_instruction()}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " The source data is already in this prompt — do not request tools."
                    " For your reference, the current date is {current_date}. {instrument_context}"
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        formatted_messages = prompt.format_messages(messages=state["messages"])
        result = llm.invoke(formatted_messages)
        analysis = result.content if hasattr(result, "content") else str(result)

        report = "\n\n".join([
            analysis.strip(),
            "---",
            "# Source Data Appendix",
            "",
            news_appendix,
            "",
            fred_appendix,
            "",
            polymarket_appendix,
        ])

        return {
            "messages": [AIMessage(content=report)],
            "news_report": report,
        }

    return news_analyst_node
