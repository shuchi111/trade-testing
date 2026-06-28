from langchain_core.messages import HumanMessage, RemoveMessage

from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.technical_indicators_tools import get_indicators
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news,
)
from tradingagents.agents.utils.market_data_validation_tools import get_verified_market_snapshot
from tradingagents.dataflows.symbol_utils import resolve_instrument_identity


def build_instrument_context(ticker: str, identity: dict | None = None) -> str:
    """Describe the exact instrument so agents preserve ticker and company identity"""
    context = (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )
    identity = identity if identity is not None else resolve_instrument_identity(ticker)
    details = []
    if identity:
        if identity.get("company_name"):
            details.append(f"Company: {identity['company_name']}")
        if identity.get("sector") and identity.get("industry"):
            details.append(f"Business classification: {identity['sector']} / {identity['industry']}")
        elif identity.get("sector"):
            details.append(f"Sector: {identity['sector']}")
        elif identity.get("industry"):
            details.append(f"Industry: {identity['industry']}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")
        if identity.get("currency"):
            details.append(f"Currency: {identity['currency']}")
    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool result explicitly disproves this identity."
        )
    return context


def get_instrument_context_from_state(state: dict) -> str:
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(state["company_of_interest"])


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add an anchored placeholder for provider compatibility."""
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        instrument_context = get_instrument_context_from_state(state)
        trade_date = state.get("trade_date", "the requested date")
        placeholder = HumanMessage(
            content=(
                "Continue the assigned analysis for this workflow. "
                f"{instrument_context} The analysis date is {trade_date}."
            )
        )
        return {"messages": removal_operations + [placeholder]}
    return delete_messages
