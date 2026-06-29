from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    get_indicators,
    get_instrument_context_from_state,
    get_stock_data,
    get_verified_market_snapshot,
    langchain_tools,
    tool_names,
)
from tradingagents.agents.utils.swing_policy import SWING_MARKET_ANALYST_INSTRUCTIONS
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):
    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        market_snapshot = state.get("market_snapshot", "")

        tools = langchain_tools([get_stock_data, get_indicators, get_verified_market_snapshot])

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator.
- close_200_sma: 200 SMA: A long-term trend benchmark.
- close_10_ema: 10 EMA: A responsive short-term average.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs.
- macds: MACD Signal: An EMA smoothing of the MACD line.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions.

Volatility Indicators:
- boll: Bollinger Middle. boll_ub: Bollinger Upper Band. boll_lb: Bollinger Lower Band.
- atr: ATR: Averages true range to measure volatility.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume.

First call get_verified_market_snapshot and cite its latest date and close. Then call get_stock_data to retrieve the CSV that is needed to generate indicators.
Then use get_indicators with the specific indicator names. If the verified snapshot is unavailable or stale, clearly say so and do not invent prices.
Write a detailed, nuanced swing-trading oriented report.

"""
            + SWING_MARKET_ANALYST_INSTRUCTIONS
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    " For your reference, the current date is {current_date}. {instrument_context} {market_snapshot}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=tool_names(tools))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(market_snapshot=market_snapshot)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {"messages": [result], "market_report": report}

    return market_analyst_node
