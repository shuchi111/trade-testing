from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_indicators, get_stock_data
from tradingagents.agents.utils.claude_skills_pack import (
    _SKILLS_OBSERVE_PREAMBLE,
    skills_observe_excerpt,
)
from tradingagents.agents.utils.swing_policy import SWING_MARKET_ANALYST_INSTRUCTIONS


def create_market_analyst(llm):
    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        skills_block = skills_observe_excerpt(state.get("portfolio_context", ""))

        tools = [get_stock_data, get_indicators]

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

Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators.
Then use get_indicators with the specific indicator names. Write a detailed, nuanced swing-trading oriented report.

"""
            + SWING_MARKET_ANALYST_INSTRUCTIONS
            + _SKILLS_OBSERVE_PREAMBLE
            + skills_block
            + "\n=== END SKILLS PACK EXCERPT ===\n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    " For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {"messages": [result], "market_report": report}

    return market_analyst_node
