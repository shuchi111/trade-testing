from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage  # noqa: F401 — re-exported for consumers

from tradingagents.agents.utils.swing_policy import SWING_DEBATE_REMINDER


def create_bull_researcher(llm: Any, memory: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the Bull Researcher node for the investment-debate phase.

    Parameters
    ----------
    llm : Any
        A LangChain-compatible chat model that implements ``invoke(prompt)``.
    memory : Any
        Agent memory store exposing ``get_memories(situation, n_matches)``
        and returning a list of ``{"recommendation": str}`` dicts.

    Returns
    -------
    Callable[[dict[str, Any]], dict[str, Any]]
        A LangGraph node function that accepts the current ``AgentState`` dict
        and returns a partial state update with ``investment_debate_state``.

    Notes
    -----
    The bull researcher builds an evidence-based case for investing in the
    stock, highlighting growth potential, competitive advantages, and positive
    market indicators.  It directly rebuts the bear analyst's most recent
    argument.
    """

    def bull_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute the bull researcher node within a LangGraph propagation.

        Parameters
        ----------
        state : dict[str, Any]
            The current ``AgentState`` containing all analyst reports and the
            investment-debate history.

        Returns
        -------
        dict[str, Any]
            Partial state update with key ``investment_debate_state``, including
            the appended bull argument in ``history`` and ``bull_history``.
        """
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")
        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = "".join(rec["recommendation"] + "\n\n" for rec in past_memories)

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators.

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history: {history}
Last bear argument: {current_response}
Reflections from similar situations: {past_memory_str}

{SWING_DEBATE_REMINDER}
"""

        response = llm.invoke(prompt)
        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
