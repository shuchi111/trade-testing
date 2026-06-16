from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage  # noqa: F401 — re-exported for consumers

from tradingagents.agents.utils.swing_policy import SWING_DEBATE_REMINDER


def create_conservative_debator(llm: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the Conservative Risk Analyst node for the risk-debate phase.

    Parameters
    ----------
    llm : Any
        A LangChain-compatible chat model that implements ``invoke(prompt)``.

    Returns
    -------
    Callable[[dict[str, Any]], dict[str, Any]]
        A LangGraph node function that accepts the current ``AgentState`` dict
        and returns a partial state update with ``risk_debate_state``.

    Notes
    -----
    The conservative analyst argues for capital preservation and low-volatility
    strategies.  It challenges high-risk elements in the trader's decision and
    advocates for reduced position sizing or avoidance of speculative trades.
    """

    def conservative_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute the conservative debator node within a LangGraph propagation.

        Parameters
        ----------
        state : dict[str, Any]
            The current ``AgentState`` containing all analyst reports, the
            trader's decision, and the current risk-debate history.

        Returns
        -------
        dict[str, Any]
            Partial state update with key ``risk_debate_state``, including the
            appended conservative argument in ``history`` and
            ``conservative_history``.
        """
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")
        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Conservative Risk Analyst, protect assets and minimize volatility. Critically examine high-risk elements and advocate for low-risk alternatives.

Trader's decision: {trader_decision}

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Conversation history: {history}
Last aggressive argument: {current_aggressive_response}
Last neutral argument: {current_neutral_response}

{SWING_DEBATE_REMINDER}

Reply in plain prose only (no Markdown or HTML)."""

        response = llm.invoke(prompt)
        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
