from __future__ import annotations

from typing import Any, Callable

from tradingagents.agents.utils.swing_policy import SWING_DEBATE_REMINDER


def create_neutral_debator(llm: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the Neutral Risk Analyst node for the risk-debate phase.

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
    The neutral analyst provides a balanced perspective, weighing both potential
    benefits and risks, and challenges both the aggressive and conservative
    analysts to ensure a well-rounded risk debate.
    """

    def neutral_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute the neutral debator node within a LangGraph propagation.

        Parameters
        ----------
        state : dict[str, Any]
            The current ``AgentState`` containing all analyst reports, the
            trader's decision, and the current risk-debate history.

        Returns
        -------
        dict[str, Any]
            Partial state update with key ``risk_debate_state``, including the
            appended neutral argument in ``history`` and ``neutral_history``.
        """
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")
        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        trader_decision = state["trader_investment_plan"]

        prompt = f"""As the Neutral Risk Analyst, provide a balanced perspective weighing both potential benefits and risks. Challenge both aggressive and conservative analysts.

Trader's decision: {trader_decision}

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Conversation history: {history}
Last aggressive argument: {current_aggressive_response}
Last conservative argument: {current_conservative_response}

{SWING_DEBATE_REMINDER}

Reply in plain prose only (no Markdown or HTML)."""

        response = llm.invoke(prompt)
        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
