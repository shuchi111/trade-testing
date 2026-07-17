from __future__ import annotations

from typing import Any, Callable

from tradingagents.agents.utils.swing_policy import (
    SWING_DEBATE_REMINDER,
    format_live_portfolio_context,
)


def create_aggressive_debator(llm: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the Aggressive Risk Analyst node for the risk-debate phase.

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
    The aggressive analyst champions high-reward opportunities and challenges
    conservative or neutral views with data-driven rebuttals.  It advocates for
    larger position sizes and higher-risk / higher-return strategies.
    """

    def aggressive_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute the aggressive debator node within a LangGraph propagation.

        Parameters
        ----------
        state : dict[str, Any]
            The current ``AgentState`` containing all analyst reports, the
            trader's decision, and the current risk-debate history.

        Returns
        -------
        dict[str, Any]
            Partial state update with key ``risk_debate_state``, including the
            appended aggressive argument in ``history`` and
            ``aggressive_history``.
        """
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        trader_decision = state["trader_investment_plan"]
        live_ctx = format_live_portfolio_context(state.get("portfolio_context", ""))

        prompt = f"""As the Aggressive Risk Analyst, champion high-reward, high-risk opportunities. Challenge conservative and neutral views with data-driven rebuttals.

CRITICAL: Read LIVE PORTFOLIO CONTEXT first. If holdings, cash, or past losses do not support aggressive risk, say so — do not ignore the DB facts.

Trader's decision: {trader_decision}

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Conversation history: {history}
Last conservative argument: {current_conservative_response}
Last neutral argument: {current_neutral_response}

{live_ctx}

{SWING_DEBATE_REMINDER}

Reply in plain prose only (no Markdown or HTML)."""

        response = llm.invoke(prompt)
        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
