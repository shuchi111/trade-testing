"""Risk Manager agent node.

This is a port of upstream ``tradingagents/agents/managers/risk_manager.py``, added
*additively* for API parity. The default swing-trader graph does NOT wire this node —
the fork's ``portfolio_manager`` already emits ``final_trade_decision``. The node is
exposed so callers who want a separate risk-judgement step (between the risk debators
and the portfolio manager) can opt in by adding it to a custom graph.

Like the rest of the fork's managers it:
- consumes ``portfolio_context`` from ``AgentState`` (paper-wallet holding),
- injects the shared swing-trading policy block,
- emits plain-text output (no Markdown).
"""

from __future__ import annotations

from typing import Any, Callable

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.swing_policy import SWING_MANAGERS_BLOCK


def create_risk_manager(llm: Any, memory: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the Risk Manager node for the trading graph.

    The risk manager judges the aggressive / conservative / neutral risk debate and
    produces a typed risk verdict (``risk_debate_state.judge_decision``) plus an
    intermediate ``investment_plan`` update. It mirrors upstream's risk-manager step
    but is written to coexist with the fork's portfolio manager.

    Parameters
    ----------
    llm : Any
        A LangChain-compatible chat model that implements ``invoke(prompt)``.
    memory : Any
        Agent memory store exposing ``get_memories(situation, n_matches)``.

    Returns
    -------
    Callable[[dict[str, Any]], dict[str, Any]]
        A LangGraph node that returns a partial state update with
        ``risk_debate_state`` and ``investment_plan``.
    """

    def risk_manager_node(state: dict[str, Any]) -> dict[str, Any]:
        instrument_context = build_instrument_context(state["company_of_interest"])
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state["history"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state.get("trader_investment_plan", "")
        portfolio_context = state.get("portfolio_context", "") or (
            f"No open position in {state['company_of_interest']}."
        )

        curr_situation = (
            f"{market_research_report}\n\n{sentiment_report}\n\n"
            f"{news_report}\n\n{fundamentals_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = "".join(rec["recommendation"] + "\n\n" for rec in past_memories)

        prompt = f"""As the Risk Manager, judge the risk analysts' debate and translate \
the trader's plan into a risk-aware intermediate decision.

{instrument_context}

{SWING_MANAGERS_BLOCK}

Context you must weigh (holdings define the basis for percent math when applicable):
Current position and holdings summary: {portfolio_context}
Trader proposed plan (verbatim reference): {trader_plan}
Lessons from past decisions: {past_memory_str}

Risk Analysts Debate History:
{history}

Produce a concise plain-text risk verdict covering:
1. Risk verdict: one of Lower / Maintain / Raise exposure for this name this week.
2. Key risks: the most material downside scenarios from the debate (margin, leverage, news, technical breaks).
3. Position-sizing guidance: respect the twenty-five thousand INR cap and the mandatory 5% trailing stop.
4. Conditions to invalidate: what would flip the verdict next week.

Output constraints: plain text only. No Markdown headings, star bullets, HTML, fenced blocks, or tables."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Risk Manager",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "investment_plan": response.content,
        }

    return risk_manager_node
