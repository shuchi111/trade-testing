from __future__ import annotations

from typing import Any, Callable

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.swing_policy import SWING_DEBATE_REMINDER


def create_research_manager(llm: Any, memory: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the Research Manager (investment debate judge) node.

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
        and returns a partial state update with ``investment_debate_state`` and
        ``investment_plan``.

    Notes
    -----
    The research manager evaluates the bull/bear debate and produces a final
    investment plan (Buy / Sell / Hold with rationale) that is passed to the
    trader and subsequently to the risk-debate phase.
    """

    def research_manager_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute the research manager node within a LangGraph propagation.

        Parameters
        ----------
        state : dict[str, Any]
            The current ``AgentState`` containing all analyst reports and the
            investment-debate history.

        Returns
        -------
        dict[str, Any]
            Partial state update with keys ``investment_debate_state`` and
            ``investment_plan``.
        """
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = "".join(rec["recommendation"] + "\n\n" for rec in past_memories)

        prompt = f"""As the portfolio manager and debate facilitator, critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if strongly justified.

Develop a concise investment PLAN for ONE weekly swing horizon (Buy / Sell / Hold family with directional nuance).

{SWING_DEBATE_REMINDER}

Write plain text only (no Markdown headings, bullets with asterisks, or tables).

Past reflections: "{past_memory_str}"

{instrument_context}

Debate History:
{history}"""

        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
