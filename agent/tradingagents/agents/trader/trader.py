from __future__ import annotations

import functools
from typing import Any, Callable

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.swing_policy import SWING_MANAGERS_BLOCK


def create_trader(llm: Any, memory: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the Trader node for the trading graph.

    Parameters
    ----------
    llm : Any
        A LangChain-compatible chat model that implements ``invoke(messages)``.
    memory : Any
        Agent memory store exposing ``get_memories(situation, n_matches)``
        and returning a list of ``{"recommendation": str}`` dicts.

    Returns
    -------
    Callable[[dict[str, Any]], dict[str, Any]]
        A LangGraph node function (partially applied with ``name="Trader"``)
        that accepts the current ``AgentState`` dict and returns a partial
        state update with ``messages``, ``trader_investment_plan``, and
        ``sender``.

    Notes
    -----
    The trader synthesises the research manager's investment plan and all
    analyst reports into a concrete trading action. Its output concludes with
    ``FINAL TRANSACTION PROPOSAL: BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL``
    for downstream narrative flow; the cached ``decision`` token is derived from the
    portfolio manager output by ``SignalProcessor``.
    """

    def trader_node(state: dict[str, Any], name: str) -> dict[str, Any]:
        """Execute the trader node within a LangGraph propagation.

        Parameters
        ----------
        state : dict[str, Any]
            The current ``AgentState`` containing all analyst reports and the
            research manager's investment plan.
        name : str
            The display name of this node, injected via ``functools.partial``.

        Returns
        -------
        dict[str, Any]
            Partial state update with keys ``messages``, ``trader_investment_plan``,
            and ``sender``.

        Raises
        ------
        Exception
            If ``llm.invoke`` fails (network, parsing, provider errors).
        """
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        portfolio_context = state.get("portfolio_context", "") or (
            f"Portfolio tracker: no quantity held reported for {company_name}. "
            "Use fundamentals and risk only — this is reporting, not a prompt to trade."
        )

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = (
            "".join(rec["recommendation"] + "\n\n" for rec in past_memories)
            if past_memories
            else "No past memories found."
        )

        context = {
            "role": "user",
            "content": (
                "Based on a comprehensive analysis by a team of analysts, here is "
                f"an investment plan tailored for {company_name}. {instrument_context} "
                f"Proposed Investment Plan: {investment_plan}\n\n"
                "Leverage these insights to make an informed and strategic decision."
            ),
        }

        messages = [
            {
                "role": "system",
                "content": (
                    f"{SWING_MANAGERS_BLOCK}\n\n"
                    "You are a swing-trading specialist. Produce plain text only "
                    "(no Markdown, no HTML). Honor the portfolio/holdings line as the "
                    "basis when quantities and average entry are given.\n"
                    "Current position and holdings: "
                    f"{portfolio_context}\nPast reflections:\n{past_memory_str}\n\n"
                    "Give one recommendation tied to ONE decision for the WEEK. "
                    "If portfolio context shows no open position for this ticker, "
                    "do not propose SELL or UNDERWEIGHT; use HOLD for bearish views. "
                    "End with a line exactly: "
                    "FINAL TRANSACTION PROPOSAL: one of BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL "
                    "(capitalized, single phrase after the colon)."
                ),
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
