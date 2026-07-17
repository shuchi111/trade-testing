from __future__ import annotations

from typing import Any, Callable

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.swing_policy import (
    SWING_MANAGERS_BLOCK,
    format_live_portfolio_context,
)


def create_portfolio_manager(llm: Any, memory: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create the Portfolio Manager node for the trading graph.

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
        and returns a partial state update with ``risk_debate_state`` and
        ``final_trade_decision``.

    Notes
    -----
    The portfolio manager acts as the final judge in the risk-debate phase.
    It synthesises the aggressive, conservative, and neutral analysts' arguments
    alongside the trader's proposed plan and past reflective memories to produce
    a Buy / Overweight / Hold / Underweight / Sell rating with an executive
    summary and investment thesis.

    The node prompt (see ``portfolio_manager_node``) requires a **fixed plain-text
    structure** ending with explicit **GTT target price** and **GTT stop price**
    lines in INR (one sentence each), aligned with the swing policy block. The
    model must also include ``Rating:``, ``Executive summary:`` (with Action plan,
    Entry strategy, Position sizing, Key risk levels, Time horizon),
    ``Investment thesis:``, and ``Weekly candles tie-in:``.

    .. warning::
        Output is an AI-generated signal for **research purposes only**.
        It is NOT financial advice and MUST NOT be used for real-money trading.
    """

    def portfolio_manager_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute the portfolio manager node within a LangGraph propagation.

        Parameters
        ----------
        state : dict[str, Any]
            The current ``AgentState`` containing all analyst reports, the
            risk-debate history, and the trader's investment plan.

        Returns
        -------
        dict[str, Any]
            Partial state update with keys ``risk_debate_state`` and
            ``final_trade_decision``.
        """
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]
        portfolio_context = state.get("portfolio_context", "") or (
            f"No open position in {state['company_of_interest']}."
        )
        live_ctx = format_live_portfolio_context(portfolio_context)

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = "".join(rec["recommendation"] + "\n\n" for rec in past_memories)

        prompt = f"""As the Portfolio Manager with 20+ years of swing-trading experience, \
synthesize the risk analysts' debate and deliver the final trading decision.

CRITICAL PROCESS (do in order):
1) Read the FULL LIVE PORTFOLIO CONTEXT (DB: holdings, trades, past AI decisions, backtests, lessons).
2) Cross-check the trader plan and risk debate against that context.
3) Think carefully about capital risk — then decide Buy / Overweight / Hold / Underweight / Sell.
Never decide from analyst narrative alone while ignoring holdings or past losses.

{SWING_MANAGERS_BLOCK}

{instrument_context}

---

Rating scale — use exactly one word for the Rating line (capitalized exactly as written):
Buy: Strong conviction to enter or add
Overweight: Favorable outlook, increase gradually
Hold: Maintain, no churn this week unless thesis broke
Underweight: Reduce partly
Sell: Exit or avoid entry

{live_ctx}

Trader proposed plan (verbatim reference): {trader_plan}
Lessons from past decisions (memory): {past_memory_str}

**Required Output Structure:**
1. **Rating**: State one of Buy / Overweight / Hold / Underweight / Sell.
2. **Executive Summary**: A concise action plan covering entry strategy, position \
   sizing, key risk levels, and time horizon.
3. **Investment Thesis**: Detailed reasoning anchored in the analysts' debate and past \
reflections.
4. **Backtest and live trade review**: What backtest results and live trade history imply.

Rating: Buy | Overweight | Hold | Underweight | Sell

Executive summary:
Action plan: prose for the week (one stance, no scalp churn).
Entry strategy: prose (add / trim / wait; levels in words if helpful).
Position sizing: prose (how heavy or light; respect swing, holdings basis, and the twenty-five thousand INR cap).
Key risk levels: prose (including the mandatory 5% trailing stop, wallet reserve, and thesis-break ideas).
Time horizon: prose (multi-week swing; not day-trade framing).

Investment thesis: detailed reasoning in one or more paragraphs; explicitly cite holdings status, \
at least one live-trade or past-AI fact, and LESSONS FROM PAST MISTAKES when relevant.

Context review (required short paragraph): confirm you checked wallet/holdings/trades/past AI \
decisions/backtests/lessons and state the key constraint that shaped today's rating.

Backtest and live trade review: summarise what the backtest strategy summary, backtest trade dates, live trade history, AI execution history, and past reports in portfolio context imply for this week's rating. Cite specific backtest entry/exit dates when warning against whipsaw. State days held and where the holding sits inside the ninety-day swing exit window when recommending any exit.

Weekly candles tie-in: how multi-week candle structure supports or contradicts the stance (pattern names in plain words).

GTT target price: INR value and one sentence (tie to >3% gain versus basis when recommending profit-taking).

GTT stop price: INR value and one sentence (use the mandatory 5% trailing stop distance, not a 10% discretionary stop).

Risk/reward ratio: numeric ratio using target reward divided by stop risk (example 1.50). Prefer HOLD if below 1.50.

AI confidence: percentage from 0% to 100% and one short reason.

When you write the final answer, reproduce the section labels above verbatim (Rating:, Executive summary:, then Action plan:, Entry strategy:, Position sizing:, Key risk levels:, Time horizon:, then Investment thesis:, Context review:, Backtest and live trade review:, Weekly candles tie-in:, GTT target price:, GTT stop price:, Risk/reward ratio:, AI confidence:) and fill in real content—do not leave the template placeholders.

Risk Analysts Debate History:
{history}

---

Be decisive only after the DB checklist. Ground every conclusion in holdings, trade history, past decisions, and analyst evidence."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
