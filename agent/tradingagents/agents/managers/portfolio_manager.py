from __future__ import annotations

from typing import Any, Callable

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.swing_policy import SWING_MANAGERS_BLOCK


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

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        past_memory_str = "".join(rec["recommendation"] + "\n\n" for rec in past_memories)

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and \
deliver the final trading decision.

{SWING_MANAGERS_BLOCK}

{instrument_context}

---

Rating scale — use exactly one word for the Rating line (capitalized exactly as written):
Buy: Strong conviction to enter or add (only when not already fully sized)
Overweight: Favorable outlook, increase gradually when room under cap
Hold: Maintain, wait, or express bearish / avoid-entry view when no shares are held
Underweight: Reduce partly — only when an open position exists in portfolio context
Sell: Exit an open position fully — only when quantity held is greater than zero

Mandatory position rule: if CURRENT TICKER FOCUS or holdings summary shows no open \
position in this name, never use Sell or Underweight. Use Hold instead.

Context (must weight all of this; holdings define basis for percent math when applicable):
Current position and holdings summary: {portfolio_context}
Trader proposed plan (verbatim reference): {trader_plan}
Lessons from past decisions: {past_memory_str}

**Required Output Structure:**
1. **Rating**: State one of Buy / Overweight / Hold / Underweight / Sell.
2. **Executive Summary**: A concise action plan covering entry strategy, position \
   sizing, key risk levels, and time horizon.
3. **Portfolio and wallet observations**: What wallet cash, holdings, cap room, \
   active stops, live trades, and backtests imply for this rating.
4. **Investment Thesis**: Detailed reasoning anchored in the analysts' debate and past \
reflections.
5. **Backtest and live trade review**: What backtest results and live trade history imply.

Rating: Buy | Overweight | Hold | Underweight | Sell

Executive summary:
Action plan: prose for the week (one stance, no scalp churn).
Entry strategy: prose (add / trim / wait; levels in words if helpful).
Position sizing: prose (how heavy or light; respect swing, holdings basis, and the twenty-five thousand INR cap).
Key risk levels: prose (including the mandatory 5% trailing stop, wallet reserve, and thesis-break ideas).
Time horizon: prose (multi-week swing; not day-trade framing).

Portfolio and wallet observations:
Position status: state held quantity, average entry, days held, or explicitly confirm no open position.
Wallet and cap: cash, reserve, exposure, and room to add under the twenty-five thousand INR cap.
Risk controls: active or required 5% trailing stop and whether the rating increases or reduces risk.
Evidence check: how recent live trades, backtests, past AI recommendations, and holding P&L support or contradict the action.

Investment thesis: detailed reasoning in one or more paragraphs anchored in the debate, fundamentals, and market context; explicitly tie to holdings and average entry when provided.

Backtest and live trade review: summarise what the backtest strategy summary, backtest trade dates, and live trade history in portfolio context imply for this week's rating. Cite specific backtest entry/exit dates when warning against whipsaw. State days held and where the holding sits inside the ninety-day swing exit window when recommending any exit.

Weekly candles tie-in: how multi-week candle structure supports or contradicts the stance (pattern names in plain words).

GTT target price: INR value and one sentence (tie to >3% gain versus basis when recommending profit-taking).

GTT stop price: INR value and one sentence (use the mandatory 5% trailing stop distance, not a 10% discretionary stop).

Risk/reward ratio: numeric ratio using target reward divided by stop risk (example 1.50).

AI confidence: percentage from 0% to 100% and one short reason.

When you write the final answer, reproduce the section labels above verbatim (Rating:, Executive summary:, then Action plan:, Entry strategy:, Position sizing:, Key risk levels:, Time horizon:, then Portfolio and wallet observations:, Position status:, Wallet and cap:, Risk controls:, Evidence check:, then Investment thesis:, Backtest and live trade review:, Weekly candles tie-in:, GTT target price:, GTT stop price:, Risk/reward ratio:, AI confidence:) and fill in real content—do not leave the template placeholders.

Risk Analysts Debate History:
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts."""

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
