from typing import Any, Dict

from langchain_openai import ChatOpenAI

from tradingagents.agents.utils.swing_policy import SWING_REFLECTION_SYSTEM


class Reflector:
    """Handles reflection on decisions and updating memory."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        return SWING_REFLECTION_SYSTEM

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        return (
            f"{current_state['market_report']}\n\n"
            f"{current_state['sentiment_report']}\n\n"
            f"{current_state['news_report']}\n\n"
            f"{current_state['fundamentals_report']}"
        )

    def _reflect_on_component(self, component_type: str, report: str, situation: str, returns_losses) -> str:
        messages = [
            ("system", self.reflection_system_prompt),
            ("human", f"Returns: {returns_losses}\n\nAnalysis/Decision: {report}\n\nObjective Market Reports for Reference: {situation}"),
        ]
        return self.quick_thinking_llm.invoke(messages).content

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]
        result = self._reflect_on_component("BULL", bull_debate_history, situation, returns_losses)
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]
        result = self._reflect_on_component("BEAR", bear_debate_history, situation, returns_losses)
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state["trader_investment_plan"]
        result = self._reflect_on_component("TRADER", trader_decision, situation, returns_losses)
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"]["judge_decision"]
        result = self._reflect_on_component("INVEST JUDGE", judge_decision, situation, returns_losses)
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_portfolio_manager(self, current_state, returns_losses, portfolio_manager_memory):
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"]["judge_decision"]
        result = self._reflect_on_component("PORTFOLIO MANAGER", judge_decision, situation, returns_losses)
        portfolio_manager_memory.add_situations([(situation, result)])
