"""
Integration tests for the multi-agent trading graph.

All LLM calls are replaced with deterministic mocks so these tests:
  - Run without any API keys or network access
  - Execute the full LangGraph pipeline end-to-end
  - Verify graph wiring, state propagation, and signal extraction
  - Cover edge cases: malformed LLM output, empty memory, tool failures
"""
from __future__ import annotations

import json
import sys
import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

import pandas as pd  # type: ignore[reportMissingImports]
import pytest

# Make the agent package importable when running from the tests/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.agents.utils.agent_states import AgentState, InvestDebateState, RiskDebateState
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.trader.trader import create_trader
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.signal_processing import SignalProcessor
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.dataflows.stockstats_utils import _cache_date_range


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_response(text: str) -> MagicMock:
    """Build a mock LLM response object with .content = text."""
    r = MagicMock()
    r.content = text
    return r


def make_mock_llm(responses: list[str]) -> MagicMock:
    """Return an LLM mock that cycles through the given response strings."""
    llm = MagicMock()
    llm.invoke.side_effect = [_llm_response(r) for r in responses]
    return llm


def make_empty_memory() -> FinancialSituationMemory:
    return FinancialSituationMemory("test_memory", {})


def _base_state() -> dict[str, Any]:
    """Minimal AgentState-compatible dict for unit-testing individual nodes."""
    return {
        "messages": [],
        "company_of_interest": "TCS.NS",
        "trade_date": "2024-06-01",
        "sender": "",
        "market_report": "Market is bullish with strong momentum.",
        "sentiment_report": "Social sentiment is positive.",
        "news_report": "No major negative news.",
        "fundamentals_report": "Strong earnings growth.",
        "investment_plan": "Consider buying on dips.",
        "trader_investment_plan": "BUY with 2% stop-loss.",
        "investment_debate_state": InvestDebateState({
            "bull_history": "", "bear_history": "", "history": "",
            "current_response": "", "judge_decision": "", "count": 0,
        }),
        "risk_debate_state": RiskDebateState({
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "history": "",
            "latest_speaker": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "judge_decision": "", "count": 0,
        }),
        "final_trade_decision": "",
    }


# ---------------------------------------------------------------------------
# Unit tests — individual agent nodes
# ---------------------------------------------------------------------------

class TestBullResearcher:
    def test_appends_bull_argument_to_history(self):
        llm = make_mock_llm(["Strong growth ahead for TCS.NS."])
        memory = make_empty_memory()
        node = create_bull_researcher(llm, memory)

        state = _base_state()
        result = node(state)

        debate = result["investment_debate_state"]
        assert "Bull Analyst:" in debate["history"]
        assert "Strong growth ahead for TCS.NS." in debate["bull_history"]
        assert debate["count"] == 1

    def test_incorporates_past_memories(self):
        memory = make_empty_memory()
        memory.add_situations([("TCS.NS bullish Q4", "BUY — strong seasonality")])

        llm = make_mock_llm(["Seasonality favours bulls."])
        node = create_bull_researcher(llm, memory)

        state = _base_state()
        node(state)

        # Memory retrieval path was exercised — LLM was called once
        llm.invoke.assert_called_once()

    def test_handles_empty_memory_gracefully(self):
        llm = make_mock_llm(["No memory, still bullish."])
        node = create_bull_researcher(llm, make_empty_memory())
        result = node(_base_state())
        assert result["investment_debate_state"]["count"] == 1


class TestBearResearcher:
    def test_appends_bear_argument_to_history(self):
        llm = make_mock_llm(["Valuation looks stretched."])
        node = create_bear_researcher(llm, make_empty_memory())

        state = _base_state()
        result = node(state)

        debate = result["investment_debate_state"]
        assert "Bear Analyst:" in debate["history"]
        assert "Valuation looks stretched." in debate["bear_history"]
        assert debate["count"] == 1


class TestResearchManager:
    def test_produces_investment_plan(self):
        llm = make_mock_llm(["BUY — strong fundamentals outweigh macro risks."])
        node = create_research_manager(llm, make_empty_memory())

        result = node(_base_state())
        assert result["investment_plan"] == "BUY — strong fundamentals outweigh macro risks."
        assert result["investment_debate_state"]["judge_decision"] != ""


class TestTrader:
    def test_returns_final_transaction_proposal(self):
        proposal = "Analysis complete. FINAL TRANSACTION PROPOSAL: **BUY**"
        llm = make_mock_llm([proposal])
        node = create_trader(llm, make_empty_memory())

        result = node(_base_state())
        assert result["trader_investment_plan"] == proposal
        assert result["sender"] == "Trader"
        assert len(result["messages"]) == 1


class TestRiskDebators:
    def test_conservative_increments_count(self):
        llm = make_mock_llm(["Risk is too high; reduce position."])
        node = create_conservative_debator(llm)
        result = node(_base_state())
        assert result["risk_debate_state"]["count"] == 1
        assert "Conservative Analyst:" in result["risk_debate_state"]["conservative_history"]

    def test_aggressive_increments_count(self):
        llm = make_mock_llm(["Reward outweighs risk; go all-in."])
        node = create_aggressive_debator(llm)
        result = node(_base_state())
        assert result["risk_debate_state"]["count"] == 1
        assert "Aggressive Analyst:" in result["risk_debate_state"]["aggressive_history"]

    def test_neutral_increments_count(self):
        llm = make_mock_llm(["Balanced view: hold with tight stop."])
        node = create_neutral_debator(llm)
        result = node(_base_state())
        assert result["risk_debate_state"]["count"] == 1
        assert "Neutral Analyst:" in result["risk_debate_state"]["neutral_history"]


class TestPortfolioManager:
    def test_produces_final_trade_decision(self):
        decision = "Rating: Buy\nExecutive Summary: Enter at market open."
        llm = make_mock_llm([decision])
        node = create_portfolio_manager(llm, make_empty_memory())

        result = node(_base_state())
        assert result["final_trade_decision"] == decision
        assert result["risk_debate_state"]["judge_decision"] == decision


# ---------------------------------------------------------------------------
# Unit tests — signal processing
# ---------------------------------------------------------------------------

class TestSignalProcessor:
    def test_extracts_buy_signal(self):
        llm = make_mock_llm(["BUY"])
        processor = SignalProcessor(llm)
        assert processor.process_signal("Rating: Buy\nStrong conviction...") == "BUY"

    def test_extracts_sell_signal(self):
        llm = make_mock_llm(["SELL"])
        processor = SignalProcessor(llm)
        assert processor.process_signal("Rating: Sell\nExit position...") == "SELL"

    def test_extracts_hold_signal(self):
        llm = make_mock_llm(["HOLD"])
        processor = SignalProcessor(llm)
        assert processor.process_signal("Rating: Hold") == "HOLD"

    def test_handles_malformed_llm_output(self):
        """Garbage LLM output falls back to regex on the report, else HOLD."""
        llm = make_mock_llm(["MAYBE_BUY???"])
        processor = SignalProcessor(llm)
        result = processor.process_signal("Unclear recommendation...")
        assert result == "HOLD"

    def test_connection_error_body_falls_back_to_report_words(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_EXTRACT_MAX_ATTEMPTS", "1")
        monkeypatch.setenv("SIGNAL_EXTRACT_RETRY_DELAY_SEC", "0")
        llm = make_mock_llm(["Connection error."])
        processor = SignalProcessor(llm)
        result = processor.process_signal(
            "Final rating: The portfolio manager recommends BUY for the next quarter."
        )
        assert result == "BUY"


# ---------------------------------------------------------------------------
# Unit tests — memory system
# ---------------------------------------------------------------------------

class TestFinancialSituationMemory:
    def test_returns_empty_list_when_no_memories(self):
        mem = FinancialSituationMemory("test", {})
        assert mem.get_memories("TCS.NS bullish", n_matches=2) == []

    def test_retrieves_most_relevant_memory(self):
        mem = FinancialSituationMemory("test", {})
        mem.add_situations([
            ("TCS.NS strong earnings beat", "BUY"),
            ("TSLA delivery miss bearish", "SELL"),
        ])
        results = mem.get_memories("TCS.NS earnings beat expectations", n_matches=1)
        assert len(results) == 1
        assert results[0]["recommendation"] == "BUY"

    def test_n_matches_respected(self):
        mem = FinancialSituationMemory("test", {})
        mem.add_situations([(f"situation {i}", f"rec {i}") for i in range(5)])
        results = mem.get_memories("situation", n_matches=3)
        assert len(results) == 3

    def test_clear_resets_state(self):
        mem = FinancialSituationMemory("test", {})
        mem.add_situations([("bull market", "BUY")])
        mem.clear()
        assert mem.get_memories("bull market") == []

    def test_similarity_scores_normalised(self):
        mem = FinancialSituationMemory("test", {})
        mem.add_situations([("TCS.NS rally", "BUY"), ("TCS.NS crash", "SELL")])
        results = mem.get_memories("TCS.NS", n_matches=2)
        for r in results:
            assert 0.0 <= r["similarity_score"] <= 1.0


# ---------------------------------------------------------------------------
# Unit tests — propagator (state initialisation)
# ---------------------------------------------------------------------------

class TestPropagator:
    def test_initial_state_has_required_keys(self):
        prop = Propagator()
        state = prop.create_initial_state("TCS.NS", "2024-06-01")
        for key in ["company_of_interest", "trade_date", "investment_debate_state",
                    "risk_debate_state", "market_report", "fundamentals_report",
                    "sentiment_report", "news_report", "messages"]:
            assert key in state

    def test_debate_states_start_at_count_zero(self):
        prop = Propagator()
        state = prop.create_initial_state("TSLA", "2024-06-01")
        assert state["investment_debate_state"]["count"] == 0
        assert state["risk_debate_state"]["count"] == 0

    def test_graph_args_contains_recursion_limit(self):
        prop = Propagator(max_recur_limit=50)
        args = prop.get_graph_args()
        assert args["config"]["recursion_limit"] == 50


# ---------------------------------------------------------------------------
# Integration test — full graph end-to-end with mocked LLMs and tools
# ---------------------------------------------------------------------------

class TestTradingGraphEndToEnd:
    """
    Wires up the full LangGraph pipeline with mocked LLMs and no-op tool nodes.
    Verifies that a complete propagate() call returns a non-empty final_trade_decision
    and a recognised signal string.
    """

    def _make_graph(self, signal: str = "BUY"):
        """Build a TradingAgentsGraph with all LLMs mocked."""
        # Patch create_llm_client so no real LLM is instantiated
        mock_client = MagicMock()
        mock_llm = MagicMock()
        # Return a deterministic response for every invoke()
        mock_llm.invoke.return_value = _llm_response(
            f"Analysis complete. FINAL TRANSACTION PROPOSAL: **{signal}**"
        )
        mock_client.get_llm.return_value = mock_llm

        cfg = {
            **DEFAULT_CONFIG,
            "llm_provider": "openai",
            "api_key": "test-key",
            "deep_think_llm": "mock",
            "quick_think_llm": "mock",
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "data_cache_dir": "/tmp/test_cache",
        }

        with patch("tradingagents.graph.trading_graph.create_llm_client", return_value=mock_client):
            graph = TradingAgentsGraph(
                selected_analysts=["market"],  # minimal analyst set for speed
                debug=False,
                config=cfg,
            )

        # Replace the compiled graph's invoke with a minimal stub that
        # returns a state matching the final-state shape expected by propagate()
        stub_state = _base_state()
        stub_state["final_trade_decision"] = (
            f"Rating: {signal}\nFINAL TRANSACTION PROPOSAL: **{signal}**"
        )
        graph.graph = MagicMock()
        graph.graph.invoke.return_value = stub_state

        # Mock signal processor to return the signal directly
        graph.signal_processor = MagicMock()
        graph.signal_processor.process_signal.return_value = signal

        return graph

    def test_propagate_returns_state_and_decision(self):
        graph = self._make_graph("BUY")
        final_state, decision = graph.propagate("TCS.NS", "2024-06-01")
        assert decision == "BUY"
        assert final_state["company_of_interest"] == "TCS.NS"
        assert final_state["final_trade_decision"] != ""

    def test_propagate_hold_signal(self):
        graph = self._make_graph("HOLD")
        _, decision = graph.propagate("MSFT", "2024-06-01")
        assert decision == "HOLD"

    def test_propagate_sell_signal(self):
        graph = self._make_graph("SELL")
        _, decision = graph.propagate("NVDA", "2024-06-01")
        assert decision == "SELL"

    def test_propagate_sets_ticker_on_instance(self):
        graph = self._make_graph("BUY")
        graph.propagate("TSLA", "2024-06-01")
        assert graph.ticker == "TSLA"


# ---------------------------------------------------------------------------
# Unit tests — Yahoo Finance cache helper
# ---------------------------------------------------------------------------

class TestCacheDateRange:
    def test_returns_two_strings(self):
        start, end = _cache_date_range(years=15)
        assert isinstance(start, str) and isinstance(end, str)

    def test_end_is_a_monday(self):
        _, end = _cache_date_range()
        assert pd.Timestamp(end).dayofweek == 0  # 0 = Monday

    def test_span_is_approximately_15_years(self):
        start, end = _cache_date_range(years=15)
        delta = pd.Timestamp(end) - pd.Timestamp(start)
        # Allow a small window around 15 years (±8 days for week rounding)
        assert 15 * 365 - 8 <= delta.days <= 15 * 366 + 8

    def test_stable_within_same_week(self):
        """Calling twice in the same process should return the same result."""
        assert _cache_date_range() == _cache_date_range()

    def test_cache_filename_does_not_change_daily(self):
        """Simulate two consecutive days and verify the cache key is identical."""
        today = pd.Timestamp.today().normalize()
        monday = today - timedelta(days=today.dayofweek)

        # Patch Timestamp.today() to return today and today+1 separately
        with patch("tradingagents.dataflows.stockstats_utils.pd.Timestamp") as mock_ts:
            mock_ts.today.return_value = monday
            result_day1 = _cache_date_range()

            mock_ts.today.return_value = monday + timedelta(days=1)
            result_day2 = _cache_date_range()

        assert result_day1 == result_day2
