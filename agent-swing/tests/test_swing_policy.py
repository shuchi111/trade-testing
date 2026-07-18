"""
Unit tests for swing trading policy strings and prompt wiring.

No LLM or network — verifies constants and that key agent nodes reference them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tradingagents.agents.analysts.market_analyst as ma
import tradingagents.agents.managers.portfolio_manager as pm
import tradingagents.agents.trader.trader as tr
from tradingagents.agents.utils import swing_policy as sp
from tradingagents.agents.utils.swing_policy import (
    SWING_DEBATE_REMINDER,
    SWING_MANAGERS_BLOCK,
    SWING_MARKET_ANALYST_INSTRUCTIONS,
)
from tradingagents.agents.researchers import bull_researcher as br
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


class TestSwingPolicyConstants:
    def test_debate_reminder_covers_swing_and_basis(self):
        low = SWING_DEBATE_REMINDER.lower()
        assert "swing" in low
        assert "average entry" in low

    def test_market_instructions_include_weekly_candles(self):
        low = SWING_MARKET_ANALYST_INSTRUCTIONS.lower()
        assert "weekly" in low
        assert SWING_DEBATE_REMINDER in SWING_MARKET_ANALYST_INSTRUCTIONS

    def test_managers_block_requires_gtt_lines(self):
        assert "GTT target price" in SWING_MANAGERS_BLOCK
        assert "5% trailing stop price" in SWING_MANAGERS_BLOCK
        assert "THREE PERCENT" in SWING_MANAGERS_BLOCK
        assert "FIVE PERCENT" in SWING_MANAGERS_BLOCK
        assert "TWENTY-FIVE THOUSAND INR" in SWING_MANAGERS_BLOCK
        assert "NINETY CALENDAR DAYS" in SWING_MANAGERS_BLOCK

    def test_module_exports_expected_names(self):
        assert hasattr(sp, "SWING_DEBATE_REMINDER")
        assert hasattr(sp, "SWING_MARKET_ANALYST_INSTRUCTIONS")
        assert hasattr(sp, "SWING_MANAGERS_BLOCK")


class TestSwingPolicyPromptInjection:
    def test_market_analyst_system_message_includes_swing_block(self):
        assert ma.SWING_MARKET_ANALYST_INSTRUCTIONS is SWING_MARKET_ANALYST_INSTRUCTIONS
        src = Path(ma.__file__).read_text(encoding="utf-8")
        assert "SWING_MARKET_ANALYST_INSTRUCTIONS" in src

    def test_portfolio_manager_prompt_imports_managers_block(self):
        assert pm.SWING_MANAGERS_BLOCK is SWING_MANAGERS_BLOCK
        src = Path(pm.__file__).read_text(encoding="utf-8")
        assert "SWING_MANAGERS_BLOCK" in src

    def test_trader_includes_managers_block(self):
        assert tr.SWING_MANAGERS_BLOCK is SWING_MANAGERS_BLOCK

    def test_bull_researcher_includes_debate_reminder(self):
        assert br.SWING_DEBATE_REMINDER is SWING_DEBATE_REMINDER


class TestProviderKwargsGoogle:
    """Regression: Google must get the same HTTP timeout/retry defaults as other providers."""

    def test_google_includes_timeout_and_max_retries(self, monkeypatch):
        monkeypatch.delenv("LLM_HTTP_TIMEOUT", raising=False)
        monkeypatch.delenv("LLM_HTTP_MAX_RETRIES", raising=False)

        g = TradingAgentsGraph.__new__(TradingAgentsGraph)
        g.config = {**DEFAULT_CONFIG, "llm_provider": "google"}
        kwargs = g._get_provider_kwargs()
        assert kwargs["timeout"] == 300.0
        assert kwargs["max_retries"] == 5

    # testing
    def test_env_overrides_apply_to_google(self, monkeypatch):
        monkeypatch.setenv("LLM_HTTP_TIMEOUT", "120")
        monkeypatch.setenv("LLM_HTTP_MAX_RETRIES", "2")

        g = TradingAgentsGraph.__new__(TradingAgentsGraph)
        g.config = {**DEFAULT_CONFIG, "llm_provider": "google"}
        kwargs = g._get_provider_kwargs()
        assert kwargs["timeout"] == 120.0
        assert kwargs["max_retries"] == 2
