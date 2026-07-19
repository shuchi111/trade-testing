"""Unit tests for Claude Skills Pack screeners + connected consensus (no network)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.agents.utils import claude_skills_pack as csp
from tradingagents.agents.utils.swing_policy import (
    DB_CONTEXT_ANALYSIS_MANDATE,
    SWING_MANAGERS_BLOCK,
    SWING_MARKET_ANALYST_INSTRUCTIONS,
)


def _synthetic_uptrend(n: int = 220, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    base = np.linspace(100, 140, n) + rng.normal(0, 0.6, n)
    # Insert a gap-up ~25 bars ago (PEAD event), then a red weekly-ish pullback, then recover.
    gap_i = n - 25
    base[gap_i:] += 8.0
    base[gap_i + 5 : gap_i + 12] -= np.linspace(0, 4, 7)
    high = base + rng.uniform(0.4, 1.2, n)
    low = base - rng.uniform(0.4, 1.2, n)
    open_ = base + rng.normal(0, 0.3, n)
    close = base.copy()
    # Force gap day open well above prior close.
    open_[gap_i] = close[gap_i - 1] * 1.05
    close[gap_i] = open_[gap_i] * 1.01
    high[gap_i] = max(high[gap_i], close[gap_i])
    vol = rng.integers(800_000, 1_200_000, n).astype(float)
    vol[-1] = vol[-20:-1].mean() * 1.8
    high[-1] = max(high[-1], close[-1] * 1.01)
    close[-1] = max(close[-1], high[-20:-1].max() * 1.001)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def _synthetic_nifty(n: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = np.linspace(22_000, 23_000, n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )


class TestScreeners:
    def test_each_screener_returns_score_and_signal(self):
        stock = _synthetic_uptrend()
        nifty = _synthetic_nifty()
        results = [
            csp.screen_vcp(stock),
            csp.screen_pead(stock, earnings_date=datetime(2025, 10, 1)),
            csp.screen_relative_strength(stock, nifty),
            csp.screen_volume_breakout(stock),
            csp.screen_momentum(stock),
        ]
        names = {r.name for r in results}
        assert names == {"VCP", "PEAD", "Relative Strength", "Volume Breakout", "Momentum"}
        for r in results:
            assert 0 <= r.score <= 100
            assert r.signal in {"bullish", "neutral", "bearish", "unavailable"}
            assert r.summary

    def test_pead_detects_gap_proxy_without_earnings_api(self):
        stock = _synthetic_uptrend()
        result = csp.screen_pead(stock)  # no earnings_date / no network ticker lookup
        assert result.name == "PEAD"
        assert result.signal != "unavailable"
        assert any("gap" in f.lower() or "stage=" in f for f in result.facts)

    def test_connect_links_screeners_and_builds_trade_plan(self):
        stock = _synthetic_uptrend()
        nifty = _synthetic_nifty()
        screeners = [
            csp.screen_vcp(stock),
            csp.screen_pead(stock),
            csp.screen_relative_strength(stock, nifty),
            csp.screen_volume_breakout(stock),
            csp.screen_momentum(stock),
        ]
        ta = csp.build_ta_snapshot(stock)
        nifty_info = csp.analyze_nifty_regime(nifty)
        vix = {"regime": "normal", "score": 60.0, "facts": ["India VIX=15 (normal)"]}
        connected = csp.connect_screener_results(screeners, ta=ta, nifty=nifty_info, vix=vix)
        assert "consensus_score" in connected
        assert connected["trade_plan"]["stance"]
        assert connected["trade_plan"]["risk_reward"] >= 1.0
        assert isinstance(connected["links"], list)

    def test_format_block_includes_all_sections(self):
        stock = _synthetic_uptrend()
        nifty = _synthetic_nifty()
        pack = {
            "ticker": "TEST.NS",
            "ok": True,
            "errors": [],
            "screeners": [
                csp.screen_vcp(stock),
                csp.screen_pead(stock),
                csp.screen_relative_strength(stock, nifty),
                csp.screen_volume_breakout(stock),
                csp.screen_momentum(stock),
            ],
            "ta": csp.build_ta_snapshot(stock),
            "nifty": csp.analyze_nifty_regime(nifty),
            "vix": {"regime": "calm", "score": 75.0, "facts": ["India VIX=12 (calm)"]},
            "connected": {},
        }
        pack["connected"] = csp.connect_screener_results(
            pack["screeners"], ta=pack["ta"], nifty=pack["nifty"], vix=pack["vix"]
        )
        text = csp.format_claude_skills_pack_block(pack)
        assert "CLAUDE SKILLS PACK" in text
        assert "5 SCREENERS" in text
        assert "CONNECTED CONSENSUS" in text
        assert "VCP" in text and "PEAD" in text
        assert "Relative Strength" in text
        assert "Volume Breakout" in text and "Momentum" in text
        assert "Trade plan stance" in text
        assert "PLAD" not in text


class TestPolicyWiring:
    def test_mandate_requires_skills_observe(self):
        assert "CLAUDE SKILLS PACK" in DB_CONTEXT_ANALYSIS_MANDATE
        assert "VCP" in DB_CONTEXT_ANALYSIS_MANDATE
        assert "PEAD" in DB_CONTEXT_ANALYSIS_MANDATE
        assert "PLAD" not in DB_CONTEXT_ANALYSIS_MANDATE

    def test_managers_and_market_mention_skills(self):
        assert "CLAUDE SKILLS PACK" in SWING_MANAGERS_BLOCK
        assert "PEAD" in SWING_MANAGERS_BLOCK
        assert "skills" in SWING_MARKET_ANALYST_INSTRUCTIONS.lower()


class TestMarketAnalystExcerpt:
    def test_skills_excerpt_helper(self):
        from tradingagents.agents.utils.claude_skills_pack import skills_observe_excerpt

        ctx = (
            "noise\n"
            "=== CLAUDE SKILLS PACK (observe BEFORE any Buy/Sell/Hold signal) ===\n"
            "[PEAD] score=70\n"
            "=== END CLAUDE SKILLS PACK ===\n"
            "tail"
        )
        excerpt = skills_observe_excerpt(ctx)
        assert "PEAD" in excerpt
        assert "END CLAUDE SKILLS PACK" in excerpt

    def test_skills_observe_excerpt_shared_helper(self):
        from tradingagents.agents.utils.claude_skills_pack import skills_observe_excerpt

        missing = skills_observe_excerpt("no pack here").lower()
        assert "missing" in missing or "gap" in missing
        packed = "=== CLAUDE SKILLS PACK ===\n[VCP]\n=== END CLAUDE SKILLS PACK ==="
        assert "VCP" in skills_observe_excerpt(packed)
