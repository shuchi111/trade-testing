"""Tests for confidence-proportional BUY sizing (₹25k is a ceiling)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
SWING_DIR = Path(__file__).resolve().parents[1] / "agent-swing"


class ConfidenceSizingAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(AGENT_DIR))

    def test_default_min_confidence_is_80(self):
        import trading_constraints as tc

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MIN_AI_CONFIDENCE_PCT", None)
            self.assertEqual(tc.min_ai_confidence_pct(), 80.0)

    def test_80_starts_at_half_cap_12500(self):
        import trading_constraints as tc

        with patch.dict(
            os.environ,
            {"MIN_AI_CONFIDENCE_PCT": "80", "CONFIDENCE_AT_MIN_SCALE": "0.5"},
            clear=False,
        ):
            budget = tc.sized_buy_budget_inr(
                cash_available=100_000,
                room_to_cap=25_000,
                confidence_pct=80,
            )
            self.assertAlmostEqual(tc.confidence_buy_scale(80), 0.5)
        self.assertAlmostEqual(budget, 12_500.0)

    def test_full_confidence_uses_full_room(self):
        import trading_constraints as tc

        with patch.dict(os.environ, {"MIN_AI_CONFIDENCE_PCT": "80"}, clear=False):
            budget = tc.sized_buy_budget_inr(
                cash_available=100_000,
                room_to_cap=25_000,
                confidence_pct=100,
            )
        self.assertAlmostEqual(budget, 25_000.0)

    def test_90_confidence_mid_remap(self):
        import trading_constraints as tc

        # 80→0.5, 100→1.0 ⇒ 90 → 0.75 → ₹18,750
        with patch.dict(
            os.environ,
            {"MIN_AI_CONFIDENCE_PCT": "80", "CONFIDENCE_AT_MIN_SCALE": "0.5"},
            clear=False,
        ):
            budget = tc.sized_buy_budget_inr(
                cash_available=100_000,
                room_to_cap=25_000,
                confidence_pct=90,
            )
        self.assertAlmostEqual(budget, 18_750.0)

    def test_missing_confidence_skips(self):
        import trading_constraints as tc

        with patch.dict(os.environ, {"CONFIDENCE_MISSING_SCALE": "0"}, clear=False):
            budget = tc.sized_buy_budget_inr(
                cash_available=100_000,
                room_to_cap=25_000,
                confidence_pct=None,
            )
        self.assertEqual(budget, 0.0)

    def test_below_min_confidence_zero_budget(self):
        import trading_constraints as tc

        with patch.dict(os.environ, {"MIN_AI_CONFIDENCE_PCT": "80"}):
            scale = tc.confidence_buy_scale(70)
            budget = tc.sized_buy_budget_inr(
                cash_available=100_000,
                room_to_cap=25_000,
                confidence_pct=70,
            )
        self.assertEqual(scale, 0.0)
        self.assertEqual(budget, 0.0)

    def test_cash_ceiling_still_applies(self):
        import trading_constraints as tc

        with patch.dict(os.environ, {"MIN_AI_CONFIDENCE_PCT": "80"}):
            budget = tc.sized_buy_budget_inr(
                cash_available=8_000,
                room_to_cap=25_000,
                confidence_pct=100,
            )
        self.assertAlmostEqual(budget, 8_000.0)


class ConfidenceSizingSwingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(AGENT_DIR) in sys.path:
            sys.path.remove(str(AGENT_DIR))
        sys.path.insert(0, str(SWING_DIR))
        sys.modules.pop("trading_constraints", None)

    def test_swing_80_is_12500(self):
        import trading_constraints as tc

        with patch.dict(
            os.environ,
            {"MIN_AI_CONFIDENCE_PCT": "80", "CONFIDENCE_AT_MIN_SCALE": "0.5"},
            clear=False,
        ):
            budget = tc.sized_buy_budget_inr(
                cash_available=100_000,
                room_to_cap=25_000,
                confidence_pct=80,
            )
        self.assertAlmostEqual(budget, 12_500.0)

    def test_swing_70_skipped(self):
        import trading_constraints as tc

        with patch.dict(os.environ, {"MIN_AI_CONFIDENCE_PCT": "80"}):
            self.assertEqual(tc.confidence_buy_scale(70), 0.0)


if __name__ == "__main__":
    unittest.main()
