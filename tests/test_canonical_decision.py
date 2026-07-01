from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from canonical_decision import coerce_decision_for_holdings  # noqa: E402


class CoerceDecisionForHoldingsTests(unittest.TestCase):
    def test_sell_without_position_becomes_hold(self):
        self.assertEqual(coerce_decision_for_holdings("SELL", 0.0), "HOLD")

    def test_underweight_without_position_becomes_hold(self):
        self.assertEqual(coerce_decision_for_holdings("UNDERWEIGHT", 0.0), "HOLD")

    def test_sell_with_position_unchanged(self):
        self.assertEqual(coerce_decision_for_holdings("SELL", 10.0), "SELL")

    def test_buy_without_position_unchanged(self):
        self.assertEqual(coerce_decision_for_holdings("BUY", 0.0), "BUY")

    def test_hold_without_position_unchanged(self):
        self.assertEqual(coerce_decision_for_holdings("HOLD", 0.0), "HOLD")

    def test_tiny_position_treated_as_no_position(self):
        self.assertEqual(coerce_decision_for_holdings("SELL", 1e-8), "HOLD")


if __name__ == "__main__":
    unittest.main()
