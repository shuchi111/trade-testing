from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# Stub heavy imports before write_recommendation_cache loads.
_stubs = {
    "tradingagents": types.ModuleType("tradingagents"),
    "tradingagents.default_config": types.ModuleType("tradingagents.default_config"),
    "tradingagents.graph": types.ModuleType("tradingagents.graph"),
    "tradingagents.graph.trading_graph": types.ModuleType("tradingagents.graph.trading_graph"),
    "yfinance": types.ModuleType("yfinance"),
}
_stubs["tradingagents.default_config"].DEFAULT_CONFIG = {}
_stubs["tradingagents.graph.trading_graph"].TradingAgentsGraph = MagicMock

for name, mod in _stubs.items():
    sys.modules.setdefault(name, mod)

from write_recommendation_cache import _positive  # noqa: E402


class PositiveLevelTests(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_positive(None))

    def test_zero_returns_none(self):
        self.assertIsNone(_positive(0))
        self.assertIsNone(_positive(0.0))

    def test_negative_returns_none(self):
        self.assertIsNone(_positive(-1.5))

    def test_positive_value_preserved(self):
        self.assertEqual(_positive(150.25), 150.25)

    def test_non_numeric_returns_none(self):
        self.assertIsNone(_positive("not-a-number"))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
