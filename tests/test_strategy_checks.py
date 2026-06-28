from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tradingagents.agents.utils.strategy_checks import build_minervini_evidence


def _install_yfinance_stub(history: pd.DataFrame):
    class FakeTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def history(self, period: str):
            return history

    return patch.dict(sys.modules, {"yfinance": types.SimpleNamespace(Ticker=FakeTicker)})


class StrategyChecksTests(unittest.TestCase):
    def test_minervini_rejects_199_close_boundary(self):
        history = pd.DataFrame(
            {"Close": [100.0] * 199, "Volume": [1000] * 199},
            index=pd.date_range("2025-01-01", periods=199),
        )

        with _install_yfinance_stub(history):
            with self.assertRaisesRegex(ValueError, "at least 200 closes"):
                build_minervini_evidence("TCS.NS")

    def test_minervini_returns_eight_threshold_checks(self):
        closes = [float(i) for i in range(100, 352)]
        history = pd.DataFrame(
            {"Close": closes, "Volume": [1000] * len(closes)},
            index=pd.date_range("2025-01-01", periods=len(closes)),
        )

        with _install_yfinance_stub(history):
            evidence = build_minervini_evidence("tcs.ns")

        self.assertEqual(evidence.ticker, "TCS.NS")
        self.assertEqual(evidence.total_count, 8)
        self.assertEqual(len(evidence.lines), 8)


if __name__ == "__main__":
    unittest.main()
