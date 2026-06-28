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

from tradingagents.dataflows.errors import StaleMarketDataError
from tradingagents.dataflows.market_data_validator import (
    format_market_snapshot,
    require_fresh_market_snapshot,
    verified_market_snapshot,
)


def _install_yfinance_stub(history: pd.DataFrame):
    class FakeTicker:
        fast_info = {"currency": "INR", "exchange": "NSE"}

        def __init__(self, symbol: str):
            self.symbol = symbol

        def history(self, period: str):
            return history

    return patch.dict(sys.modules, {"yfinance": types.SimpleNamespace(Ticker=FakeTicker)})


class MarketDataValidatorTests(unittest.TestCase):
    def test_timezone_aware_latest_date_is_compared_in_utc(self):
        history = pd.DataFrame(
            {"Close": [100.0], "Volume": [1000]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-06-27 23:30:00", tz="America/New_York")]),
        )

        with _install_yfinance_stub(history):
            snapshot = verified_market_snapshot(
                "tcs.ns",
                "2026-06-28",
                max_stale_days=0,
            )

        self.assertEqual(snapshot.latest_date, "2026-06-28")
        self.assertEqual(snapshot.age_days, 0)
        self.assertFalse(snapshot.stale)

    def test_require_fresh_rejects_future_market_bar(self):
        history = pd.DataFrame(
            {"Close": [100.0], "Volume": [1000]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-06-29", tz="UTC")]),
        )

        with _install_yfinance_stub(history):
            with self.assertRaises(StaleMarketDataError):
                require_fresh_market_snapshot("TCS.NS", "2026-06-28")

    def test_format_market_snapshot_includes_freshness(self):
        text = format_market_snapshot(
            {
                "vendor": "yfinance",
                "ticker": "TCS.NS",
                "latest_close": 100.0,
                "latest_date": "2026-06-28",
                "latest_volume": 1000,
                "stale": False,
                "stale_reason": "",
            }
        )

        self.assertIn("TCS.NS close 100.0", text)
        self.assertIn("freshness fresh", text)


if __name__ == "__main__":
    unittest.main()
