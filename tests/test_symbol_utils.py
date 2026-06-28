from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tradingagents.dataflows.symbol_utils import normalize_symbol, resolve_instrument_identity


def _install_yfinance_stub(info: dict):
    class FakeTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol
            self.info = info

    return patch.dict(sys.modules, {"yfinance": types.SimpleNamespace(Ticker=FakeTicker)})


class SymbolUtilsTests(unittest.TestCase):
    def tearDown(self):
        resolve_instrument_identity.cache_clear()

    def test_normalize_symbol_preserves_exchange_suffix(self):
        self.assertEqual(normalize_symbol("  tcs.ns "), "TCS.NS")
        self.assertEqual(normalize_symbol("vod.l"), "VOD.L")

    def test_resolve_identity_filters_empty_placeholder_values(self):
        info = {
            "longName": "Tata Consultancy Services",
            "shortName": "n/a",
            "sector": "Technology",
            "industry": "null",
            "exchange": "NSE",
            "currency": "INR",
            "quoteType": "EQUITY",
        }

        with _install_yfinance_stub(info):
            identity = resolve_instrument_identity("tcs.ns")

        self.assertEqual(identity["company_name"], "Tata Consultancy Services")
        self.assertEqual(identity["sector"], "Technology")
        self.assertEqual(identity["exchange"], "NSE")
        self.assertEqual(identity["currency"], "INR")
        self.assertNotIn("industry", identity)

    def test_resolve_identity_fail_open_on_vendor_error(self):
        class FailingTicker:
            @property
            def info(self):
                raise RuntimeError("vendor down")

        with patch.dict(sys.modules, {"yfinance": types.SimpleNamespace(Ticker=lambda symbol: FailingTicker())}):
            self.assertEqual(resolve_instrument_identity("TCS.NS"), {})


if __name__ == "__main__":
    unittest.main()
