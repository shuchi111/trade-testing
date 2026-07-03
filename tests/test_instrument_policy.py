from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import instrument_policy


class InstrumentPolicyTests(unittest.TestCase):
    def test_fractional_tickers(self):
        self.assertTrue(instrument_policy.is_fractional_ticker("BTC-USD"))
        self.assertTrue(instrument_policy.is_fractional_ticker("eth-usd"))
        self.assertFalse(instrument_policy.is_fractional_ticker("TCS.NS"))

    def test_quote_to_inr_crypto(self):
        with unittest.mock.patch.object(instrument_policy, "usd_inr_rate", return_value=80.0):
            self.assertEqual(
                instrument_policy.quote_to_inr("BTC-USD", 1000.0, "USD"),
                80_000.0,
            )

    def test_quote_to_inr_nse_unchanged(self):
        self.assertEqual(
            instrument_policy.quote_to_inr("TCS.NS", 3500.0, "INR"),
            3500.0,
        )

    def test_size_buy_quantity_fractional(self):
        qty = instrument_policy.size_buy_quantity(
            buy_value_inr=10_000.0,
            price_inr=5_500_000.0,
            fractional=True,
        )
        self.assertAlmostEqual(qty, 0.001818, places=6)

    def test_size_buy_quantity_whole_share(self):
        qty = instrument_policy.size_buy_quantity(
            buy_value_inr=10_000.0,
            price_inr=3_000.0,
            fractional=False,
        )
        self.assertEqual(qty, 3.0)


if __name__ == "__main__":
    unittest.main()
