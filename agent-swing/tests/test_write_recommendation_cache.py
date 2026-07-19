"""Unit tests for write_recommendation_cache.

Uses unittest.mock to fake psycopg2 — no real database needed.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

# Stub heavy imports before the module loads.
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

from write_recommendation_cache import (
    upsert_cache_row,
    _is_buy_signal,
    _adjust_to_last_trading_day,
    _extract_signal_metrics,
    run_single_recommendation,
)


# ---------------------------------------------------------------------------
# _adjust_to_last_trading_day
# ---------------------------------------------------------------------------


class TestAdjustToLastTradingDay(unittest.TestCase):
    def test_monday_unchanged(self):
        # 2024-06-03 is a Monday
        self.assertEqual(_adjust_to_last_trading_day(date(2024, 6, 3)), date(2024, 6, 3))

    def test_friday_unchanged(self):
        self.assertEqual(_adjust_to_last_trading_day(date(2024, 5, 31)), date(2024, 5, 31))

    def test_saturday_rolls_to_friday(self):
        # 2024-06-01 is Saturday → should roll to Friday 2024-05-31
        self.assertEqual(_adjust_to_last_trading_day(date(2024, 6, 1)), date(2024, 5, 31))

    def test_sunday_rolls_to_friday(self):
        # 2024-06-02 is Sunday → should roll to Friday 2024-05-31
        self.assertEqual(_adjust_to_last_trading_day(date(2024, 6, 2)), date(2024, 5, 31))


# ---------------------------------------------------------------------------
# _is_buy_signal
# ---------------------------------------------------------------------------


class TestIsBuySignal(unittest.TestCase):
    def test_exact_buy(self):
        self.assertTrue(_is_buy_signal("BUY"))

    def test_exact_overweight(self):
        self.assertTrue(_is_buy_signal("OVERWEIGHT"))

    def test_lowercase(self):
        self.assertTrue(_is_buy_signal("buy"))

    def test_prefixed_format(self):
        self.assertTrue(_is_buy_signal("RECOMMENDATION: BUY"))

    def test_equals_format(self):
        self.assertTrue(_is_buy_signal("ACTION=BUY"))

    def test_first_word_buy(self):
        self.assertTrue(_is_buy_signal("BUY **strong**"))

    def test_hold_is_not_buy(self):
        self.assertFalse(_is_buy_signal("HOLD"))

    def test_sell_is_not_buy(self):
        self.assertFalse(_is_buy_signal("SELL"))

    def test_none_is_not_buy(self):
        self.assertFalse(_is_buy_signal(None))

    def test_empty_is_not_buy(self):
        self.assertFalse(_is_buy_signal(""))


# ---------------------------------------------------------------------------
# _extract_signal_metrics — positive-only guard for parsed levels
# ---------------------------------------------------------------------------


class TestExtractSignalMetrics(unittest.TestCase):
    def test_literal_zero_levels_fall_back_and_never_store_zero(self):
        # LLM emitted "0" for both target and stop (typical for a SELL/HOLD).
        metrics = _extract_signal_metrics(
            decision="SELL",
            final_trade_decision="Target price: 0\nStop loss: 0\nConfidence: 92%",
            reference_price=1557.88,
        )
        # Stop must fall back to the 5% guard, not 0.
        self.assertAlmostEqual(metrics["stop_loss_price"], 1557.88 * 0.95, places=4)
        # Risk must be the small 5% distance, NOT entry - 0 = entry.
        self.assertAlmostEqual(metrics["risk_amount"], 1557.88 * 0.05, places=4)
        # No target for a SELL → None (not 0), and reward/RR stay None.
        self.assertIsNone(metrics["target_price"])
        self.assertIsNone(metrics["reward_amount"])
        self.assertIsNone(metrics["risk_reward_ratio"])
        self.assertEqual(metrics["ai_confidence_pct"], 92.0)
        # Critically, nothing is a stored 0.
        for key in ("target_price", "stop_loss_price", "risk_amount", "reward_amount"):
            self.assertNotEqual(metrics[key], 0)

    def test_negative_level_treated_as_missing(self):
        metrics = _extract_signal_metrics(
            decision="SELL",
            final_trade_decision="Stop loss: -50",
            reference_price=1000.0,
        )
        self.assertAlmostEqual(metrics["stop_loss_price"], 950.0, places=4)
        self.assertAlmostEqual(metrics["risk_amount"], 50.0, places=4)

    def test_real_parsed_levels_are_preserved(self):
        metrics = _extract_signal_metrics(
            decision="BUY",
            final_trade_decision="Target: 1680\nStop loss: 1496\nConfidence 86%",
            reference_price=1574.82,
        )
        self.assertAlmostEqual(metrics["target_price"], 1680.0, places=4)
        self.assertAlmostEqual(metrics["stop_loss_price"], 1496.0, places=4)
        self.assertAlmostEqual(metrics["risk_amount"], 1574.82 - 1496.0, places=4)
        self.assertAlmostEqual(metrics["reward_amount"], 1680.0 - 1574.82, places=4)
        self.assertEqual(metrics["ai_confidence_pct"], 86.0)

    def test_buy_derives_target_from_1_5r_when_absent(self):
        metrics = _extract_signal_metrics(
            decision="BUY",
            final_trade_decision="No explicit levels here",
            reference_price=100.0,
        )
        # Stop fallback = 95, risk = 5, target = 100 + 5 * 1.5 = 107.5
        self.assertAlmostEqual(metrics["stop_loss_price"], 95.0, places=4)
        self.assertAlmostEqual(metrics["target_price"], 107.5, places=4)
        self.assertAlmostEqual(metrics["risk_reward_ratio"], 1.5, places=4)


# ---------------------------------------------------------------------------
# upsert_cache_row
# ---------------------------------------------------------------------------


class TestInsertCacheRow(unittest.TestCase):
    """Tests for upsert_cache_row (append-only INSERT)."""

    def _make_conn(self) -> MagicMock:
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cur)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        return conn, cur

    def test_inserts_with_correct_params(self):
        conn, cur = self._make_conn()
        upsert_cache_row(
            conn,
            ticker="tcs.ns",
            trade_date="2024-06-01",
            decision="BUY",
            final_trade_decision="FINAL: BUY",
            reference_price=3500.0,
            holding_quantity=10.0,
            holding_avg_entry=3400.0,
            source="test",
        )
        # BUY triggers three inserts: cache + buy_signals + history
        self.assertEqual(cur.execute.call_count, 3)

        # First call: ai_recommendation_cache
        first_sql = cur.execute.call_args_list[0][0][0]
        first_params = cur.execute.call_args_list[0][0][1]
        self.assertIn("INSERT INTO ai_recommendation_cache", first_sql)
        self.assertNotIn("ON CONFLICT", first_sql)
        self.assertEqual(first_params[0], "TCS.NS")
        self.assertEqual(first_params[2], "BUY")

        # Second call: ai_buy_signals
        second_sql = cur.execute.call_args_list[1][0][0]
        second_params = cur.execute.call_args_list[1][0][1]
        self.assertIn("INSERT INTO ai_buy_signals", second_sql)
        self.assertEqual(second_params[0], "TCS.NS")
        self.assertEqual(second_params[1], "2024-06-01")
        self.assertEqual(second_params[2], "BUY")
        self.assertEqual(second_params[3], 3500.0)

    def test_hold_does_not_insert_buy_signal(self):
        conn, cur = self._make_conn()
        upsert_cache_row(
            conn,
            ticker="TCS.NS",
            trade_date="2024-06-01",
            decision="HOLD",
            final_trade_decision="",
            reference_price=None,
            holding_quantity=0,
            holding_avg_entry=0,
            source="cron",
        )
        # HOLD inserts cache + history (no buy_signals).
        self.assertEqual(cur.execute.call_count, 2)
        self.assertIn("ai_recommendation_cache", cur.execute.call_args_list[0][0][0])
        all_sql = " ".join(c[0][0] for c in cur.execute.call_args_list)
        self.assertNotIn("ai_buy_signals", all_sql)
        conn.commit.assert_called_once()

    def test_empty_decision_no_buy_signal(self):
        conn, cur = self._make_conn()
        upsert_cache_row(
            conn,
            ticker="TCS.NS",
            trade_date="2024-06-01",
            decision=None,
            final_trade_decision=None,
            reference_price=100.0,
            holding_quantity=0,
            holding_avg_entry=0,
            source="test",
        )
        # Empty/unknown decision inserts cache + history (no buy_signals).
        self.assertEqual(cur.execute.call_count, 2)
        cache_params = cur.execute.call_args_list[0][0][1]
        self.assertEqual(cache_params[2], "")
        self.assertEqual(cache_params[3], "")
        all_sql = " ".join(c[0][0] for c in cur.execute.call_args_list)
        self.assertNotIn("ai_buy_signals", all_sql)

    def test_overweight_also_triggers_buy_signal(self):
        conn, cur = self._make_conn()
        upsert_cache_row(
            conn,
            ticker="INFY.NS",
            trade_date="2024-06-01",
            decision="OVERWEIGHT",
            final_trade_decision="",
            reference_price=1400.0,
            holding_quantity=0,
            holding_avg_entry=0,
            source="cron",
        )
        # OVERWEIGHT inserts cache + buy_signals + history.
        self.assertEqual(cur.execute.call_count, 3)
        buy_sql = cur.execute.call_args_list[1][0][0]
        self.assertIn("ai_buy_signals", buy_sql)


# ---------------------------------------------------------------------------
# run_single_recommendation error paths
# ---------------------------------------------------------------------------


class TestRunSingleRecommendationErrors(unittest.TestCase):
    def test_missing_api_key(self):
        with patch.dict(
            os.environ,
            {
                "Z_API_KEY": "",
                "GLM_API_KEY": "",
                "ANTHROPIC_AUTH_TOKEN": "",
                "ANTHROPIC_API_KEY": "",
            },
        ):
            result = run_single_recommendation(
                ticker="TCS.NS",
                trade_date="2024-06-01",
            )
        self.assertFalse(result["ok"])
        self.assertIn("API_KEY", result["error"])

    def test_missing_database_url(self):
        with patch.dict(os.environ, {"Z_API_KEY": "test", "DATABASE_URL": "", "DIRECT_URL": ""}):
            result = run_single_recommendation(
                ticker="TCS.NS",
                trade_date="2024-06-01",
            )
        self.assertFalse(result["ok"])
        self.assertIn("DIRECT_URL", result["error"])

    def test_negative_holding_quantity(self):
        result = run_single_recommendation(
            ticker="TCS.NS",
            trade_date="2024-06-01",
            holding_quantity=-5,
        )
        self.assertFalse(result["ok"])
        self.assertIn(">= 0", result["error"])

    def test_negative_holding_avg_entry(self):
        result = run_single_recommendation(
            ticker="TCS.NS",
            trade_date="2024-06-01",
            holding_avg_entry=-100,
        )
        self.assertFalse(result["ok"])
        self.assertIn(">= 0", result["error"])


if __name__ == "__main__":
    unittest.main()
