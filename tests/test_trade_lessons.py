"""Unit tests for trade_lessons risk gates and lesson helpers."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import trade_lessons


class NormalizeTickerTests(unittest.TestCase):
    def test_accepts_nse_and_crypto(self):
        self.assertEqual(trade_lessons.normalize_ticker("tcs.ns"), "TCS.NS")
        self.assertEqual(trade_lessons.normalize_ticker("BTC-USD"), "BTC-USD")

    def test_rejects_invalid(self):
        with self.assertRaises(ValueError):
            trade_lessons.normalize_ticker("   ")
        with self.assertRaises(ValueError):
            trade_lessons.normalize_ticker("DROP TABLE;")


class MeaningfulLossTests(unittest.TestCase):
    def test_absolute_floor(self):
        with patch.object(trade_lessons, "min_loss_inr_for_cooldown", return_value=100.0):
            with patch.object(trade_lessons, "min_loss_pct_for_cooldown", return_value=0.05):
                self.assertTrue(trade_lessons._is_meaningful_loss(-150.0, 10_000.0))
                self.assertFalse(trade_lessons._is_meaningful_loss(-50.0, 10_000.0))

    def test_percentage_of_notional(self):
        with patch.object(trade_lessons, "min_loss_inr_for_cooldown", return_value=100.0):
            with patch.object(trade_lessons, "min_loss_pct_for_cooldown", return_value=0.05):
                # -80 INR on 1_000 notional = 8% -> meaningful via pct even below INR floor
                self.assertTrue(trade_lessons._is_meaningful_loss(-80.0, 1_000.0))
                # -40 INR on 1_000 = 4% -> below both thresholds
                self.assertFalse(trade_lessons._is_meaningful_loss(-40.0, 1_000.0))


class BuildRuleLessonTests(unittest.TestCase):
    def test_loss_lesson_mentions_cooldown(self):
        trade = {
            "ticker": "TCS.NS",
            "realized_pnl": -500.0,
            "outcome": "loss",
            "quantity": 2,
            "price": 3500.0,
            "trade_date": date(2026, 7, 10),
        }
        with patch.object(trade_lessons, "recent_loss_cooldown_days", return_value=10):
            situation, lesson = trade_lessons.build_rule_lesson(
                trade, entry_decision="BUY", report_snip="weekly uptrend"
            )
        self.assertIn("TCS.NS", situation)
        self.assertIn("MISTAKE LESSON", lesson)
        self.assertIn("Cool-off 10 days", lesson)


class RecentLossBlocksBuyTests(unittest.TestCase):
    def _conn_with_rows(self, rows):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        cur.fetchall.return_value = rows
        return conn, cur

    def test_blocks_when_absolute_loss_within_cooldown(self):
        conn, cur = self._conn_with_rows([(date(2026, 7, 8), -250.0, 5_000.0)])
        with patch.object(trade_lessons, "min_loss_inr_for_cooldown", return_value=100.0):
            with patch.object(trade_lessons, "min_loss_pct_for_cooldown", return_value=0.05):
                with patch.object(trade_lessons, "recent_loss_cooldown_days", return_value=10):
                    blocked, reason = trade_lessons.recent_loss_blocks_buy(
                        conn, "TCS.NS", as_of=date(2026, 7, 15)
                    )
        self.assertTrue(blocked)
        self.assertIn("recent_loss_cooldown", reason)
        self.assertIn("TCS.NS", reason)
        cur.execute.assert_called_once()

    def test_allows_when_loss_too_small(self):
        conn, _ = self._conn_with_rows([(date(2026, 7, 8), -20.0, 10_000.0)])
        with patch.object(trade_lessons, "min_loss_inr_for_cooldown", return_value=100.0):
            with patch.object(trade_lessons, "min_loss_pct_for_cooldown", return_value=0.05):
                blocked, reason = trade_lessons.recent_loss_blocks_buy(
                    conn, "TCS.NS", as_of=date(2026, 7, 15)
                )
        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    def test_rejects_bad_ticker(self):
        with self.assertRaises(ValueError):
            trade_lessons.recent_loss_blocks_buy(MagicMock(), "bad ticker!")


class PortfolioQualityGateTests(unittest.TestCase):
    def _conn_with_quality(self, winning, losing, gross_profit, gross_loss):
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        cur.fetchone.return_value = (winning, losing, gross_profit, gross_loss)
        return conn

    def test_blocks_low_win_rate_and_deep_negative_expectancy(self):
        # 1 win @ 100, 4 losses @ 500 each -> win_rate=20%, expectancy = -380 INR
        # At max_pos=25_000 -> expectancy_pct = -1.52% < -0.8%
        conn = self._conn_with_quality(1, 4, 100.0, 2_000.0)
        with patch.object(trade_lessons, "max_position_inr", return_value=25_000.0):
            with patch.object(trade_lessons, "quality_min_closed_trades", return_value=5):
                with patch.object(trade_lessons, "quality_win_rate_max_pct", return_value=35.0):
                    with patch.object(
                        trade_lessons, "quality_expectancy_pct_of_cap", return_value=-0.8
                    ):
                        blocked, reason = trade_lessons.portfolio_quality_blocks_new_risk(conn)
        self.assertTrue(blocked)
        self.assertIn("portfolio_quality_gate", reason)
        self.assertIn("expectancy_pct_of_cap", reason)

    def test_scales_with_larger_max_position(self):
        # Same book: expectancy -380 INR is only -0.076% of 500k -> should NOT block
        conn = self._conn_with_quality(1, 4, 100.0, 2_000.0)
        with patch.object(trade_lessons, "max_position_inr", return_value=500_000.0):
            with patch.object(trade_lessons, "quality_min_closed_trades", return_value=5):
                with patch.object(trade_lessons, "quality_win_rate_max_pct", return_value=35.0):
                    with patch.object(
                        trade_lessons, "quality_expectancy_pct_of_cap", return_value=-0.8
                    ):
                        blocked, reason = trade_lessons.portfolio_quality_blocks_new_risk(conn)
        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    def test_skips_when_too_few_closed_trades(self):
        conn = self._conn_with_quality(1, 2, 50.0, 900.0)
        with patch.object(trade_lessons, "quality_min_closed_trades", return_value=5):
            blocked, _ = trade_lessons.portfolio_quality_blocks_new_risk(conn)
        self.assertFalse(blocked)


class HarvestToBuyGateFlowTests(unittest.TestCase):
    """Lightweight integration: closed loss -> harvest path -> buy gate sees loss."""

    def test_closed_loss_then_buy_gate_blocks(self):
        trade = {
            "id": "trade-1",
            "ticker": "INFY.NS",
            "trade_date": date(2026, 7, 5),
            "quantity": 10,
            "price": 1500.0,
            "total_value": 15_000.0,
            "realized_pnl": -500.0,
            "outcome": "loss",
        }
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        cur.rowcount = 1

        with (
            patch.object(trade_lessons, "load_closed_sells_for_reflection", return_value=[trade]),
            patch.object(trade_lessons, "_prior_buy_decision", return_value="BUY"),
            patch.object(trade_lessons, "_prior_report_snippet", return_value=""),
            patch.object(trade_lessons, "ensure_lessons_table"),
        ):
            result = trade_lessons.harvest_lessons_from_closed_trades(conn, limit=10)

        self.assertEqual(result["written"], 1)
        self.assertEqual(result["closed_sells"], 1)

        gate_conn = MagicMock()
        gate_cur = MagicMock()
        gate_conn.cursor.return_value.__enter__.return_value = gate_cur
        gate_cur.fetchall.return_value = [(date(2026, 7, 5), -500.0, 15_000.0)]
        with patch.object(trade_lessons, "min_loss_inr_for_cooldown", return_value=100.0):
            with patch.object(trade_lessons, "min_loss_pct_for_cooldown", return_value=0.05):
                blocked, reason = trade_lessons.recent_loss_blocks_buy(
                    gate_conn, "INFY.NS", as_of=date(2026, 7, 12)
                )
        self.assertTrue(blocked)
        self.assertIn("INFY.NS", reason)


if __name__ == "__main__":
    unittest.main()
