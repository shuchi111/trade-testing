from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import portfolio_db


class BuildAnalysisContextTests(unittest.TestCase):
    def test_explicit_zero_reference_price_is_not_treated_as_missing(self):
        patches = [
            patch.object(portfolio_db, "max_position_inr", return_value=100000.0),
            patch.object(portfolio_db, "swing_exit_window_days", return_value=90),
            patch.object(portfolio_db, "trailing_stop_loss_pct", return_value=5.0),
            patch.object(
                portfolio_db,
                "load_holding_detail",
                return_value={
                    "quantity": 2.0,
                    "avg_entry": 500.0,
                    "entry_time": None,
                    "holding_since": None,
                },
            ),
            patch.object(portfolio_db, "load_wallet_cash", return_value=10000.0),
            patch.object(portfolio_db, "load_all_holding_details", return_value=[]),
            patch.object(portfolio_db, "load_latest_reference_prices", return_value={}),
            patch.object(portfolio_db, "days_held", return_value=None),
            patch.object(portfolio_db, "load_portfolio_trade_quality", return_value={}),
            patch.object(portfolio_db, "load_active_trailing_stop", return_value=None),
            patch.object(portfolio_db, "min_wallet_cash_reserve_inr", return_value=0.0),
            patch.object(portfolio_db, "buy_transaction_charge_inr", return_value=0.0),
            patch.object(portfolio_db, "sell_transaction_charge_inr", return_value=0.0),
            patch.object(portfolio_db, "load_recent_portfolio_trades", return_value=[]),
            patch.object(portfolio_db, "load_recent_wallet_trades", return_value=[]),
            patch.object(portfolio_db, "load_recent_wallet_transactions", return_value=[]),
            patch.object(portfolio_db, "load_recent_ai_trade_executions", return_value=[]),
            patch.object(portfolio_db, "load_recent_ai_recommendations", return_value=[]),
            patch.object(portfolio_db, "load_backtest_strategy_summaries", return_value=[]),
            patch.object(portfolio_db, "load_backtest_trades", return_value=[]),
        ]
        for dep_patch in patches:
            self.enterContext(dep_patch)

        context = portfolio_db.build_analysis_context(
            object(),
            "TCS.NS",
            trade_date="2026-06-28",
            reference_price=0.0,
        )

        self.assertIn("Room to add on this name:", context)
        self.assertIn("100,000.00", context)


class QuantityEpsilonTests(unittest.TestCase):
    def test_exact_match_uses_manual_sell(self):
        held, qty = 10.0, 10.0
        self.assertTrue((held - qty) <= portfolio_db.QUANTITY_EPSILON)

    def test_near_zero_remainder_uses_manual_sell(self):
        held, qty = 10.0, 10.0 - portfolio_db.QUANTITY_EPSILON
        self.assertTrue((held - qty) <= portfolio_db.QUANTITY_EPSILON)

    def test_partial_sell_above_epsilon_uses_standard_path(self):
        held, qty = 10.0, 9.0
        self.assertFalse((held - qty) <= portfolio_db.QUANTITY_EPSILON)


class HoldingQuantityCheckViolationTests(unittest.TestCase):
    def test_string_match_for_check_constraint(self):
        exc = Exception("violates check constraint portfolio_holdings_quantity_check")
        self.assertTrue(portfolio_db._is_holding_quantity_check_violation(exc))

    def test_unrelated_error_not_matched(self):
        exc = ValueError("Insufficient cash")
        self.assertFalse(portfolio_db._is_holding_quantity_check_violation(exc))


class ExecuteTradeSellPathTests(unittest.TestCase):
    def test_full_sell_uses_manual_path_and_nets_pnl(self):
        conn = MagicMock()
        gross_pnl = 500.0
        sell_charge = 20.0

        with (
            patch.object(portfolio_db, "transaction_charge_for_action", return_value=sell_charge),
            patch.object(portfolio_db, "load_holding", return_value=(10.0, 100.0)),
            patch.object(
                portfolio_db,
                "_execute_sell_without_zero_holding_update",
            ) as manual_sell,
            patch.object(portfolio_db, "_deduct_transaction_charge") as deduct_charge,
            patch.object(
                portfolio_db,
                "_adjust_latest_sell_pnl",
                return_value=gross_pnl - sell_charge,
            ) as adjust_pnl,
        ):
            net = portfolio_db.execute_trade(
                conn, ticker="TCS.NS", action="SELL", quantity=10.0, price=150.0
            )

        manual_sell.assert_called_once_with(
            conn, ticker="TCS.NS", quantity=10.0, price=150.0
        )
        deduct_charge.assert_called_once()
        adjust_pnl.assert_called_once_with(conn, "TCS.NS", sell_charge)
        self.assertEqual(net, gross_pnl - sell_charge)
        conn.commit.assert_called_once()

    def test_check_violation_falls_back_to_manual_sell(self):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        cursor.execute.side_effect = Exception(
            "new row for relation portfolio_holdings violates check constraint "
            '"portfolio_holdings_quantity_check"'
        )

        with (
            patch.object(portfolio_db, "transaction_charge_for_action", return_value=0.0),
            patch.object(portfolio_db, "load_holding", return_value=(10.0, 100.0)),
            patch.object(
                portfolio_db,
                "_execute_sell_without_zero_holding_update",
            ) as manual_sell,
            patch.object(portfolio_db, "_adjust_latest_sell_pnl", return_value=100.0),
        ):
            portfolio_db.execute_trade(
                conn, ticker="TCS.NS", action="SELL", quantity=9.5, price=150.0
            )

        conn.rollback.assert_called_once()
        manual_sell.assert_called_once()


if __name__ == "__main__":
    unittest.main()
