from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
