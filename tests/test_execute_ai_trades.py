from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("psycopg2", types.SimpleNamespace(connect=lambda *args, **kwargs: None))

import execute_ai_trades


class DecideAndExecuteTests(unittest.TestCase):
    def _patch_common_dependencies(self, *, reference_price: float | None, cash: float):
        logged: list[dict] = []

        def _mock_snapshot(*_args, **_kwargs):
            return types.SimpleNamespace(latest_close=reference_price)

        patches = [
            patch.object(execute_ai_trades, "already_executed", return_value=False),
            patch.object(
                execute_ai_trades,
                "latest_recommendation",
                return_value={
                    "id": "reco-1",
                    "decision": "BUY",
                    "final_trade_decision": "",
                    "reference_price": reference_price,
                },
            ),
            patch.object(execute_ai_trades, "require_fresh_market_snapshot", side_effect=_mock_snapshot),
            patch.object(execute_ai_trades, "resolve_canonical_decision", return_value="BUY"),
            patch.object(execute_ai_trades, "recommendation_bucket", return_value="buy"),
            patch.object(execute_ai_trades, "is_overweight", return_value=False),
            patch.object(execute_ai_trades, "load_holding", return_value=(0.0, 0.0)),
            patch.object(execute_ai_trades, "evaluate_trailing_stop", return_value=None),
            patch.object(execute_ai_trades, "load_wallet_cash", return_value=cash),
            patch.object(execute_ai_trades, "max_position_inr", return_value=100000.0),
            patch.object(execute_ai_trades, "min_wallet_cash_reserve_inr", return_value=0.0),
            patch.object(execute_ai_trades, "buy_transaction_charge_inr", return_value=0.0),
            patch.object(execute_ai_trades, "execute_trade", Mock()),
            patch.object(execute_ai_trades, "log_execution", side_effect=lambda *args, **kwargs: logged.append(kwargs)),
        ]
        return patches, logged

    def test_missing_reference_price_skips_without_unverified_fallback(self):
        patches, logged = self._patch_common_dependencies(reference_price=None, cash=100000.0)
        for dep_patch in patches:
            self.enterContext(dep_patch)

        result = execute_ai_trades.decide_and_execute(
            object(),
            ticker="TCS.NS",
            trade_date="2026-06-28",
            dry_run=False,
            settings={"auto_trade": True, "max_position_inr": 100000.0},
        )

        self.assertEqual(result["action_taken"], "SKIP")
        self.assertEqual(result["skip_reason"], "no_price")
        self.assertEqual(logged[-1]["skip_reason"], "no_price")
        execute_ai_trades.execute_trade.assert_not_called()

    def test_buy_below_share_price_skips_with_whole_share_reason(self):
        patches, logged = self._patch_common_dependencies(reference_price=40000.0, cash=30000.0)
        for dep_patch in patches:
            self.enterContext(dep_patch)

        result = execute_ai_trades.decide_and_execute(
            object(),
            ticker="MRF.NS",
            trade_date="2026-06-28",
            dry_run=False,
            settings={"auto_trade": True, "max_position_inr": 100000.0},
        )

        self.assertEqual(result["action_taken"], "SKIP")
        self.assertEqual(result["skip_reason"], "insufficient_cash_for_whole_share")
        self.assertEqual(logged[-1]["skip_reason"], "insufficient_cash_for_whole_share")
        execute_ai_trades.execute_trade.assert_not_called()


class TradeBlockSkipReasonTests(unittest.TestCase):
    def test_no_open_position(self):
        exc = ValueError("No open position to sell for TCS.NS")
        self.assertEqual(execute_ai_trades.trade_block_skip_reason(exc), "no_position_to_sell")

    def test_cannot_sell_oversize(self):
        exc = ValueError("Cannot sell 10 TCS.NS; current holding is 5")
        self.assertEqual(execute_ai_trades.trade_block_skip_reason(exc), "no_position_to_sell")

    def test_insufficient_cash(self):
        exc = ValueError("Insufficient cash: need 1,000.00")
        self.assertEqual(execute_ai_trades.trade_block_skip_reason(exc), "insufficient_cash")


if __name__ == "__main__":
    unittest.main()
