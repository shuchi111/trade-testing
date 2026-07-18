"""Unit tests for backtest financial calculations and signal logic.

No network or database — uses mocks and synthetic price data.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import extract_metrics
from backtest.signal_builder import compute_signal_accuracy
from backtest.strategies.macd import MacdStrategy
from backtest.strategies.rsi import RsiStrategy
from backtest.strategies.bollinger import BollingerStrategy


def _mock_portfolio(
    trade_returns: list[float],
    win_rate: float = 40.0,
    stats_overrides: dict | None = None,
) -> MagicMock:
    """Build a minimal VectorBT Portfolio mock for extract_metrics tests."""
    stats = {
        "Win Rate [%]": win_rate,
        "Annualized Return [%]": 12.0,
        "Sharpe Ratio": 1.2,
        "Sortino Ratio": 1.5,
        "Calmar Ratio": 0.8,
        "Profit Factor": 1.4,
        "Total Trades": len(trade_returns),
        "Avg Holding Period": 10.0,
    }
    if stats_overrides:
        stats.update(stats_overrides)

    pf = MagicMock()
    pf.stats.return_value = stats
    pf.trades.records_readable = pd.DataFrame({"Return": trade_returns})
    pf.total_return.return_value = 0.08
    pf.max_drawdown.return_value = -0.12
    pf.value.return_value = pd.Series([100_000.0, 108_000.0])
    return pf


class TestExtractMetrics(unittest.TestCase):
    def test_expectancy_with_mixed_wins_and_losses(self):
        # 2 wins (+5%, +3%), 3 losses (-2%, -1%, -4%)
        pf = _mock_portfolio([0.05, -0.02, 0.03, -0.01, -0.04], win_rate=40.0)
        metrics = extract_metrics(pf)

        self.assertAlmostEqual(metrics["avg_win_pct"], 4.0)
        self.assertAlmostEqual(metrics["avg_loss_pct"], -70 / 30)
        self.assertAlmostEqual(metrics["winning_trades"], 2)
        self.assertAlmostEqual(metrics["losing_trades"], 3)

        # expectancy = 0.4*4 + 0.6*(-2.333...) = 1.6 - 1.4 = 0.2
        self.assertAlmostEqual(metrics["expectancy_pct"], 0.2, places=1)

    def test_breakeven_trade_counts_as_loss(self):
        pf = _mock_portfolio([0.0, 0.05], win_rate=50.0)
        metrics = extract_metrics(pf)

        self.assertEqual(metrics["winning_trades"], 1)
        self.assertEqual(metrics["losing_trades"], 1)

    def test_risk_reward_none_when_no_losses(self):
        pf = _mock_portfolio([0.05, 0.03], win_rate=100.0)
        metrics = extract_metrics(pf)

        self.assertIsNone(metrics["risk_reward"])

    def test_risk_reward_ratio(self):
        pf = _mock_portfolio([0.10, -0.05], win_rate=50.0)
        metrics = extract_metrics(pf)

        self.assertAlmostEqual(metrics["risk_reward"], 2.0)

    def test_safe_float_handles_nan_and_inf_in_stats(self):
        pf = _mock_portfolio(
            [0.05],
            win_rate=float("nan"),
            stats_overrides={
                "Sharpe Ratio": float("inf"),
                "Sortino Ratio": None,
                "Profit Factor": "bad",
            },
        )
        metrics = extract_metrics(pf)

        self.assertEqual(metrics["win_rate_pct"], 0.0)
        self.assertEqual(metrics["sharpe_ratio"], 0.0)
        self.assertEqual(metrics["sortino_ratio"], 0.0)
        self.assertEqual(metrics["profit_factor"], 0.0)

    def test_empty_trades_dataframe(self):
        pf = MagicMock()
        pf.stats.return_value = {"Win Rate [%]": 0, "Total Trades": 0}
        pf.trades.records_readable = pd.DataFrame()
        pf.total_return.return_value = 0.0
        pf.max_drawdown.return_value = 0.0
        pf.value.return_value = pd.Series([100_000.0])

        metrics = extract_metrics(pf)

        self.assertEqual(metrics["total_trades"], 0)
        self.assertEqual(metrics["winning_trades"], 0)
        self.assertEqual(metrics["losing_trades"], 0)
        self.assertEqual(metrics["avg_win_pct"], 0.0)
        self.assertEqual(metrics["avg_loss_pct"], 0.0)
        self.assertIsNone(metrics["risk_reward"])


class TestComputeSignalAccuracy(unittest.TestCase):
    def _price_series(self, n: int = 30, start: float = 100.0, step: float = 1.0) -> pd.Series:
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.Series([start + i * step for i in range(n)], index=dates)

    def test_buy_correct_in_uptrend(self):
        price = self._price_series()
        recs = pd.DataFrame([{"trade_date": price.index[0], "bucket": "buy"}])

        result = compute_signal_accuracy(recs, price, hold_days=5)

        self.assertEqual(result["total_buys"], 1)
        self.assertEqual(result["correct_buys"], 1)
        self.assertEqual(result["buy_precision"], 1.0)

    def test_sell_correct_in_downtrend(self):
        price = self._price_series(step=-1.0)
        recs = pd.DataFrame([{"trade_date": price.index[0], "bucket": "sell"}])

        result = compute_signal_accuracy(recs, price, hold_days=5)

        self.assertEqual(result["total_sells"], 1)
        self.assertEqual(result["correct_sells"], 1)
        self.assertEqual(result["sell_precision"], 1.0)

    def test_uses_trading_days_not_calendar_days(self):
        price = self._price_series(n=10, step=1.0)
        recs = pd.DataFrame([{"trade_date": price.index[0], "bucket": "buy"}])

        result = compute_signal_accuracy(recs, price, hold_days=3)

        # 3 trading days forward: index 0 -> index 3
        entry = price.iloc[0]
        exit_ = price.iloc[3]
        expected_correct = (exit_ - entry) / entry > 0

        self.assertEqual(result["correct_buys"], int(expected_correct))
        self.assertEqual(result["total_buys"], 1)

    def test_single_price_point_skipped(self):
        dates = pd.date_range("2024-01-01", periods=1, freq="B")
        price = pd.Series([100.0], index=dates)
        recs = pd.DataFrame([{"trade_date": dates[0], "bucket": "buy"}])

        result = compute_signal_accuracy(recs, price, hold_days=20)

        self.assertEqual(result["total_buys"], 0)
        self.assertIsNone(result["buy_precision"])

    def test_empty_price_series(self):
        price = pd.Series(dtype=float)
        recs = pd.DataFrame([{"trade_date": pd.Timestamp("2024-01-01"), "bucket": "buy"}])

        result = compute_signal_accuracy(recs, price)

        self.assertEqual(result["total_buys"], 0)
        self.assertEqual(result["total_sells"], 0)
        self.assertIsNone(result["directional_acc"])

    def test_hold_and_unknown_buckets_ignored(self):
        price = self._price_series()
        recs = pd.DataFrame(
            [
                {"trade_date": price.index[0], "bucket": "hold"},
                {"trade_date": price.index[1], "bucket": "unknown"},
            ]
        )

        result = compute_signal_accuracy(recs, price)

        self.assertEqual(result["total_buys"], 0)
        self.assertEqual(result["total_sells"], 0)


class TestStrategySignals(unittest.TestCase):
    def _price(self, n: int = 120) -> pd.Series:
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        # Oscillating series so indicators can fire
        values = 100 + 10 * np.sin(np.linspace(0, 8 * math.pi, n))
        return pd.Series(values, index=dates)

    def test_rsi_exit_not_same_bar_as_entry(self):
        price = self._price()
        entries, exits = RsiStrategy().generate_signals(price)

        overlap = (entries & exits).sum()
        self.assertEqual(overlap, 0, "RSI entry and exit must not fire on same bar")

    def test_bollinger_exit_not_same_bar_as_entry(self):
        price = self._price()
        entries, exits = BollingerStrategy().generate_signals(price)

        overlap = (entries & exits).sum()
        self.assertEqual(overlap, 0)

    def test_macd_exit_not_same_bar_as_entry(self):
        price = self._price()
        entries, exits = MacdStrategy().generate_signals(price)

        overlap = (entries & exits).sum()
        self.assertEqual(overlap, 0)

    def test_macd_exits_all_nan_after_shift(self):
        """Short series: shift(1) leaves all-NaN exits — must not raise."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        price = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0], index=dates)

        entries, exits = MacdStrategy().generate_signals(price)

        self.assertEqual(len(exits), len(price))
        self.assertFalse(exits.isna().any())


if __name__ == "__main__":
    unittest.main()
