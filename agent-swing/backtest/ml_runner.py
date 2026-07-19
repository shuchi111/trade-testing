"""Fast ML-only backtest for the ML tab (~1–3 min vs 20+ min walk-forward per bar).

Retrains LightGBM every 21 bars (~monthly) instead of every bar. Full accuracy
walk-forward remains available via ``python -m backtest.runner``.

Usage:
  python -m backtest.ml_runner --ticker RELIANCE.NS
  python -m backtest.ml_runner --ticker TCS.NS --full-walkforward
"""
from __future__ import annotations

import argparse
import logging
import traceback

import pandas as pd

from .config import (
    BACKTEST_END,
    BACKTEST_START,
    DEFAULT_FEES,
    DEFAULT_INIT_CASH,
    DEFAULT_SLIPPAGE,
)
from .data_loader import load_price_data, load_volume_data
from .db_store import store_backtest_result, store_trade_logs
from .runner import extract_metrics, extract_trade_logs, generate_diagnostic
from .signal_builder import compute_ic_from_predictions
from .strategies.ml_forecast import MlForecastStrategy

logger = logging.getLogger(__name__)


def _fast_ml_strategy(full_walkforward: bool) -> MlForecastStrategy:
    if full_walkforward:
        return MlForecastStrategy()
    return MlForecastStrategy(
        retrain_step=21,
        lgb_params={
            "n_estimators": 120,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "n_jobs": 1,
        },
    )


def run_ml_backtest(
    ticker: str,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    init_cash: float = DEFAULT_INIT_CASH,
    fees: float = DEFAULT_FEES,
    slippage: float = DEFAULT_SLIPPAGE,
    store: bool = True,
    full_walkforward: bool = False,
) -> dict | None:
    """Run ml_forecast only — fast periodic retrain by default."""
    mode = "full walk-forward" if full_walkforward else "fast (retrain every 21 bars)"
    print(f"\n{'='*60}")
    print(f"  ML backtest: {ticker}  ({start} to {end})  [{mode}]")
    print(f"{'='*60}")

    price = load_price_data(ticker, start, end)
    if price.empty:
        print(f"  No price data for {ticker}")
        return None

    volume = load_volume_data(ticker, start, end)
    strategy = _fast_ml_strategy(full_walkforward)
    if volume is not None:
        strategy._volume = volume.reindex(price.index)

    print(f"\n  Strategy: {strategy.name} (retrain_step={strategy.retrain_step})")
    try:
        pf = strategy.run(price, init_cash=init_cash, fees=fees, slippage=slippage)
    except Exception as e:
        logger.error("ml_forecast failed for %s: %s\n%s", ticker, e, traceback.format_exc())
        print(f"    ERROR: {e}")
        return None

    metrics = extract_metrics(pf)
    metrics["strategy_name"] = strategy.name
    metrics["ticker"] = ticker
    metrics["date_from"] = start
    metrics["date_to"] = end
    metrics["init_cash"] = init_cash
    metrics["fees_pct"] = fees
    metrics["slippage_pct"] = slippage
    metrics["strategy_config"] = getattr(strategy, "config", {})

    metrics["ml_horizon"] = getattr(strategy, "last_horizon", None)
    metrics["ml_train_rows"] = getattr(strategy, "last_train_rows", None)
    metrics["ml_feature_count"] = getattr(strategy, "last_feature_count", None)
    _retrain = getattr(strategy, "last_retrain_date", None)
    metrics["ml_retrain_date"] = (
        str(_retrain.date()) if hasattr(_retrain, "date") else (str(_retrain) if _retrain else None)
    )
    preds = getattr(strategy, "last_predictions", None)
    if preds is not None:
        horizon = getattr(strategy, "last_horizon", None) or 5
        ml_ic = compute_ic_from_predictions(preds, price, hold_days=int(horizon))
        metrics["ic"] = ml_ic.get("ic")
        metrics["rank_ic"] = ml_ic.get("rank_ic")

    _ret = metrics.get("total_return_pct")
    _sharpe = metrics.get("sharpe_ratio")
    _trades = metrics.get("total_trades")
    print(
        f"    Return: {_ret:+.1f}%  |  Sharpe: {_sharpe:.2f}  |  Trades: {_trades}"
        if _ret is not None and _sharpe is not None
        else f"    Return: {_ret}  |  Trades: {_trades}"
    )

    e_reasons = x_reasons = e_vals = x_vals = None
    try:
        e_reasons, x_reasons, e_vals, x_vals = strategy.build_trade_reasons(price)
    except Exception:
        pass

    trade_logs = extract_trade_logs(
        pf, ticker, strategy.name,
        entry_reasons=e_reasons, exit_reasons=x_reasons,
        entry_values=e_vals, exit_values=x_vals,
    )
    metrics["diagnostic"] = generate_diagnostic(metrics, trade_logs)
    diag = str(metrics["diagnostic"]).encode("ascii", errors="replace").decode("ascii")
    print(f"    {diag}")

    if store:
        try:
            result_id = store_backtest_result(metrics)
            if trade_logs:
                store_trade_logs(result_id, trade_logs)
            metrics["db_result_id"] = result_id
            print(f"    Stored result id: {result_id}")
        except Exception as e:
            logger.error("DB store failed: %s\n%s", e, traceback.format_exc())
            print(f"    DB store failed: {e}")
            raise SystemExit(1) from e

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ML-only VectorBT backtest (ml_forecast)")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker e.g. RELIANCE.NS")
    parser.add_argument("--start", default=BACKTEST_START)
    parser.add_argument("--end", default=BACKTEST_END)
    parser.add_argument("--no-store", action="store_true")
    parser.add_argument(
        "--full-walkforward",
        action="store_true",
        help="Retrain every bar (slow, 15–30+ min)",
    )
    args = parser.parse_args()

    run_ml_backtest(
        args.ticker.strip().upper(),
        start=args.start,
        end=args.end,
        store=not args.no_store,
        full_walkforward=args.full_walkforward,
    )


if __name__ == "__main__":
    main()
