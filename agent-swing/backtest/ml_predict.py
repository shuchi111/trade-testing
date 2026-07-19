"""Live LightGBM inference for the ML tab (/api/ml/predict).

Trains once on realized labels and predicts the latest bar — fast path for
the ML tab (full walk-forward is only used during backtests).

CLI:
  python -m backtest.ml_predict --ticker TCS.NS --json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _configure_runtime() -> None:
    """Quiet noisy third-party warnings on Windows CLI/API runs."""
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Could not find the number of physical cores*",
        category=UserWarning,
    )


def _top_importances(importances: dict[str, float] | None, n: int = 10) -> list[dict]:
    if not importances:
        return []
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return [{"feature": name, "importance": round(float(val), 6)} for name, val in ranked[:n]]


def predict_latest(
    ticker: str,
    start: str = "2022-01-01",
    end: str | None = None,
    *,
    include_features: bool = True,
) -> dict:
    """Run ML inference for the latest trading bar."""
    _configure_runtime()
    try:
        from .data_loader import load_price_data, load_volume_data
        from .strategies.ml_forecast import MlForecastStrategy
    except ImportError:
        from data_loader import load_price_data, load_volume_data
        from strategies.ml_forecast import MlForecastStrategy

    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    price = load_price_data(ticker, start, end)
    if price.empty:
        return {"ok": False, "error": f"No price data for {ticker}"}

    volume = load_volume_data(ticker, start, end)
    strategy = MlForecastStrategy()
    if volume is not None:
        strategy._volume = volume.reindex(price.index)

    pred = strategy.predict_live(price)
    if pred is None or not np.isfinite(pred):
        return {
            "ok": False,
            "error": "Insufficient history to train ML model (need ~120+ bars)",
        }

    entries = strategy._cached_entries
    exits = strategy._cached_exits
    side = "HOLD"
    if entries is not None and bool(entries.iloc[-1]):
        side = "BUY"
    elif exits is not None and bool(exits.iloc[-1]):
        side = "SELL"

    threshold = max(strategy.buy_threshold, strategy.sell_threshold, 1e-9)
    confidence = min(1.0, abs(float(pred)) / threshold)

    latest_idx = price.index[-1]
    as_of = str(latest_idx.date()) if hasattr(latest_idx, "date") else str(latest_idx)
    close = float(price.iloc[-1])

    return {
        "ok": True,
        "ticker": ticker,
        "as_of": as_of,
        "side": side,
        "predicted_return_pct": round(float(pred) * 100, 4),
        "confidence": round(confidence, 4),
        "horizon_days": strategy.last_horizon or strategy.horizon,
        "reference_price": round(close, 4),
        "buy_threshold_pct": round(strategy.buy_threshold * 100, 2),
        "sell_threshold_pct": round(strategy.sell_threshold * 100, 2),
        "train_rows": strategy.last_train_rows,
        "feature_count": strategy.last_feature_count,
        "retrain_date": (
            str(strategy.last_retrain_date.date())
            if strategy.last_retrain_date is not None and hasattr(strategy.last_retrain_date, "date")
            else None
        ),
        "top_features": _top_importances(strategy.last_feature_importances) if include_features else [],
    }


def _has_recent_ml_backtest(ticker: str, skip_days: int) -> bool:
    """True if ml_forecast backtest exists within skip_days."""
    if skip_days <= 0:
        return False
    try:
        from .config import SUPABASE_KEY, SUPABASE_URL
        from .db_store import get_supabase
    except ImportError:
        from config import SUPABASE_KEY, SUPABASE_URL
        from db_store import get_supabase

    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        sb = get_supabase()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=skip_days)).isoformat()
        resp = (
            sb.table("bt_strategy_results")
            .select("id, created_at")
            .eq("ticker", ticker.upper())
            .eq("strategy_name", "ml_forecast")
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return bool(resp.data)
    except Exception as exc:
        logger.debug("Recent backtest check failed for %s: %s", ticker, exc)
        return False


def predict_universe(
    tickers: list[str] | None = None,
    start: str = "2022-01-01",
    *,
    stocks_only: bool = True,
    with_backtest: bool = False,
    skip_backtest_days: int = 7,
    store: bool = True,
) -> dict:
    """Run live ML inference for many tickers (one process, sequential)."""
    _configure_runtime()
    t0 = time.perf_counter()
    try:
        from .config import TICKERS
    except ImportError:
        from config import TICKERS

    universe = tickers or list(TICKERS)
    if stocks_only:
        universe = [t for t in universe if not t.endswith("-USD")]

    signals: list[dict] = []
    backtest_rows: list[dict] = []
    backtest_ok = backtest_failed = backtest_skipped = 0

    if with_backtest:
        try:
            from .ml_runner import run_ml_backtest
        except ImportError:
            from ml_runner import run_ml_backtest

    for ticker in universe:
        row = predict_latest(ticker, start=start, include_features=False)
        signals.append(row)

        if with_backtest and row.get("ok"):
            t = str(row.get("ticker") or ticker).upper()
            if _has_recent_ml_backtest(t, skip_backtest_days):
                backtest_skipped += 1
                backtest_rows.append({"ticker": t, "ok": True, "skipped": True, "reason": "recent backtest"})
                continue
            try:
                metrics = run_ml_backtest(t, start=start, store=True)
                if metrics:
                    backtest_ok += 1
                    backtest_rows.append({
                        "ticker": t,
                        "ok": True,
                        "skipped": False,
                        "sharpe_ratio": metrics.get("sharpe_ratio"),
                        "total_return_pct": metrics.get("total_return_pct"),
                        "db_result_id": metrics.get("db_result_id"),
                    })
                else:
                    backtest_failed += 1
                    backtest_rows.append({"ticker": t, "ok": False, "skipped": False, "error": "backtest returned no metrics"})
            except (Exception, SystemExit) as exc:
                backtest_failed += 1
                backtest_rows.append({"ticker": t, "ok": False, "skipped": False, "error": str(exc)})

    ok_rows = [r for r in signals if r.get("ok")]
    summary = {
        "total": len(signals),
        "ok": len(ok_rows),
        "failed": len(signals) - len(ok_rows),
        "buy": sum(1 for r in ok_rows if r.get("side") == "BUY"),
        "sell": sum(1 for r in ok_rows if r.get("side") == "SELL"),
        "hold": sum(1 for r in ok_rows if r.get("side") == "HOLD"),
    }
    as_of_dates = [r.get("as_of") for r in ok_rows if r.get("as_of")]
    duration_sec = round(time.perf_counter() - t0, 1)

    backtest_summary = None
    if with_backtest:
        backtest_summary = {
            "total": len(universe),
            "ok": backtest_ok,
            "failed": backtest_failed,
            "skipped": backtest_skipped,
            "rows": backtest_rows,
        }

    payload = {
        "ok": True,
        "as_of": max(as_of_dates) if as_of_dates else None,
        "summary": summary,
        "signals": signals,
        "with_backtest": with_backtest,
        "backtest_summary": backtest_summary,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "duration_sec": duration_sec,
        "cached": False,
    }

    if store:
        try:
            from .universe_cache import store_universe_signals
        except ImportError:
            from universe_cache import store_universe_signals
        store_info = store_universe_signals(payload)
        payload["store_info"] = store_info

    return payload


if __name__ == "__main__":
    _configure_runtime()
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Live LightGBM inference")
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--universe", action="store_true", help="All NSE stocks in config")
    parser.add_argument("--with-backtest", action="store_true", help="Also run ml_forecast backtest per ticker (stores to bt_strategy_results)")
    parser.add_argument("--skip-backtest-days", type=int, default=7, help="Skip backtest if ml_forecast ran within N days")
    parser.add_argument("--no-store", action="store_true", help="Do not persist universe scan to cache/DB")
    parser.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = parser.parse_args()

    if args.universe:
        result = predict_universe(
            start=args.start,
            stocks_only=True,
            with_backtest=args.with_backtest,
            skip_backtest_days=args.skip_backtest_days,
            store=not args.no_store,
        )
    else:
        result = predict_latest(args.ticker, start=args.start)
    if args.json:
        print(json.dumps(result))
    else:
        if not result.get("ok"):
            print(result.get("error", "unknown error"))
        else:
            print(
                f"{result['ticker']} {result['as_of']}: {result['side']} "
                f"(pred {result['predicted_return_pct']:+.2f}%, conf {result['confidence']:.0%})"
            )
