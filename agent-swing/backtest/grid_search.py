"""
VectorBT grid search — test 100+ parameter combinations per strategy.

Usage:
  python -m backtest.grid_search --ticker RELIANCE.NS --strategy rsi
  python -m backtest.grid_search --ticker TCS.NS --strategy sma_crossover --store
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import BACKTEST_END, BACKTEST_START, DEFAULT_FEES, DEFAULT_INIT_CASH, DEFAULT_SLIPPAGE
from .data_loader import load_price_data
from .db_store import get_supabase, store_backtest_result
from .param_grids import GRIDS
from .results_ranker import passes_filters, rank_score
from .runner import extract_metrics, generate_diagnostic
from .strategies.bollinger import BollingerStrategy
from .strategies.composite import CompositeStrategy
from .strategies.ensemble import EnsembleStrategy
from .strategies.keltner import KeltnerStrategy
from .strategies.macd import MacdStrategy
from .strategies.prophet_forecast import ProphetForecastStrategy
from .strategies.rsi import RsiStrategy
from .strategies.sma_crossover import SmaCrossoverStrategy

logger = logging.getLogger(__name__)


def build_strategy(name: str, params: dict):
    if name == "sma_crossover":
        return SmaCrossoverStrategy(fast=params["fast"], slow=params["slow"])
    if name == "rsi":
        return RsiStrategy(
            window=params["window"],
            oversold=params["oversold"],
            overbought=params["overbought"],
        )
    if name == "bollinger":
        return BollingerStrategy(window=params["window"], std_dev=params["std_dev"])
    if name == "keltner":
        return KeltnerStrategy(window=params["window"], atr_mult=params["atr_mult"])
    if name == "macd":
        return MacdStrategy(fast=params["fast"], slow=params["slow"], signal=params["signal"])
    if name == "composite":
        return CompositeStrategy(**params)
    if name == "prophet_forecast":
        return ProphetForecastStrategy(
            horizon=int(params.get("horizon", 5)),
            buy_threshold=float(params.get("buy_threshold", 0.005)),
            sell_threshold=float(params.get("sell_threshold", 0.005)),
            retrain_step=int(params.get("retrain_step", 42)),
        )
    if name == "ensemble":
        return EnsembleStrategy(min_votes=int(params.get("min_votes", 2)))
    raise ValueError(f"Unknown strategy: {name}")


def run_grid(
    ticker: str,
    strategy_name: str,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    store: bool = True,
    top_n: int = 5,
) -> list[dict]:
    if strategy_name not in GRIDS:
        raise ValueError(f"No grid for {strategy_name}. Available: {list(GRIDS)}")

    price = load_price_data(ticker, start, end)
    if price.empty:
        logger.warning("No price data for %s", ticker)
        return []

    combos = GRIDS[strategy_name]()
    logger.info("Grid search %s/%s — %d combinations", ticker, strategy_name, len(combos))

    results: list[dict] = []
    for params in combos:
        try:
            strategy = build_strategy(strategy_name, params)
            pf = strategy.run(price, init_cash=DEFAULT_INIT_CASH, fees=DEFAULT_FEES, slippage=DEFAULT_SLIPPAGE)
            metrics = extract_metrics(pf)
            metrics["strategy_name"] = strategy_name
            metrics["ticker"] = ticker
            metrics["params"] = params
            metrics["score"] = rank_score(metrics)
            if passes_filters(metrics):
                results.append(metrics)
        except Exception as e:
            logger.debug("Combo failed %s: %s", params, e)

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    if store and results:
        sb = get_supabase()
        rows = []
        for r in results[:50]:
            rows.append({
                "strategy_name": strategy_name,
                "ticker": ticker,
                "date_from": start,
                "date_to": end,
                "params": r["params"],
                "score": r["score"],
                "rank": r["rank"],
                "metrics": {k: v for k, v in r.items() if k not in ("params",)},
            })
        for i in range(0, len(rows), 50):
            sb.table("bt_grid_search_results").insert(rows[i : i + 50]).execute()

        for r in results[:top_n]:
            opt_name = f"{strategy_name}_optimized"
            diag = generate_diagnostic({**r, "strategy_name": opt_name})
            store_backtest_result({
                **r,
                "strategy_name": opt_name,
                "date_from": start,
                "date_to": end,
                "strategy_config": r["params"],
                "diagnostic": diag,
                "run_by": "grid_search",
            })

    logger.info("Top score: %.4f (%d passing combos)", results[0]["score"] if results else 0, len(results))
    return results[:top_n]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="VectorBT parameter grid search")
    p.add_argument("--ticker", required=True)
    p.add_argument("--strategy", required=True, choices=list(GRIDS.keys()))
    p.add_argument("--start", default=BACKTEST_START)
    p.add_argument("--end", default=BACKTEST_END)
    p.add_argument("--no-store", action="store_true")
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()

    top = run_grid(
        args.ticker,
        args.strategy,
        start=args.start,
        end=args.end,
        store=not args.no_store,
        top_n=args.top,
    )
    for r in top:
        print(
            f"  rank={r['rank']} score={r['score']:.3f} "
            f"return={r.get('total_return_pct', 0):+.1f}% "
            f"sharpe={r.get('sharpe_ratio', 0):.2f} params={r['params']}"
        )
    if not top:
        sys.exit(1)


if __name__ == "__main__":
    main()
