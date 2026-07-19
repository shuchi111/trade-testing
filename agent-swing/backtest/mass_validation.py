"""
Mass strategy validation — video-style grid sweep (1,000+ combos per ticker).

Tests every parameter combo in param_grids across tickers, ranks survivors that pass
the same filters as the ML tab validation funnel.

Usage:
  python -m backtest.mass_validation --ticker RELIANCE.NS --json
  python -m backtest.mass_validation --universe --json --max-tickers 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import BACKTEST_END, BACKTEST_START, TICKERS
from .grid_search import run_grid
from .param_grids import GRIDS

logger = logging.getLogger(__name__)

STOCK_TICKERS = [t for t in TICKERS if t.endswith(".NS")]


def count_experiments(tickers: list[str], strategies: list[str]) -> int:
    return sum(len(GRIDS[s]()) for s in strategies) * len(tickers)


def run_mass_validation(
    tickers: list[str],
    strategies: list[str] | None = None,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    store: bool = True,
    top_per_strategy: int = 10,
    quiet: bool = False,
) -> dict:
    strategies = strategies or list(GRIDS.keys())
    total_experiments = count_experiments(tickers, strategies)
    survivors: list[dict] = []
    tested_by_strategy: dict[str, int] = {}
    survivor_by_strategy: dict[str, int] = {}

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  Mass validation — {len(tickers)} tickers × {len(strategies)} families")
        print(f"  {total_experiments:,} total backtests")
        print(f"{'='*60}")

    for ti, ticker in enumerate(tickers, 1):
        if not quiet:
            print(f"\n[{ti}/{len(tickers)}] {ticker}")
        for strategy_name in strategies:
            n_combos = len(GRIDS[strategy_name]())
            tested_by_strategy[strategy_name] = tested_by_strategy.get(strategy_name, 0) + n_combos
            try:
                top = run_grid(
                    ticker,
                    strategy_name,
                    start=start,
                    end=end,
                    store=store,
                    top_n=top_per_strategy,
                )
            except Exception as exc:
                logger.warning("Grid failed %s/%s: %s", ticker, strategy_name, exc)
                continue
            if top:
                survivor_by_strategy[strategy_name] = (
                    survivor_by_strategy.get(strategy_name, 0) + len(top)
                )
                for row in top:
                    survivors.append(row)

    survivors.sort(key=lambda x: x.get("score", 0), reverse=True)
    for i, row in enumerate(survivors, 1):
        row["global_rank"] = i

    survival_rate = (
        (len(survivors) / total_experiments * 100) if total_experiments else 0.0
    )

    mean_rev = {"rsi", "bollinger", "keltner"}
    mean_rev_survivors = sum(
        1 for s in survivors if s.get("strategy_name") in mean_rev
    )
    mean_rev_rate = (
        (mean_rev_survivors / len(survivors) * 100) if survivors else 0.0
    )

    summary = {
        "ok": True,
        "total_experiments": total_experiments,
        "survivor_count": len(survivors),
        "survival_rate_pct": round(survival_rate, 2),
        "mean_reversion_survivors": mean_rev_survivors,
        "mean_reversion_share_pct": round(mean_rev_rate, 1),
        "tested_by_strategy": tested_by_strategy,
        "survivor_by_strategy": survivor_by_strategy,
        "tickers": tickers,
        "strategies": strategies,
        "top_picks": [
            {
                "rank": r.get("global_rank"),
                "ticker": r.get("ticker"),
                "strategy_name": r.get("strategy_name"),
                "params": r.get("params"),
                "score": r.get("score"),
                "total_return_pct": r.get("total_return_pct"),
                "sharpe_ratio": r.get("sharpe_ratio"),
                "total_trades": r.get("total_trades"),
                "max_drawdown_pct": r.get("max_drawdown_pct"),
                "profit_factor": r.get("profit_factor"),
                "expectancy_pct": r.get("expectancy_pct"),
            }
            for r in survivors[:20]
        ],
    }

    if not quiet:
        print(f"\n  Survivors: {len(survivors):,} / {total_experiments:,} ({survival_rate:.2f}%)")
        if survivors:
            print(
                f"  Mean-reversion share: {mean_rev_rate:.0f}% "
                f"({mean_rev_survivors}/{len(survivors)})"
            )
            print("  Top 5:")
            for r in survivors[:5]:
                print(
                    f"    #{r['global_rank']} {r['ticker']} {r['strategy_name']} "
                    f"score={r.get('score', 0):.3f} sharpe={r.get('sharpe_ratio', 0):.2f} "
                    f"params={r.get('params')}"
                )
        else:
            print("  No survivors passed all filters.")

    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Mass grid validation (video-style)")
    p.add_argument("--ticker", help="Single ticker e.g. RELIANCE.NS")
    p.add_argument("--universe", action="store_true", help="All NSE tickers in config")
    p.add_argument("--max-tickers", type=int, default=0, help="Cap universe size (0=all)")
    p.add_argument("--start", default=BACKTEST_START)
    p.add_argument("--end", default=BACKTEST_END)
    p.add_argument("--no-store", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.universe:
        tickers = STOCK_TICKERS
        if args.max_tickers > 0:
            tickers = tickers[: args.max_tickers]
    elif args.ticker:
        tickers = [args.ticker.strip().upper()]
    else:
        p.error("Provide --ticker or --universe")

    out = run_mass_validation(
        tickers,
        start=args.start,
        end=args.end,
        store=not args.no_store,
        quiet=args.json,
    )
    if args.json:
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
