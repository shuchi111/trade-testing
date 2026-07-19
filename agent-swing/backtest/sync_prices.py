"""CLI: sync 10–15y Yahoo daily bars into market_daily_bars.

Usage:
  python -m backtest.sync_prices --ticker RELIANCE.NS
  python -m backtest.sync_prices --all
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import BACKTEST_END, BACKTEST_START, BACKTEST_YEARS, TICKERS
from .price_history import coverage_summary, sync_ticker_history


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sync Yahoo Finance daily bars into market_daily_bars")
    p.add_argument("--ticker", help="Single ticker (e.g. RELIANCE.NS)")
    p.add_argument("--all", action="store_true", help="Sync all configured TICKERS")
    p.add_argument("--start", default=BACKTEST_START, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=BACKTEST_END, help="End date YYYY-MM-DD")
    p.add_argument("--coverage-only", action="store_true", help="Print DB coverage JSON only")
    args = p.parse_args(argv)

    if args.coverage_only:
        tickers = [args.ticker] if args.ticker else list(TICKERS)
        out = [coverage_summary(t) for t in tickers if t]
        print(json.dumps(out if len(out) > 1 else (out[0] if out else {}), indent=2, default=str))
        return 0

    if not args.ticker and not args.all:
        p.error("Provide --ticker or --all")

    tickers = list(TICKERS) if args.all else [args.ticker.strip().upper()]
    print(f"Syncing ~{BACKTEST_YEARS}y daily bars ({args.start} → {args.end})")
    ok = 0
    for t in tickers:
        print(f"\n  {t} …", flush=True)
        try:
            result = sync_ticker_history(t, start=args.start, end=args.end)
            cov = result.get("coverage") or {}
            print(
                f"    downloaded={result.get('downloaded_rows')} "
                f"upserted={result.get('upserted_rows')} "
                f"bars={cov.get('bars')} "
                f"{cov.get('date_from')} → {cov.get('date_to')} "
                f"({cov.get('years')}y)"
            )
            if result.get("ok"):
                ok += 1
        except Exception as exc:
            print(f"    ERROR: {exc}")
    print(f"\nDone: {ok}/{len(tickers)} tickers OK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
