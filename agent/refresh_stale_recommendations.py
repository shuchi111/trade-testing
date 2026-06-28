"""
Re-run recommendation when spot vs reference drifts or cache is too old.

Compared prices use yfinance split-adjusted closes where available (see
fetch_last_close). Old rows may still look odd until the next upsert.

Scheduled via GitHub Actions. Uses DATABASE_URL like write_recommendation_cache.

Env:
  PRICE_REFRESH_RATIO — decimal in [0, 1], e.g. 0.03 = ±3%.
  MAX_CACHE_AGE_DAYS — positive integer refresh safety window.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

SWING_TRADER_ROOT = ROOT.parent
load_dotenv(SWING_TRADER_ROOT / ".env.local")
load_dotenv(SWING_TRADER_ROOT / ".env")

import datetime as dt
import psycopg2

from tradingagents.dataflows.market_data_validator import require_fresh_market_snapshot
from write_recommendation_cache import run_single_recommendation


def _parse_refresh_ratio(raw: str) -> float:
    ratio = float(raw)
    if ratio < 0 or ratio > 1:
        print(
            f"PRICE_REFRESH_RATIO must be between 0 and 1 inclusive, got {ratio}",
            file=sys.stderr,
        )
        sys.exit(1)
    return ratio


def _parse_max_age_days(raw: str) -> int:
    n = int(raw)
    if n < 1:
        print(
            f"MAX_CACHE_AGE_DAYS must be a positive integer, got {n}",
            file=sys.stderr,
        )
        sys.exit(1)
    return n


def main() -> None:
    """Refresh stale ai_recommendation_cache rows based on age and price drift."""
    ratio = _parse_refresh_ratio(os.getenv("PRICE_REFRESH_RATIO", "0.03"))
    max_age = _parse_max_age_days(os.getenv("MAX_CACHE_AGE_DAYS", "10"))

    from db_url import resolve_psycopg2_url

    db_url = resolve_psycopg2_url()
    if not db_url:
        print("DIRECT_URL or DATABASE_URL missing", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, trade_date, reference_price,
                       (computed_at AT TIME ZONE 'UTC')::date,
                       holding_quantity, holding_avg_entry
                FROM ai_recommendation_cache
                ORDER BY ticker
                """
            )
            rows = cur.fetchall()
    except Exception as e:
        print(f"[refresh_stale] query failed: {e}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    refreshed_tickers: list[str] = []
    skipped_tickers: list[str] = []

    from market_date import ist_today

    today = ist_today()
    today_str = today.strftime("%Y-%m-%d")

    try:
        for (
            ticker,
            _stored_trade_date,
            ref_px,
            cache_day,
            h_qty,
            h_entry,
        ) in rows:
            try:
                current = require_fresh_market_snapshot(ticker, today_str).latest_close
            except Exception as exc:
                print(
                    f"[refresh_stale] SKIP {ticker}: stale_or_missing_market_data: {exc}",
                    file=sys.stderr,
                )
                skipped_tickers.append(ticker)
                continue
            if isinstance(cache_day, dt.date):
                days_old = (today - cache_day).days
            else:
                days_old = max_age + 1

            need = False
            reason = ""

            if ref_px is None or float(ref_px) <= 0:
                need = True
                reason = "missing_reference_price"
            elif days_old >= max_age:
                need = True
                reason = f"cache_age_days={days_old}>={max_age}"
            elif current is not None and float(ref_px) > 0:
                pct_change = abs(current - float(ref_px)) / float(ref_px)
                if pct_change >= ratio:
                    need = True
                    reason = (
                        f"price_move={pct_change:.4f}>={ratio} "
                        f"(ref={float(ref_px):.6g} spot={current:.6g})"
                    )

            if not need:
                skipped_tickers.append(ticker)
                continue

            print(f"[refresh_stale] {ticker}: {reason}", flush=True)

            result = run_single_recommendation(
                ticker=ticker,
                trade_date=today_str,
                holding_quantity=float(h_qty or 0),
                holding_avg_entry=float(h_entry or 0),
                source="github_action_price_refresh",
                debug=False,
                db_conn=conn,
            )
            if result.get("ok"):
                refreshed_tickers.append(ticker)
            else:
                msg = result.get("error")
                print(
                    f"[refresh_stale] FAIL {ticker}: {msg}",
                    file=sys.stderr,
                )
    finally:
        conn.close()

    msg_done = (
        f"[refresh_stale] done refreshed={len(refreshed_tickers)} "
        f"skipped={len(skipped_tickers)}"
    )
    print(msg_done, flush=True)


if __name__ == "__main__":
    main()
