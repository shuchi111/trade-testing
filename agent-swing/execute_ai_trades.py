"""
Autonomous AI paper-trade executor.

Reads latest AI recommendation + portfolio state, applies risk rules,
and calls execute_wallet_trade (or logs dry-run).

Usage:
  python execute_ai_trades.py --ticker RELIANCE.NS --trade-date 2026-05-29
  python execute_ai_trades.py --all --dry-run
  python execute_ai_trades.py --ticker TCS.NS --execute
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

SWING_TRADER_ROOT = ROOT.parent
load_dotenv(SWING_TRADER_ROOT / ".env.local")
load_dotenv(SWING_TRADER_ROOT / ".env")

import psycopg2  # type: ignore[reportMissingModuleSource]

from db_url import resolve_psycopg2_url
from instrument_policy import (
    is_fractional_ticker,
    min_buy_notional_inr,
    quote_to_inr,
    size_buy_quantity,
)
from market_price import fetch_last_close
from portfolio_db import (
    ADMIN_WALLET_ID,
    MAX_POSITION_INR,
    MIN_WALLET_CASH_RESERVE_INR,
    execute_trade,
    load_holding,
    load_wallet_cash,
)
from recommendation_bucket import is_overweight, recommendation_bucket
from trade_lessons import portfolio_quality_blocks_new_risk, recent_loss_blocks_buy

logger = logging.getLogger("execute_ai_trades")

SETTINGS_ID = "00000000-0000-0000-0000-000000000002"


def load_settings(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT auto_trade, dry_run, max_position_pct, max_positions, min_cash_reserve_pct
            FROM ai_trading_settings WHERE id = %s
            """,
            (SETTINGS_ID,),
        )
        row = cur.fetchone()
    if not row:
        return {
            "auto_trade": True,
            "dry_run": False,
            "max_position_pct": 1.0,
            "max_positions": 5,
            "min_cash_reserve_pct": 0.0,
        }
    return {
        "auto_trade": bool(row[0]),
        "dry_run": bool(row[1]),
        "max_position_pct": float(row[2]),
        "max_positions": int(row[3]),
        "min_cash_reserve_pct": float(row[4]),
    }


def latest_recommendation(conn, ticker: str, trade_date: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, decision, reference_price, computed_at
            FROM ai_recommendation_cache
            WHERE UPPER(ticker) = UPPER(%s) AND trade_date = %s::date
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            (ticker, trade_date),
        )
        row = cur.fetchone()
        if row:
            return {
                "id": str(row[0]),
                "decision": row[1] or "",
                "reference_price": float(row[2]) if row[2] is not None else None,
            }
        cur.execute(
            """
            SELECT id, decision, reference_price, computed_at
            FROM ai_recommendation_cache
            WHERE UPPER(ticker) = UPPER(%s)
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            (ticker,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "decision": row[1] or "",
        "reference_price": float(row[2]) if row[2] is not None else None,
    }


def load_prior_execution(conn, ticker: str, trade_date: str, dry_run: bool) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT recommendation_id, action_taken, decision
            FROM ai_trade_executions
            WHERE UPPER(ticker) = UPPER(%s) AND trade_date = %s::date AND dry_run = %s
            """,
            (ticker, trade_date, dry_run),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "recommendation_id": str(row[0]) if row[0] else None,
        "action_taken": row[1] or "",
        "decision": row[2] or "",
    }


def should_skip_idempotent(
    prior: dict | None,
    *,
    current_reco_id: str,
    current_decision: str,
) -> bool:
    """Return True when today's execution already reflects this recommendation.

    Re-run when the AI decision changed (e.g. morning SELL → afternoon BUY) or when
    a prior SKIP/HOLD can be retried after a new cache row is written.
    """
    if not prior:
        return False
    prior_id = prior.get("recommendation_id")
    prior_action = prior.get("action_taken") or ""
    prior_decision = (prior.get("decision") or "").strip()
    decision = (current_decision or "").strip()

    if prior_id and prior_id == current_reco_id:
        return True
    if prior_action in ("BUY", "SELL") and prior_decision == decision:
        return True
    return False


def log_execution(conn, *, ticker, trade_date, decision, action, qty, price, pnl,
                  recommendation_id, skip_reason, dry_run) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ai_trade_executions (
              wallet_id, ticker, trade_date, decision, action_taken,
              quantity, price, pnl, recommendation_id, skip_reason, dry_run
            ) VALUES (%s, %s, %s::date, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, trade_date, dry_run) DO UPDATE SET
              decision = EXCLUDED.decision,
              action_taken = EXCLUDED.action_taken,
              quantity = EXCLUDED.quantity,
              price = EXCLUDED.price,
              pnl = EXCLUDED.pnl,
              recommendation_id = EXCLUDED.recommendation_id,
              skip_reason = EXCLUDED.skip_reason
            """,
            (
                ADMIN_WALLET_ID,
                ticker.upper(),
                trade_date,
                decision,
                action,
                qty,
                price,
                pnl,
                recommendation_id,
                skip_reason,
                dry_run,
            ),
        )
    conn.commit()


def decide_and_execute(
    conn,
    *,
    ticker: str,
    trade_date: str,
    dry_run: bool,
    settings: dict,
    force: bool = False,
) -> dict:
    ticker = ticker.strip().upper()
    reco = latest_recommendation(conn, ticker, trade_date)
    if not reco:
        log_execution(
            conn, ticker=ticker, trade_date=trade_date, decision="",
            action="SKIP", qty=None, price=None, pnl=None,
            recommendation_id=None, skip_reason="no_recommendation", dry_run=dry_run,
        )
        return {"ok": True, "ticker": ticker, "action_taken": "SKIP", "skip_reason": "no_recommendation"}

    if not force:
        prior = load_prior_execution(conn, ticker, trade_date, dry_run)
        if should_skip_idempotent(
            prior,
            current_reco_id=reco["id"],
            current_decision=reco["decision"],
        ):
            return {
                "ok": True,
                "ticker": ticker,
                "decision": reco["decision"],
                "action_taken": "SKIP",
                "skip_reason": "already_executed",
            }

    decision = reco["decision"]
    bucket = recommendation_bucket(decision)
    hold_qty, avg_entry = load_holding(conn, ticker)
    price = quote_to_inr(ticker, reco["reference_price"] or fetch_last_close(ticker))
    if not price or price <= 0:
        log_execution(
            conn, ticker=ticker, trade_date=trade_date, decision=decision,
            action="SKIP", qty=None, price=None, pnl=None,
            recommendation_id=reco["id"], skip_reason="no_price", dry_run=dry_run,
        )
        return {"ok": True, "ticker": ticker, "action_taken": "SKIP", "skip_reason": "no_price"}

    cash = load_wallet_cash(conn)
    current_position_value = hold_qty * price
    room_to_cap = max(0.0, MAX_POSITION_INR - current_position_value)

    action = "HOLD"
    qty = 0.0
    skip_reason = None
    size_mult = 1.0

    fractional = is_fractional_ticker(ticker)
    min_notional = min_buy_notional_inr(ticker)

    if bucket == "buy":
        try:
            as_of = date.fromisoformat(str(trade_date)[:10])
        except ValueError:
            as_of = date.today()
        blocked, reason = recent_loss_blocks_buy(conn, ticker, as_of=as_of)
        if blocked:
            action, skip_reason = "SKIP", reason
        elif hold_qty > 0 and not is_overweight(decision):
            action, skip_reason = "SKIP", "already_holding_no_overweight"
        else:
            quality_block, quality_reason = portfolio_quality_blocks_new_risk(conn)
            if quality_block and hold_qty <= 0:
                action, skip_reason = "SKIP", quality_reason
            elif not fractional and room_to_cap < price:
                action, skip_reason = "SKIP", "max_position_cap_reached"
            elif fractional and room_to_cap < min_notional:
                action, skip_reason = "SKIP", "max_position_cap_reached"
            elif cash <= MIN_WALLET_CASH_RESERVE_INR:
                action, skip_reason = "SKIP", "insufficient_cash"
            else:
                cash_after_hard_reserve = max(0.0, cash - MIN_WALLET_CASH_RESERVE_INR)
                buy_value = min(cash_after_hard_reserve, room_to_cap)
                if fractional and buy_value < min_notional:
                    action, skip_reason = "SKIP", "below_min_crypto_notional"
                elif not fractional and buy_value < price:
                    action, skip_reason = "SKIP", "insufficient_cash_for_whole_share"
                else:
                    qty = size_buy_quantity(
                        buy_value_inr=buy_value,
                        price_inr=price,
                        fractional=fractional,
                    )
                    cost = qty * price
                    if qty <= 0:
                        skip = (
                            "below_min_crypto_notional"
                            if fractional
                            else "insufficient_cash_for_whole_share"
                        )
                        action, skip_reason = "SKIP", skip
                    elif cost + MIN_WALLET_CASH_RESERVE_INR > cash:
                        qty = size_buy_quantity(
                            buy_value_inr=max(0.0, cash - MIN_WALLET_CASH_RESERVE_INR),
                            price_inr=price,
                            fractional=fractional,
                        )
                        if qty <= 0:
                            skip = (
                                "below_min_crypto_notional"
                                if fractional
                                else "insufficient_cash_for_whole_share"
                            )
                            action, skip_reason = "SKIP", skip
                        else:
                            action = "BUY"
                    else:
                        action = "BUY"

    elif bucket == "sell":
        if hold_qty > 0:
            action = "SELL"
            qty = hold_qty
        else:
            action, skip_reason = "HOLD", "no_position_to_sell"

    else:
        action = "HOLD"

    if action in ("BUY", "SELL") and not settings.get("auto_trade", True) and not force:
        action, skip_reason = "SKIP", "auto_trade_disabled"

    executed = False
    pnl = None
    if action in ("BUY", "SELL") and not dry_run:
        execute_trade(conn, ticker=ticker, action=action, quantity=qty, price=price)
        executed = True
        if action == "SELL" and avg_entry > 0:
            pnl = (price - avg_entry) * qty

    log_execution(
        conn, ticker=ticker, trade_date=trade_date, decision=decision,
        action=action, qty=qty if action in ("BUY", "SELL") else None,
        price=price if action in ("BUY", "SELL") else None,
        pnl=pnl, recommendation_id=reco["id"], skip_reason=skip_reason, dry_run=dry_run,
    )

    return {
        "ok": True,
        "ticker": ticker,
        "decision": decision,
        "bucket": bucket,
        "action_taken": action,
        "quantity": qty if action in ("BUY", "SELL") else None,
        "price": price if action in ("BUY", "SELL") else None,
        "executed": executed,
        "dry_run": dry_run,
        "skip_reason": skip_reason,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Execute AI paper trades from recommendations")
    p.add_argument("--ticker", help="Single ticker (e.g. RELIANCE.NS)")
    p.add_argument("--all", action="store_true", help="All tickers with latest recommendation")
    p.add_argument("--trade-date", default="", help="YYYY-MM-DD (default: UTC today)")
    p.add_argument("--dry-run", action="store_true", help="Log only, no wallet trades")
    p.add_argument("--execute", action="store_true", help="Actually execute trades (overrides settings dry_run)")
    p.add_argument("--force", action="store_true", help="Re-run even if already executed today")
    args = p.parse_args()

    db_url = resolve_psycopg2_url()
    if not db_url:
        logger.error("Missing DIRECT_URL or DATABASE_URL")
        sys.exit(1)

    trade_date = args.trade_date.strip() or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = psycopg2.connect(db_url)
    try:
        settings = load_settings(conn)
        if args.dry_run:
            dry_run = True
        elif args.execute:
            dry_run = False
        else:
            dry_run = settings.get("dry_run", False)

        tickers: list[str] = []
        if args.ticker:
            tickers = [args.ticker.strip().upper()]
        elif args.all:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT UPPER(ticker) FROM ai_recommendation_cache ORDER BY 1"
                )
                tickers = [r[0] for r in cur.fetchall()]
        else:
            logger.error("Provide --ticker or --all")
            sys.exit(1)

        results = []
        for sym in tickers:
            out = decide_and_execute(
                conn, ticker=sym, trade_date=trade_date,
                dry_run=dry_run, settings=settings, force=args.force,
            )
            results.append(out)
            logger.info("%s", out)

        print(json.dumps({"ok": True, "trade_date": trade_date, "dry_run": dry_run, "results": results}))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
