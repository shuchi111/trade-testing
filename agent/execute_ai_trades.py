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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

SWING_TRADER_ROOT = ROOT.parent
load_dotenv(SWING_TRADER_ROOT / ".env.local")
load_dotenv(SWING_TRADER_ROOT / ".env")

import psycopg2

from canonical_decision import resolve_canonical_decision
from db_url import resolve_psycopg2_url
from market_date import market_trade_date
from tradingagents.dataflows.market_data_validator import require_fresh_market_snapshot
from portfolio_db import (
    ADMIN_WALLET_ID,
    evaluate_trailing_stop,
    execute_trade,
    load_holding,
    load_wallet_cash,
    mark_trailing_stop_triggered,
)
from recommendation_bucket import is_overweight, recommendation_bucket
from trading_constraints import (
    buy_transaction_charge_inr,
    max_position_inr,
    min_wallet_cash_reserve_inr,
    sell_transaction_charge_inr,
)

logger = logging.getLogger("execute_ai_trades")

SETTINGS_ID = "00000000-0000-0000-0000-000000000002"


def trade_block_skip_reason(exc: ValueError) -> str:
    message = str(exc).lower()
    if "no open position" in message or "cannot sell" in message:
        return "no_position_to_sell"
    return "insufficient_cash"


def load_settings(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT auto_trade, dry_run
            FROM ai_trading_settings WHERE id = %s
            """,
            (SETTINGS_ID,),
        )
        row = cur.fetchone()
    if not row:
        base = {
            "auto_trade": True,
            "dry_run": False,
        }
    else:
        base = {
            "auto_trade": bool(row[0]),
            "dry_run": bool(row[1]),
        }
    base["max_position_inr"] = max_position_inr()
    return base


def latest_recommendation(conn, ticker: str, trade_date: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, decision, final_trade_decision, reference_price, computed_at
            FROM ai_recommendation_cache
            WHERE UPPER(ticker) = UPPER(%s) AND trade_date = %s::date
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            (ticker, trade_date),
        )
        row = cur.fetchone()
    if not row:
        return None
    return {
        "id": str(row[0]),
        "decision": row[1] or "",
        "final_trade_decision": row[2] or "",
        "reference_price": float(row[3]) if row[3] is not None else None,
    }


def already_executed(conn, ticker: str, trade_date: str, dry_run: bool) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM ai_trade_executions
            WHERE UPPER(ticker) = UPPER(%s) AND trade_date = %s::date AND dry_run = %s
            """,
            (ticker, trade_date, dry_run),
        )
        return cur.fetchone() is not None


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
    if not force and already_executed(conn, ticker, trade_date, dry_run):
        return {"ok": True, "ticker": ticker, "action_taken": "SKIP", "skip_reason": "already_executed"}

    reco = latest_recommendation(conn, ticker, trade_date)
    if not reco:
        log_execution(
            conn, ticker=ticker, trade_date=trade_date, decision="",
            action="SKIP", qty=None, price=None, pnl=None,
            recommendation_id=None, skip_reason="no_recommendation", dry_run=dry_run,
        )
        return {"ok": True, "ticker": ticker, "action_taken": "SKIP", "skip_reason": "no_recommendation"}

    decision = resolve_canonical_decision(
        reco["decision"],
        reco.get("final_trade_decision") or "",
    )
    bucket = recommendation_bucket(decision)
    hold_qty, avg_entry = load_holding(conn, ticker)
    try:
        snapshot = require_fresh_market_snapshot(ticker, trade_date)
    except Exception as exc:
        log_execution(
            conn, ticker=ticker, trade_date=trade_date, decision=decision,
            action="SKIP", qty=None, price=None, pnl=None,
            recommendation_id=reco["id"], skip_reason=f"stale_market_data:{exc}", dry_run=dry_run,
        )
        return {"ok": True, "ticker": ticker, "action_taken": "SKIP", "skip_reason": "stale_market_data"}

    price = reco["reference_price"] or snapshot.latest_close
    if not price or price <= 0:
        log_execution(
            conn, ticker=ticker, trade_date=trade_date, decision=decision,
            action="SKIP", qty=None, price=None, pnl=None,
            recommendation_id=reco["id"], skip_reason="no_price", dry_run=dry_run,
        )
        return {"ok": True, "ticker": ticker, "action_taken": "SKIP", "skip_reason": "no_price"}

    trailing = evaluate_trailing_stop(conn, ticker, price) if hold_qty > 0 else None
    if trailing and trailing.get("status") == "BREACHED":
        action = "SELL"
        qty = hold_qty
        skip_reason = "trailing_stop_5pct"
        executed = False
        pnl = None

        if not settings.get("auto_trade", True) and not force:
            action, qty, skip_reason = "SKIP", 0.0, "auto_trade_disabled"
        elif not dry_run:
            try:
                net_pnl = execute_trade(conn, ticker=ticker, action="SELL", quantity=qty, price=price)
                executed = True
                mark_trailing_stop_triggered(conn, trailing["id"])
                if net_pnl is not None:
                    pnl = net_pnl
                elif avg_entry > 0:
                    pnl = (price - avg_entry) * qty - sell_transaction_charge_inr()
            except ValueError as exc:
                action, qty, skip_reason = "SKIP", 0.0, trade_block_skip_reason(exc)
                logger.warning("Trailing stop sell blocked for %s: %s", ticker, exc)

        log_execution(
            conn, ticker=ticker, trade_date=trade_date, decision=decision,
            action=action, qty=qty if action == "SELL" else None,
            price=price if action == "SELL" else None,
            pnl=pnl, recommendation_id=reco["id"], skip_reason=skip_reason, dry_run=dry_run,
        )
        return {
            "ok": True,
            "ticker": ticker,
            "decision": decision,
            "bucket": bucket,
            "action_taken": action,
            "quantity": qty if action == "SELL" else None,
            "price": price if action == "SELL" else None,
            "executed": executed,
            "dry_run": dry_run,
            "skip_reason": skip_reason,
        }

    cash = load_wallet_cash(conn)
    max_inr = settings.get("max_position_inr") or max_position_inr()
    min_cash_inr = min_wallet_cash_reserve_inr()

    action = "HOLD"
    qty = 0.0
    skip_reason = None
    size_mult = 1.0 if not is_overweight(decision) else 1.0

    current_position_value = hold_qty * price
    room_to_cap = max(0.0, max_inr - current_position_value)

    if bucket == "buy":
        if hold_qty > 0 and not is_overweight(decision):
            action, skip_reason = "SKIP", "already_holding_no_overweight"
        elif room_to_cap < price:
            action, skip_reason = "SKIP", "max_position_cap_reached"
        elif cash <= 0:
            action, skip_reason = "SKIP", "insufficient_cash"
        else:
            buy_charge = buy_transaction_charge_inr()
            if cash <= buy_charge + min_cash_inr:
                action, skip_reason = "SKIP", "insufficient_cash"
            else:
                cash_for_trade = max(0.0, cash - buy_charge - min_cash_inr)
                buy_value = min(cash_for_trade, room_to_cap)
                if buy_value < price:
                    action, skip_reason = "SKIP", "insufficient_cash_for_whole_share"
                else:
                    qty = int(buy_value // price)
                    cost = qty * price
                    if qty <= 0:
                        action, skip_reason = "SKIP", "insufficient_cash_for_whole_share"
                    elif cost + buy_charge + min_cash_inr > cash:
                        qty = int((cash - buy_charge - min_cash_inr) // price)
                        cost = qty * price
                        if qty <= 0:
                            action, skip_reason = "SKIP", "insufficient_cash_for_whole_share"
                        else:
                            action = "BUY"
                    else:
                        action = "BUY"

    elif bucket == "sell":
        if hold_qty <= 0:
            action, skip_reason = "HOLD", "no_position_to_sell"
        else:
            action = "SELL"
            qty = hold_qty

    else:
        action = "HOLD"

    if action in ("BUY", "SELL") and not settings.get("auto_trade", True) and not force:
        action, skip_reason = "SKIP", "auto_trade_disabled"

    executed = False
    pnl = None
    if action in ("BUY", "SELL") and not dry_run:
        try:
            net_pnl = execute_trade(conn, ticker=ticker, action=action, quantity=qty, price=price)
            executed = True
            if action == "SELL":
                if net_pnl is not None:
                    pnl = net_pnl
                elif avg_entry > 0:
                    pnl = (price - avg_entry) * qty - sell_transaction_charge_inr()
        except ValueError as exc:
            action, skip_reason = "SKIP", trade_block_skip_reason(exc)
            logger.warning("Trade blocked for %s: %s", ticker, exc)

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
    p.add_argument("--trade-date", default="", help="YYYY-MM-DD (default: IST market day)")
    p.add_argument("--dry-run", action="store_true", help="Log only, no wallet trades")
    p.add_argument("--execute", action="store_true", help="Actually execute trades (overrides settings dry_run)")
    p.add_argument("--force", action="store_true", help="Re-run even if already executed today")
    args = p.parse_args()

    db_url = resolve_psycopg2_url()
    if not db_url:
        logger.error("Missing DIRECT_URL or DATABASE_URL")
        sys.exit(1)

    trade_date = args.trade_date.strip() or market_trade_date()

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
