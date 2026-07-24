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
    ensure_active_trailing_stop,
    evaluate_trailing_stop,
    execute_trade,
    load_holding,
    load_open_holding_tickers,
    load_wallet_cash,
    mark_trailing_stop_triggered,
)
from recommendation_bucket import is_overweight, recommendation_bucket
from trade_lessons import portfolio_quality_blocks_new_risk, recent_loss_blocks_buy
from trading_constraints import (
    confidence_buy_scale,
    min_ai_confidence_pct,
    sell_transaction_charge_inr,
    sized_buy_budget_inr,
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
    """Load latest reco + AI confidence for confidence-proportional BUY sizing."""

    def _from_row(row, *, has_metrics: bool, has_final: bool) -> dict:
        # Column layouts:
        # metrics+final: id, decision, final_trade_decision, reference_price, ai_confidence_pct, risk_reward_ratio
        # metrics only:  id, decision, reference_price, ai_confidence_pct, risk_reward_ratio
        if has_final:
            decision = row[1] or ""
            final_td = row[2] or ""
            ref = row[3]
            conf_idx, rr_idx = 4, 5
        else:
            decision = row[1] or ""
            final_td = ""
            ref = row[2]
            conf_idx, rr_idx = 3, 4

        conf = None
        rr = None
        if has_metrics:
            if row[conf_idx] is not None:
                conf = float(row[conf_idx])
            if len(row) > rr_idx and row[rr_idx] is not None:
                rr = float(row[rr_idx])
        if conf is None:
            try:
                from tradingagents.graph.confidence_extraction import (
                    parse_explicit_confidence_pct,
                )

                conf = parse_explicit_confidence_pct(f"{decision}\n{final_td}")
            except Exception:
                conf = None
        return {
            "id": str(row[0]),
            "decision": decision,
            "final_trade_decision": final_td,
            "reference_price": float(ref) if ref is not None else None,
            "ai_confidence_pct": conf,
            "risk_reward_ratio": rr,
        }

    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT id, decision, final_trade_decision, reference_price,
                       ai_confidence_pct, risk_reward_ratio
                FROM ai_recommendation_cache
                WHERE UPPER(ticker) = UPPER(%s) AND trade_date = %s::date
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                (ticker, trade_date),
            )
            row = cur.fetchone()
            if row:
                return _from_row(row, has_metrics=True, has_final=True)
            cur.execute(
                """
                SELECT id, decision, final_trade_decision, reference_price,
                       ai_confidence_pct, risk_reward_ratio
                FROM ai_recommendation_cache
                WHERE UPPER(ticker) = UPPER(%s)
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                (ticker,),
            )
            row = cur.fetchone()
            if row:
                return _from_row(row, has_metrics=True, has_final=True)
        except Exception:
            conn.rollback()
            try:
                cur.execute(
                    """
                    SELECT id, decision, reference_price, ai_confidence_pct, risk_reward_ratio
                    FROM ai_recommendation_cache
                    WHERE UPPER(ticker) = UPPER(%s) AND trade_date = %s::date
                    ORDER BY computed_at DESC
                    LIMIT 1
                    """,
                    (ticker, trade_date),
                )
                row = cur.fetchone()
                if row:
                    return _from_row(row, has_metrics=True, has_final=False)
                cur.execute(
                    """
                    SELECT id, decision, reference_price, ai_confidence_pct, risk_reward_ratio
                    FROM ai_recommendation_cache
                    WHERE UPPER(ticker) = UPPER(%s)
                    ORDER BY computed_at DESC
                    LIMIT 1
                    """,
                    (ticker,),
                )
                row = cur.fetchone()
                if row:
                    return _from_row(row, has_metrics=True, has_final=False)
            except Exception:
                conn.rollback()
                cur.execute(
                    """
                    SELECT id, decision, reference_price
                    FROM ai_recommendation_cache
                    WHERE UPPER(ticker) = UPPER(%s) AND trade_date = %s::date
                    ORDER BY computed_at DESC
                    LIMIT 1
                    """,
                    (ticker, trade_date),
                )
                row = cur.fetchone()
                if row:
                    return _from_row(row, has_metrics=False, has_final=False)
                cur.execute(
                    """
                    SELECT id, decision, reference_price
                    FROM ai_recommendation_cache
                    WHERE UPPER(ticker) = UPPER(%s)
                    ORDER BY computed_at DESC
                    LIMIT 1
                    """,
                    (ticker,),
                )
                row = cur.fetchone()
                if row:
                    return _from_row(row, has_metrics=False, has_final=False)
    return None


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


def _force_trailing_stop_sell(
    conn,
    *,
    ticker: str,
    trade_date: str,
    dry_run: bool,
    settings: dict,
    force: bool,
    hold_qty: float,
    avg_entry: float,
    price: float,
    decision: str,
    recommendation_id: str | None,
) -> dict | None:
    """If trail breached, force-SELL and return result dict; else None."""
    if hold_qty <= 0 or price <= 0:
        return None

    ensure_active_trailing_stop(
        conn,
        ticker=ticker,
        quantity=hold_qty,
        entry_price=avg_entry,
        latest_price=price,
    )
    trailing = evaluate_trailing_stop(conn, ticker, price)
    if not trailing or trailing.get("status") != "BREACHED":
        return None

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
            conn.commit()
            if net_pnl is not None:
                pnl = float(net_pnl)
            elif avg_entry > 0:
                # execute_trade already nets the fee into DB; mirror that in the log fallback.
                pnl = (price - avg_entry) * qty - sell_transaction_charge_inr()
        except ValueError as exc:
            action, qty, skip_reason = "SKIP", 0.0, trade_block_skip_reason(exc)
            logger.warning("Trailing stop sell blocked for %s: %s", ticker, exc)

    log_execution(
        conn,
        ticker=ticker,
        trade_date=trade_date,
        decision=decision or "TRAILING_STOP",
        action=action,
        qty=qty if action == "SELL" else None,
        price=price if action == "SELL" else None,
        pnl=pnl,
        recommendation_id=recommendation_id,
        skip_reason=skip_reason,
        dry_run=dry_run,
    )
    return {
        "ok": True,
        "ticker": ticker,
        "decision": decision or "TRAILING_STOP",
        "action_taken": action,
        "quantity": qty if action == "SELL" else None,
        "price": price if action == "SELL" else None,
        "executed": executed,
        "dry_run": dry_run,
        "skip_reason": skip_reason,
    }


def enforce_trailing_stops_all(
    conn,
    *,
    trade_date: str,
    dry_run: bool,
    settings: dict,
    force: bool = True,
) -> list[dict]:
    """Scan all open AI holdings and force-SELL any breached 5% trails."""
    results: list[dict] = []
    for ticker in load_open_holding_tickers(conn):
        hold_qty, avg_entry = load_holding(conn, ticker)
        price = quote_to_inr(ticker, fetch_last_close(ticker))
        if not price or price <= 0:
            results.append({"ok": True, "ticker": ticker, "action_taken": "SKIP", "skip_reason": "no_price"})
            continue
        out = _force_trailing_stop_sell(
            conn,
            ticker=ticker,
            trade_date=trade_date,
            dry_run=dry_run,
            settings=settings,
            force=force,
            hold_qty=hold_qty,
            avg_entry=avg_entry,
            price=price,
            decision="TRAILING_STOP",
            recommendation_id=None,
        )
        if out is None:
            results.append({
                "ok": True,
                "ticker": ticker,
                "action_taken": "HOLD",
                "skip_reason": "trail_active",
                "price": price,
            })
        else:
            results.append(out)
    return results


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
    hold_qty, avg_entry = load_holding(conn, ticker)

    ref = reco["reference_price"] if reco else None
    price = quote_to_inr(ticker, ref or fetch_last_close(ticker))

    # Trailing stop runs BEFORE idempotency — prior HOLD/BUY today must not block a trail exit.
    if hold_qty > 0 and price and price > 0:
        stop_out = _force_trailing_stop_sell(
            conn,
            ticker=ticker,
            trade_date=trade_date,
            dry_run=dry_run,
            settings=settings,
            force=force,
            hold_qty=hold_qty,
            avg_entry=avg_entry,
            price=price,
            decision=(reco["decision"] if reco else "TRAILING_STOP"),
            recommendation_id=(reco["id"] if reco else None),
        )
        if stop_out is not None:
            return stop_out

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
                conf = reco.get("ai_confidence_pct")
                size_mult = confidence_buy_scale(conf)
                if size_mult <= 0:
                    action, skip_reason = (
                        "SKIP",
                        f"confidence_below_{min_ai_confidence_pct():.0f}",
                    )
                else:
                    cash_after_hard_reserve = max(0.0, cash - MIN_WALLET_CASH_RESERVE_INR)
                    # ₹25k room_to_cap is a CEILING only; AI confidence scales the fill.
                    # No risk-% / equity sizing — confidence only.
                    buy_value = sized_buy_budget_inr(
                        cash_available=cash_after_hard_reserve,
                        room_to_cap=room_to_cap,
                        confidence_pct=conf,
                    )
                    logger.info(
                        "BUY size %s conf=%s scale=%.2f room=%.0f budget=%.0f",
                        ticker,
                        conf,
                        size_mult,
                        room_to_cap,
                        buy_value,
                    )
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
                                buy_value_inr=max(
                                    0.0,
                                    min(cash - MIN_WALLET_CASH_RESERVE_INR, buy_value),
                                ),
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
        try:
            net_pnl = execute_trade(conn, ticker=ticker, action=action, quantity=qty, price=price)
            executed = True
            if action == "SELL":
                if net_pnl is not None:
                    pnl = float(net_pnl)
                elif avg_entry > 0:
                    pnl = (price - avg_entry) * qty - sell_transaction_charge_inr()
            if action == "BUY":
                ensure_active_trailing_stop(
                    conn,
                    ticker=ticker,
                    quantity=hold_qty + qty,
                    entry_price=avg_entry if avg_entry > 0 else price,
                    latest_price=price,
                )
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
    p.add_argument(
        "--enforce-stops",
        action="store_true",
        help="Scan open holdings and force-SELL any breached 5%% trailing stops",
    )
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

        if args.enforce_stops:
            results = enforce_trailing_stops_all(
                conn,
                trade_date=trade_date,
                dry_run=dry_run,
                settings=settings,
                force=True,
            )
            for out in results:
                logger.info("%s", out)
            print(json.dumps({
                "ok": True,
                "mode": "enforce_stops",
                "trade_date": trade_date,
                "dry_run": dry_run,
                "results": results,
            }))
            return

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
            logger.error("Provide --ticker, --all, or --enforce-stops")
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