"""
ML paper-trade executor (separate wallet from AI).

Reads ml_signal_cache, applies min_pred_return_pct gate, trades ML wallet …0003.

Usage:
  python execute_ml_trades.py --ticker RELIANCE.NS --dry-run
  python execute_ml_trades.py --all --execute
  python execute_ml_trades.py --all --trade-date 2026-07-18
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parent / ".env.local")
load_dotenv(ROOT.parent / ".env")

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
    MAX_POSITION_INR,
    MIN_WALLET_CASH_RESERVE_INR,
    ML_WALLET_ID,
    execute_trade,
    load_holding,
    load_wallet_cash,
)

logger = logging.getLogger("execute_ml_trades")

SETTINGS_ID = "00000000-0000-0000-0000-000000000004"


def load_settings(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT auto_trade, dry_run, max_position_pct, max_positions,
                   min_cash_reserve_pct, min_pred_return_pct, risk_pct
            FROM ml_trading_settings WHERE id = %s
            """,
            (SETTINGS_ID,),
        )
        row = cur.fetchone()
    if not row:
        return {
            "auto_trade": False,
            "dry_run": True,
            "max_position_pct": 0.10,
            "max_positions": 5,
            "min_cash_reserve_pct": 0.05,
            "min_pred_return_pct": 0.5,
            "risk_pct": 0.02,
        }
    return {
        "auto_trade": bool(row[0]),
        "dry_run": bool(row[1]),
        "max_position_pct": float(row[2]),
        "max_positions": int(row[3]),
        "min_cash_reserve_pct": float(row[4]),
        "min_pred_return_pct": float(row[5]),
        "risk_pct": float(row[6]),
    }


def count_open_positions(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM portfolio_holdings
            WHERE wallet_id = %s AND quantity > 0
            """,
            (ML_WALLET_ID,),
        )
        row = cur.fetchone()
    return int(row[0] or 0) if row else 0


def latest_signal(conn, ticker: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, signal, pred_return_pct, as_of, passes_min_pred,
                   min_pred_return_pct, computed_at, raw
            FROM ml_signal_cache
            WHERE UPPER(ticker) = UPPER(%s)
            """,
            (ticker,),
        )
        row = cur.fetchone()
    if not row:
        return None
    raw = row[7] if isinstance(row[7], dict) else {}
    ref = raw.get("reference_price") if isinstance(raw, dict) else None
    return {
        "id": str(row[0]),
        "signal": (row[1] or "hold").lower(),
        "pred_return_pct": float(row[2]) if row[2] is not None else None,
        "as_of": str(row[3])[:10] if row[3] else None,
        "passes_min_pred": bool(row[4]),
        "min_pred_return_pct": float(row[5]) if row[5] is not None else None,
        "reference_price": float(ref) if ref is not None else None,
    }


def load_prior_execution(conn, ticker: str, trade_date: str, dry_run: bool) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT action_taken FROM ml_trade_executions
            WHERE UPPER(ticker) = UPPER(%s) AND trade_date = %s::date AND dry_run = %s
            """,
            (ticker, trade_date, dry_run),
        )
        row = cur.fetchone()
    if not row:
        return False
    return (row[0] or "") in ("BUY", "SELL", "SKIP", "HOLD")


def log_execution(
    conn,
    *,
    ticker: str,
    trade_date: str,
    signal: str,
    action: str,
    qty,
    price,
    pnl,
    pred_return_pct,
    signal_id,
    skip_reason,
    dry_run: bool,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ml_trade_executions (
              wallet_id, ticker, trade_date, signal, action_taken,
              quantity, price, pnl, pred_return_pct, signal_id, skip_reason, dry_run
            ) VALUES (%s, %s, %s::date, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, trade_date, dry_run) DO UPDATE SET
              signal = EXCLUDED.signal,
              action_taken = EXCLUDED.action_taken,
              quantity = EXCLUDED.quantity,
              price = EXCLUDED.price,
              pnl = EXCLUDED.pnl,
              pred_return_pct = EXCLUDED.pred_return_pct,
              signal_id = EXCLUDED.signal_id,
              skip_reason = EXCLUDED.skip_reason
            """,
            (
                ML_WALLET_ID,
                ticker.upper(),
                trade_date,
                signal,
                action,
                qty,
                price,
                pnl,
                pred_return_pct,
                signal_id,
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
    sig = latest_signal(conn, ticker)
    if not sig:
        log_execution(
            conn,
            ticker=ticker,
            trade_date=trade_date,
            signal="",
            action="SKIP",
            qty=None,
            price=None,
            pnl=None,
            pred_return_pct=None,
            signal_id=None,
            skip_reason="no_signal",
            dry_run=dry_run,
        )
        return {"ok": True, "ticker": ticker, "action_taken": "SKIP", "skip_reason": "no_signal"}

    if not force and load_prior_execution(conn, ticker, trade_date, dry_run):
        return {
            "ok": True,
            "ticker": ticker,
            "action_taken": "SKIP",
            "skip_reason": "already_executed",
        }

    signal = sig["signal"]
    pred = sig["pred_return_pct"]
    min_pred = float(
        settings.get("min_pred_return_pct")
        if settings.get("min_pred_return_pct") is not None
        else (sig.get("min_pred_return_pct") or 0.5)
    )

    hold_qty, avg_entry = load_holding(conn, ticker, wallet_id=ML_WALLET_ID)
    price = quote_to_inr(ticker, sig.get("reference_price") or fetch_last_close(ticker))
    if not price or price <= 0:
        log_execution(
            conn,
            ticker=ticker,
            trade_date=trade_date,
            signal=signal,
            action="SKIP",
            qty=None,
            price=None,
            pnl=None,
            pred_return_pct=pred,
            signal_id=sig["id"],
            skip_reason="no_price",
            dry_run=dry_run,
        )
        return {"ok": True, "ticker": ticker, "action_taken": "SKIP", "skip_reason": "no_price"}

    # Stale: signal as_of older than trade_date by >1 day
    if sig.get("as_of") and str(sig["as_of"]) < trade_date:
        from datetime import date as date_cls

        try:
            d0 = date_cls.fromisoformat(str(sig["as_of"])[:10])
            d1 = date_cls.fromisoformat(trade_date[:10])
            if (d1 - d0).days > 1:
                log_execution(
                    conn,
                    ticker=ticker,
                    trade_date=trade_date,
                    signal=signal,
                    action="SKIP",
                    qty=None,
                    price=None,
                    pnl=None,
                    pred_return_pct=pred,
                    signal_id=sig["id"],
                    skip_reason="stale",
                    dry_run=dry_run,
                )
                return {"ok": True, "ticker": ticker, "action_taken": "SKIP", "skip_reason": "stale"}
        except ValueError:
            pass

    cash = load_wallet_cash(conn, wallet_id=ML_WALLET_ID)
    open_n = count_open_positions(conn)
    max_positions = int(settings.get("max_positions") or 5)
    reserve_pct = float(settings.get("min_cash_reserve_pct") or 0.05)
    reserve = max(MIN_WALLET_CASH_RESERVE_INR, cash * reserve_pct)
    risk_pct = float(settings.get("risk_pct") or 0.02)
    max_pos_pct = float(settings.get("max_position_pct") or 0.10)

    action = "HOLD"
    qty = 0.0
    skip_reason = None

    if signal == "buy":
        if pred is None or pred < min_pred:
            action, skip_reason = "SKIP", "below_min_pred"
        elif hold_qty > 0:
            action, skip_reason = "SKIP", "already_holding"
        elif open_n >= max_positions:
            action, skip_reason = "SKIP", "max_positions"
        elif cash <= reserve:
            action, skip_reason = "SKIP", "no_cash"
        else:
            buy_budget = min(
                max(0.0, cash - reserve),
                cash * max_pos_pct,
                cash * risk_pct * 5,  # soft cap from risk setting
                MAX_POSITION_INR,
            )
            fractional = is_fractional_ticker(ticker)
            min_notional = min_buy_notional_inr(ticker)
            if fractional and buy_budget < min_notional:
                action, skip_reason = "SKIP", "no_cash"
            elif not fractional and buy_budget < price:
                action, skip_reason = "SKIP", "no_cash"
            else:
                qty = size_buy_quantity(
                    buy_value_inr=buy_budget,
                    price_inr=price,
                    fractional=fractional,
                )
                if qty <= 0:
                    action, skip_reason = "SKIP", "no_cash"
                else:
                    action = "BUY"

    elif signal == "sell":
        if hold_qty > 0:
            action = "SELL"
            qty = hold_qty
        else:
            action, skip_reason = "HOLD", "no_position_to_sell"
    else:
        action, skip_reason = "HOLD", "hold"

    if action in ("BUY", "SELL") and not settings.get("auto_trade", False) and not force:
        # Allow --execute CLI to force when settings.auto_trade is false via force or --execute path
        if not force:
            action, skip_reason = "SKIP", "auto_trade_disabled"

    executed = False
    pnl = None
    if action in ("BUY", "SELL") and not dry_run:
        execute_trade(
            conn,
            ticker=ticker,
            action=action,
            quantity=qty,
            price=price,
            wallet_id=ML_WALLET_ID,
        )
        executed = True
        if action == "SELL" and avg_entry > 0:
            pnl = (price - avg_entry) * qty

    log_execution(
        conn,
        ticker=ticker,
        trade_date=trade_date,
        signal=signal,
        action=action,
        qty=qty if action in ("BUY", "SELL") else None,
        price=price if action in ("BUY", "SELL") else None,
        pnl=pnl,
        pred_return_pct=pred,
        signal_id=sig["id"],
        skip_reason=skip_reason,
        dry_run=dry_run,
    )

    return {
        "ok": True,
        "ticker": ticker,
        "signal": signal,
        "pred_return_pct": pred,
        "action_taken": action,
        "quantity": qty if action in ("BUY", "SELL") else None,
        "price": price if action in ("BUY", "SELL") else None,
        "executed": executed,
        "dry_run": dry_run,
        "skip_reason": skip_reason,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Execute ML paper trades from ml_signal_cache")
    p.add_argument("--ticker")
    p.add_argument("--all", action="store_true")
    p.add_argument("--trade-date", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true", help="Live paper (overrides settings dry_run)")
    p.add_argument("--force", action="store_true")
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
            # --execute implies allow trading even if auto_trade false
            settings = {**settings, "auto_trade": True}
        else:
            dry_run = settings.get("dry_run", True)

        if args.ticker:
            tickers = [args.ticker.strip().upper()]
        elif args.all:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT UPPER(ticker) FROM ml_signal_cache ORDER BY 1")
                tickers = [r[0] for r in cur.fetchall()]
        else:
            logger.error("Provide --ticker or --all")
            sys.exit(1)

        cash_before = load_wallet_cash(conn, wallet_id=ML_WALLET_ID)
        results = []
        for sym in tickers:
            out = decide_and_execute(
                conn,
                ticker=sym,
                trade_date=trade_date,
                dry_run=dry_run,
                settings=settings,
                force=args.force or args.execute,
            )
            results.append(out)
            logger.info("%s", out)

        cash_after = load_wallet_cash(conn, wallet_id=ML_WALLET_ID)
        print(
            json.dumps(
                {
                    "ok": True,
                    "trade_date": trade_date,
                    "dry_run": dry_run,
                    "wallet_id": ML_WALLET_ID,
                    "cash_before": cash_before,
                    "cash_after": cash_after,
                    "results": results,
                }
            )
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
