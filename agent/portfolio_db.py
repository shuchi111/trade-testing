"""Paper wallet helpers for AI recommendation + execution pipelines."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from trading_constraints import (
    buy_transaction_charge_inr,
    max_position_inr,
    min_wallet_cash_reserve_inr,
    sell_transaction_charge_inr,
    trailing_stop_loss_pct,
    swing_exit_window_days,
    transaction_charge_for_action,
)
from tradingagents.agents.utils.strategy_checks import format_minervini_evidence

ADMIN_WALLET_ID = "00000000-0000-0000-0000-000000000001"
logger = logging.getLogger(__name__)


def _rollback_after_optional_query_error(conn, label: str, exc: Exception) -> None:
    """Clear psycopg2's aborted transaction state after a fallback query fails."""
    try:
        conn.rollback()
    except Exception as rollback_err:
        logger.warning("%s failed: %s; rollback failed: %s", label, exc, rollback_err)
        return
    logger.warning("%s failed: %s", label, exc)


def load_holding(conn, ticker: str) -> tuple[float, float]:
    """Return (quantity, avg_entry) for ticker in admin wallet."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT quantity, avg_entry
            FROM portfolio_holdings
            WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s)
            """,
            (ADMIN_WALLET_ID, ticker),
        )
        row = cur.fetchone()
    if not row:
        return 0.0, 0.0
    return float(row[0] or 0), float(row[1] or 0)


def load_holding_detail(conn, ticker: str) -> dict[str, Any]:
    """Return holding row including entry_time when present."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT quantity, avg_entry, entry_time
            FROM portfolio_holdings
            WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s)
            """,
            (ADMIN_WALLET_ID, ticker),
        )
        row = cur.fetchone()
    if not row:
        return {"quantity": 0.0, "avg_entry": 0.0, "entry_time": None, "holding_since": None}
    entry_time = row[2]
    if isinstance(entry_time, datetime):
        entry_time = entry_time.date()
    holding_since = current_open_position_since(conn, ticker) or entry_time
    return {
        "quantity": float(row[0] or 0),
        "avg_entry": float(row[1] or 0),
        "entry_time": entry_time,
        "holding_since": holding_since,
    }


def load_all_holding_details(conn) -> list[dict[str, Any]]:
    """Return all open holdings with entry dates for portfolio-wide context."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, quantity, avg_entry, entry_time
                FROM portfolio_holdings
                WHERE wallet_id = %s AND quantity > 0
                ORDER BY entry_time ASC, ticker ASC
                """,
                (ADMIN_WALLET_ID,),
            )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_all_holding_details", exc)
        return []

    out: list[dict[str, Any]] = []
    for ticker, qty, avg_entry, entry_time in rows:
        if isinstance(entry_time, datetime):
            entry_time = entry_time.date()
        out.append(
            {
                "ticker": str(ticker).upper(),
                "quantity": float(qty or 0),
                "avg_entry": float(avg_entry or 0),
                "entry_time": entry_time,
                "holding_since": current_open_position_since(conn, str(ticker)),
            }
        )
    return out


def load_latest_reference_prices(conn, tickers: list[str]) -> dict[str, float]:
    """Latest cached reference price per ticker, used only for context estimates."""
    if not tickers:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (UPPER(ticker)) UPPER(ticker), reference_price
                FROM ai_recommendation_cache
                WHERE UPPER(ticker) = ANY(%s) AND reference_price IS NOT NULL
                ORDER BY UPPER(ticker), computed_at DESC
                """,
                ([t.upper() for t in tickers],),
            )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_latest_reference_prices", exc)
        return {}
    return {
        str(ticker).upper(): float(price)
        for ticker, price in rows
        if price is not None and float(price) > 0
    }


def load_wallet_cash(conn) -> float:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_cash FROM wallet_accounts WHERE id = %s",
            (ADMIN_WALLET_ID,),
        )
        row = cur.fetchone()
    return float(row[0]) if row else 0.0


def count_open_positions(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM portfolio_holdings WHERE wallet_id = %s AND quantity > 0",
            (ADMIN_WALLET_ID,),
        )
        row = cur.fetchone()
    return int(row[0]) if row else 0


def portfolio_value(conn, prices: dict[str, float]) -> float:
    """Cash + mark-to-market of holdings."""
    cash = load_wallet_cash(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, quantity, avg_entry FROM portfolio_holdings WHERE wallet_id = %s",
            (ADMIN_WALLET_ID,),
        )
        rows = cur.fetchall()
    holdings_val = 0.0
    for ticker, qty, avg_entry in rows:
        price = prices.get(str(ticker).upper()) or float(avg_entry or 0)
        holdings_val += float(qty) * price
    return cash + holdings_val


def load_recent_portfolio_trades(conn, ticker: str, limit: int = 5) -> list[dict[str, Any]]:
    """Recent live paper trades for one ticker."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT action, trade_time, quantity, price, total_value, realized_pnl
                FROM portfolio_trades
                WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s)
                ORDER BY trade_time DESC
                LIMIT %s
                """,
                (ADMIN_WALLET_ID, ticker, limit),
            )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_recent_portfolio_trades", exc)
        return []
    out = []
    for action, trade_time, qty, price, total_value, pnl in rows:
        ts = trade_time.date() if isinstance(trade_time, datetime) else trade_time
        out.append(
            {
                "action": action,
                "trade_time": ts,
                "quantity": float(qty or 0),
                "price": float(price or 0),
                "total_value": float(total_value or 0),
                "realized_pnl": None if pnl is None else float(pnl),
            }
        )
    return out


def current_open_position_since(conn, ticker: str) -> date | None:
    """Date when the current uninterrupted open position first became positive."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT action, trade_time, quantity
                FROM portfolio_trades
                WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s)
                ORDER BY trade_time ASC
                """,
                (ADMIN_WALLET_ID, ticker),
            )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "current_open_position_since", exc)
        return None

    running_qty = 0.0
    since: date | None = None
    for action, trade_time, qty in rows:
        before = running_qty
        running_qty += float(qty or 0) if str(action).upper() == "BUY" else -float(qty or 0)
        if before <= 0 and running_qty > 0:
            since = trade_time.date() if isinstance(trade_time, datetime) else trade_time
        if running_qty <= 0:
            since = None
    return since


def load_recent_wallet_trades(conn, limit: int = 10) -> list[dict[str, Any]]:
    """Recent live paper trades across the whole wallet."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, action, trade_time, quantity, price, total_value, realized_pnl
                FROM portfolio_trades
                WHERE wallet_id = %s
                ORDER BY trade_time DESC
                LIMIT %s
                """,
                (ADMIN_WALLET_ID, limit),
            )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_recent_wallet_trades", exc)
        return []
    out = []
    for ticker, action, trade_time, qty, price, total_value, pnl in rows:
        ts = trade_time.date() if isinstance(trade_time, datetime) else trade_time
        out.append(
            {
                "ticker": str(ticker).upper(),
                "action": action,
                "trade_time": ts,
                "quantity": float(qty or 0),
                "price": float(price or 0),
                "total_value": float(total_value or 0),
                "realized_pnl": None if pnl is None else float(pnl),
            }
        )
    return out


def load_recent_wallet_transactions(conn, limit: int = 12) -> list[dict[str, Any]]:
    """Recent wallet ledger rows shown in Wallet/Activity views."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT type, amount, balance_after, note, created_at
                FROM wallet_transactions
                WHERE wallet_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (ADMIN_WALLET_ID, limit),
            )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_recent_wallet_transactions", exc)
        return []
    out = []
    for tx_type, amount, balance_after, note, created_at in rows:
        ts = created_at.date() if isinstance(created_at, datetime) else created_at
        out.append(
            {
                "type": str(tx_type or ""),
                "amount": float(amount or 0),
                "balance_after": float(balance_after or 0),
                "note": str(note or ""),
                "created_at": ts,
            }
        )
    return out


def load_recent_ai_trade_executions(
    conn, ticker: str | None = None, limit: int = 12
) -> list[dict[str, Any]]:
    """Recent autonomous execution decisions, including skips, buys, and sells."""
    try:
        with conn.cursor() as cur:
            if ticker:
                cur.execute(
                    """
                    SELECT ticker, trade_date, decision, action_taken, quantity, price, pnl,
                           skip_reason, dry_run, created_at
                    FROM ai_trade_executions
                    WHERE UPPER(ticker) = UPPER(%s)
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (ticker, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT ticker, trade_date, decision, action_taken, quantity, price, pnl,
                           skip_reason, dry_run, created_at
                    FROM ai_trade_executions
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_recent_ai_trade_executions", exc)
        return []
    return [
        {
            "ticker": str(r[0]).upper(),
            "trade_date": r[1],
            "decision": r[2] or "",
            "action_taken": r[3] or "",
            "quantity": None if r[4] is None else float(r[4]),
            "price": None if r[5] is None else float(r[5]),
            "pnl": None if r[6] is None else float(r[6]),
            "skip_reason": r[7] or "",
            "dry_run": bool(r[8]),
            "created_at": r[9],
        }
        for r in rows
    ]


def load_portfolio_trade_quality(conn) -> dict[str, Any]:
    """Dashboard-style realised trade-quality metrics from the live trade ledger."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  COUNT(*) FILTER (WHERE action = 'BUY') AS buy_trades,
                  COUNT(*) FILTER (WHERE action = 'SELL') AS sell_trades,
                  COUNT(*) FILTER (WHERE action = 'SELL' AND realized_pnl > 0) AS winning_trades,
                  COUNT(*) FILTER (WHERE action = 'SELL' AND realized_pnl < 0) AS losing_trades,
                  COALESCE(SUM(realized_pnl) FILTER (WHERE action = 'SELL'), 0) AS realised_pnl,
                  COALESCE(SUM(realized_pnl) FILTER (WHERE action = 'SELL' AND realized_pnl > 0), 0) AS gross_profit,
                  COALESCE(ABS(SUM(realized_pnl) FILTER (WHERE action = 'SELL' AND realized_pnl < 0)), 0) AS gross_loss
                FROM portfolio_trades
                WHERE wallet_id = %s
                """,
                (ADMIN_WALLET_ID,),
            )
            row = cur.fetchone()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_portfolio_trade_quality", exc)
        return {}
    if not row:
        return {}
    buy_trades = int(row[0] or 0)
    sell_trades = int(row[1] or 0)
    winning_trades = int(row[2] or 0)
    losing_trades = int(row[3] or 0)
    closed = winning_trades + losing_trades
    gross_profit = float(row[5] or 0)
    gross_loss = float(row[6] or 0)
    avg_win = gross_profit / winning_trades if winning_trades else 0.0
    avg_loss = gross_loss / losing_trades if losing_trades else 0.0
    win_rate = (winning_trades / closed * 100.0) if closed else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
    expectancy = (win_rate / 100.0) * avg_win - (1 - win_rate / 100.0) * avg_loss if closed else 0.0
    return {
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": win_rate,
        "realised_pnl": float(row[4] or 0),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }


def last_live_buy_date(conn, ticker: str) -> date | None:
    """Most recent filled BUY for ticker from live ledger or executions."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_time FROM portfolio_trades
                WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s) AND action = 'BUY'
                ORDER BY trade_time DESC LIMIT 1
                """,
                (ADMIN_WALLET_ID, ticker),
            )
            row = cur.fetchone()
            if row and row[0]:
                ts = row[0]
                return ts.date() if isinstance(ts, datetime) else ts
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "last_live_buy_date.portfolio_trades", exc)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_date FROM ai_trade_executions
                WHERE UPPER(ticker) = UPPER(%s) AND action_taken = 'BUY' AND dry_run = false
                ORDER BY trade_date DESC LIMIT 1
                """,
                (ticker,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0]
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "last_live_buy_date.ai_trade_executions", exc)
    return None


def days_held(conn, ticker: str, as_of: date) -> int | None:
    """Calendar days since current open holding started; purchase day is day 0."""
    detail = load_holding_detail(conn, ticker)
    entry = detail.get("holding_since") or detail.get("entry_time")
    if detail["quantity"] <= 0:
        return None
    if not entry:
        entry = last_live_buy_date(conn, ticker)
    if not entry:
        return None
    return max(0, (as_of - entry).days)


def load_active_trailing_stop(conn, ticker: str) -> dict[str, Any] | None:
    """Return the active persistent trailing stop for ticker, if present."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, quantity, entry_price, trailing_pct, highest_price, current_stop_price
                FROM portfolio_trailing_stops
                WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s) AND status = 'ACTIVE'
                LIMIT 1
                """,
                (ADMIN_WALLET_ID, ticker),
            )
            row = cur.fetchone()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_active_trailing_stop", exc)
        return None
    if not row:
        return None
    return {
        "id": str(row[0]),
        "quantity": float(row[1] or 0),
        "entry_price": float(row[2] or 0),
        "trailing_pct": float(row[3] or trailing_stop_loss_pct()),
        "highest_price": float(row[4] or 0),
        "current_stop_price": float(row[5] or 0),
    }


def evaluate_trailing_stop(conn, ticker: str, current_price: float) -> dict[str, Any] | None:
    """Update active stop on new highs; return breach details when stop is hit."""
    stop = load_active_trailing_stop(conn, ticker)
    if not stop or current_price <= 0:
        return None

    if current_price <= stop["current_stop_price"]:
        return {"status": "BREACHED", **stop}

    highest = max(stop["highest_price"], current_price)
    pct = stop["trailing_pct"] or trailing_stop_loss_pct()
    next_stop = round(highest * (1 - pct / 100.0), 4)
    if highest != stop["highest_price"] or next_stop != stop["current_stop_price"]:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE portfolio_trailing_stops
                   SET highest_price = %s, current_stop_price = %s, updated_at = now()
                 WHERE id = %s
                """,
                (highest, next_stop, stop["id"]),
            )
        conn.commit()
        stop = {**stop, "highest_price": highest, "current_stop_price": next_stop}
    return {"status": "ACTIVE", **stop}


def mark_trailing_stop_triggered(conn, stop_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE portfolio_trailing_stops
               SET status = 'TRIGGERED', closed_at = now(), updated_at = now()
             WHERE id = %s
            """,
            (stop_id,),
        )


def load_recent_ai_recommendations(conn, ticker: str, limit: int = 8) -> list[dict[str, Any]]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_date, decision, bucket, reference_price
                FROM ai_recommendation_history
                WHERE UPPER(ticker) = UPPER(%s)
                ORDER BY trade_date DESC, computed_at DESC
                LIMIT %s
                """,
                (ticker, limit),
            )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_recent_ai_recommendations", exc)
        return []
    return [
        {
            "trade_date": r[0],
            "decision": r[1] or "",
            "bucket": r[2] or "",
            "reference_price": None if r[3] is None else float(r[3]),
        }
        for r in rows
    ]


def load_backtest_strategy_summaries(conn, ticker: str, limit: int = 8) -> list[dict[str, Any]]:
    """Latest backtest run per strategy for ticker (from bt_strategy_results)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (strategy_name)
                  strategy_name, date_from, date_to, total_return_pct, win_rate_pct,
                  total_trades, max_drawdown_pct, expectancy_pct, avg_holding_days
                FROM bt_strategy_results
                WHERE UPPER(ticker) = UPPER(%s)
                ORDER BY strategy_name, created_at DESC
                """,
                (ticker,),
            )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_backtest_strategy_summaries", exc)
        return []
    out = []
    for row in rows[:limit]:
        out.append(
            {
                "strategy_name": row[0],
                "date_from": row[1],
                "date_to": row[2],
                "total_return_pct": row[3],
                "win_rate_pct": row[4],
                "total_trades": row[5],
                "max_drawdown_pct": row[6],
                "expectancy_pct": row[7],
                "avg_holding_days": row[8],
            }
        )
    return sorted(
        out,
        key=lambda x: float(x["total_return_pct"] or -999),
        reverse=True,
    )


def load_backtest_trades(
    conn, ticker: str, strategy_name: str | None = None, limit: int = 8
) -> list[dict[str, Any]]:
    try:
        with conn.cursor() as cur:
            if strategy_name:
                cur.execute(
                    """
                    SELECT entry_date, exit_date, entry_price, exit_price,
                           return_pct, pnl, entry_reason, exit_reason, is_win, strategy_name
                    FROM bt_trade_log
                    WHERE UPPER(ticker) = UPPER(%s) AND strategy_name = %s
                    ORDER BY entry_date DESC
                    LIMIT %s
                    """,
                    (ticker, strategy_name, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT entry_date, exit_date, entry_price, exit_price,
                           return_pct, pnl, entry_reason, exit_reason, is_win, strategy_name
                    FROM bt_trade_log
                    WHERE UPPER(ticker) = UPPER(%s)
                    ORDER BY entry_date DESC
                    LIMIT %s
                    """,
                    (ticker, limit),
                )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_backtest_trades", exc)
        return []
    return [
        {
            "entry_date": r[0],
            "exit_date": r[1],
            "entry_price": r[2],
            "exit_price": r[3],
            "return_pct": r[4],
            "pnl": r[5],
            "entry_reason": r[6] or "",
            "exit_reason": r[7] or "",
            "is_win": r[8],
            "strategy_name": r[9] or "",
        }
        for r in rows
    ]


def _fmt_inr(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"₹{value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def build_analysis_context(
    conn,
    ticker: str,
    *,
    trade_date: str,
    reference_price: float | None = None,
) -> str:
    """Rich portfolio + history context for LLM agents."""
    ticker = ticker.strip().upper()
    as_of = datetime.strptime(trade_date, "%Y-%m-%d").date()
    cap = max_position_inr()
    exit_window_days = swing_exit_window_days()
    trailing_pct = trailing_stop_loss_pct()

    detail = load_holding_detail(conn, ticker)
    qty = detail["quantity"]
    avg_entry = detail["avg_entry"]
    cash = load_wallet_cash(conn)
    all_holdings = load_all_holding_details(conn)
    all_prices = load_latest_reference_prices(conn, [h["ticker"] for h in all_holdings])
    if reference_price and reference_price > 0:
        all_prices[ticker] = reference_price

    current_value = qty * reference_price if reference_price and qty > 0 else qty * avg_entry if qty > 0 else 0.0
    room_to_cap = max(0.0, cap - current_value)
    held_days = days_held(conn, ticker, as_of)

    lines: list[str] = [
        "=== WEBSITE CONTEXT COVERAGE ===",
        "Use every section below as if reviewing the website tabs before deciding: wallet, holdings, holding dates, live trades, backtests, prior AI history, active stops, and mandatory rules.",
        "",
        "=== LIVE PORTFOLIO ===",
    ]
    total_cost = 0.0
    total_mark = 0.0
    for h in all_holdings:
        h_cost = h["quantity"] * h["avg_entry"]
        h_mark_price = all_prices.get(h["ticker"]) or h["avg_entry"]
        total_cost += h_cost
        total_mark += h["quantity"] * h_mark_price
    lines.append(
        f"Wallet cash: {_fmt_inr(cash)} | Open positions: {len(all_holdings)} | "
        f"Open cost {_fmt_inr(total_cost)} | Estimated holdings value {_fmt_inr(total_mark)} | "
        f"Estimated equity {_fmt_inr(cash + total_mark)}"
    )

    quality = load_portfolio_trade_quality(conn)
    if quality:
        pf = quality["profit_factor"]
        pf_text = "∞" if pf is None and quality["gross_profit"] > 0 else f"{pf:.2f}" if pf is not None else "n/a"
        lines.append(
            "Dashboard trade quality: "
            f"buy trades={quality['buy_trades']}, sell trades={quality['sell_trades']}, "
            f"win rate={quality['win_rate_pct']:.1f}%, realised PnL={_fmt_inr(quality['realised_pnl'])}, "
            f"avg win={_fmt_inr(quality['average_win'])}, avg loss={_fmt_inr(quality['average_loss'])}, "
            f"profit factor={pf_text}, expectancy={_fmt_inr(quality['expectancy'])}"
        )

    lines.append("")
    lines.append("=== ALL OPEN HOLDINGS (purchase date and days held) ===")
    if all_holdings:
        for h in all_holdings:
            h_ticker = h["ticker"]
            h_entry = h.get("holding_since") or h.get("entry_time")
            h_days = None
            if h_entry:
                h_date = h_entry if isinstance(h_entry, date) else None
                if h_date:
                    h_days = max(0, (as_of - h_date).days)
            h_mark_price = all_prices.get(h_ticker) or h["avg_entry"]
            h_cost = h["quantity"] * h["avg_entry"]
            h_value = h["quantity"] * h_mark_price
            h_pnl = h_value - h_cost
            h_room = max(0.0, cap - h_value)
            h_stop = load_active_trailing_stop(conn, h_ticker)
            stop_text = (
                f", trailing stop {_fmt_inr(h_stop['current_stop_price'])}"
                if h_stop else ""
            )
            entry_text = h_entry.isoformat() if hasattr(h_entry, "isoformat") else "unknown"
            days_text = f"{h_days} days" if h_days is not None else "days unknown"
            lines.append(
                f"{h_ticker}: qty {h['quantity']:.0f}, avg {_fmt_inr(h['avg_entry'])}, "
                f"purchased {entry_text}, held {days_text}, mark {_fmt_inr(h_mark_price)}, "
                f"value {_fmt_inr(h_value)}, PnL {_fmt_inr(h_pnl)}, cap room {_fmt_inr(h_room)}{stop_text}"
            )
    else:
        lines.append("No open holdings in wallet.")

    lines.append("")
    lines.append("=== CURRENT TICKER FOCUS ===")
    if qty > 0 and avg_entry > 0:
        unrealized_pct = None
        if reference_price and avg_entry > 0:
            unrealized_pct = (reference_price - avg_entry) / avg_entry * 100.0
        entry = detail.get("holding_since") or detail.get("entry_time")
        entry_str = entry.isoformat() if hasattr(entry, "isoformat") else "unknown"
        hold_str = f"{held_days} days held" if held_days is not None else "hold duration unknown"
        lines.append(
            f"Hold: {qty:.0f} {ticker} @ {_fmt_inr(avg_entry)} "
            f"(opened {entry_str}, {hold_str}, {exit_window_days}-day swing exit window)"
        )
        if unrealized_pct is not None:
            lines.append(f"Unrealized vs entry: {_fmt_pct(unrealized_pct)} at LTP {_fmt_inr(reference_price)}")
        active_stop = load_active_trailing_stop(conn, ticker)
        if active_stop:
            lines.append(
                f"Active mandatory trailing stop: {active_stop['trailing_pct']:.0f}% trail, "
                f"highest {_fmt_inr(active_stop['highest_price'])}, "
                f"stop {_fmt_inr(active_stop['current_stop_price'])}"
            )
    else:
        lines.append(f"No open position in {ticker}.")

    lines.append(
        f"Per-stock cap: {_fmt_inr(cap)} | Room to add on this name: {_fmt_inr(room_to_cap)} "
        f"(keep at least {_fmt_inr(min_wallet_cash_reserve_inr())} cash)"
    )
    txn_buy = buy_transaction_charge_inr()
    txn_sell = sell_transaction_charge_inr()
    if txn_buy > 0 or txn_sell > 0:
        parts: list[str] = []
        if txn_buy > 0:
            parts.append(f"BUY {_fmt_inr(txn_buy)}")
        if txn_sell > 0:
            parts.append(
                f"SELL {_fmt_inr(txn_sell)} (exit penalty: STT + DP + brokerage paper model)"
            )
        lines.append(
            f"Paper transaction charges: {', '.join(parts)} — "
            "cash and sell PnL reflect applicable charges"
        )

    lines.append("")
    try:
        lines.append(format_minervini_evidence(ticker))
    except Exception as exc:
        lines.append("=== MINERVINI STRATEGY EVIDENCE ===")
        lines.append(f"Unavailable from real OHLCV for {ticker}: {exc}")

    live_trades = load_recent_portfolio_trades(conn, ticker)
    lines.append("")
    lines.append("=== LIVE TRADE HISTORY (most recent first) ===")
    if live_trades:
        for t in live_trades:
            pnl_str = "—" if t["realized_pnl"] is None else _fmt_inr(t["realized_pnl"])
            lines.append(
                f"{t['trade_time']} {t['action']} {t['quantity']:.0f} @ {_fmt_inr(t['price'])} "
                f"value {_fmt_inr(t['total_value'])} PnL {pnl_str}"
            )
    else:
        lines.append("No live paper trades recorded for this ticker.")

    wallet_trades = load_recent_wallet_trades(conn)
    lines.append("")
    lines.append("=== RECENT WALLET TRADES (all tickers) ===")
    if wallet_trades:
        for t in wallet_trades:
            pnl_str = "—" if t["realized_pnl"] is None else _fmt_inr(t["realized_pnl"])
            lines.append(
                f"{t['trade_time']} {t['ticker']} {t['action']} {t['quantity']:.0f} "
                f"@ {_fmt_inr(t['price'])} value {_fmt_inr(t['total_value'])} PnL {pnl_str}"
            )
    else:
        lines.append("No wallet-level trade history yet.")

    wallet_transactions = load_recent_wallet_transactions(conn)
    lines.append("")
    lines.append("=== WALLET ACTIVITY LEDGER (cash movements and balances) ===")
    if wallet_transactions:
        for tx in wallet_transactions:
            note = f" note={tx['note']}" if tx["note"] else ""
            lines.append(
                f"{tx['created_at']} {tx['type']} amount {_fmt_inr(tx['amount'])} "
                f"balance_after {_fmt_inr(tx['balance_after'])}{note}"
            )
    else:
        lines.append("No wallet transaction ledger rows found.")

    ticker_execs = load_recent_ai_trade_executions(conn, ticker=ticker, limit=8)
    wallet_execs = load_recent_ai_trade_executions(conn, limit=8)
    lines.append("")
    lines.append("=== AI EXECUTION HISTORY (recommendation outcomes and skips) ===")
    if ticker_execs:
        lines.append(f"Recent executions for {ticker}:")
        for ex in ticker_execs:
            pnl_str = "—" if ex["pnl"] is None else _fmt_inr(ex["pnl"])
            price_str = "—" if ex["price"] is None else _fmt_inr(ex["price"])
            reason = f", reason={ex['skip_reason']}" if ex["skip_reason"] else ""
            dry = " dry-run" if ex["dry_run"] else ""
            lines.append(
                f"{ex['trade_date']} {ex['action_taken']}{dry} qty={ex['quantity']} "
                f"price={price_str} pnl={pnl_str}{reason}"
            )
    else:
        lines.append(f"No AI execution rows for {ticker}.")
    if wallet_execs:
        lines.append("Recent executions across wallet:")
        for ex in wallet_execs:
            pnl_str = "—" if ex["pnl"] is None else _fmt_inr(ex["pnl"])
            reason = f", reason={ex['skip_reason']}" if ex["skip_reason"] else ""
            lines.append(
                f"{ex['trade_date']} {ex['ticker']} {ex['action_taken']} pnl={pnl_str}{reason}"
            )

    summaries = load_backtest_strategy_summaries(conn, ticker)
    lines.append("")
    lines.append("=== BACKTEST STRATEGY SUMMARY (per ticker) ===")
    if summaries:
        best = summaries[0]
        worst = summaries[-1] if len(summaries) > 1 else None
        for s in summaries:
            lines.append(
                f"{s['strategy_name']}: return {_fmt_pct(float(s['total_return_pct']) if s['total_return_pct'] is not None else None)} "
                f"trades={s['total_trades']} win={s['win_rate_pct']}% "
                f"maxDD={s['max_drawdown_pct']}% expectancy={s['expectancy_pct']}% "
                f"({s['date_from']} to {s['date_to']})"
            )
        bt_strategy = best["strategy_name"]
        if worst and worst["strategy_name"] != best["strategy_name"]:
            lines.append(
                f"Best backtest: {best['strategy_name']}; weakest: {worst['strategy_name']} — "
                "avoid repeating high-churn losing patterns (e.g. MACD whipsaw)."
            )
    else:
        lines.append("No backtest results in database — run Backtest Lab for this ticker first.")
        bt_strategy = None

    bt_trades = load_backtest_trades(
        conn, ticker, strategy_name=bt_strategy if summaries else None, limit=6
    )
    lines.append("")
    lines.append("=== BACKTEST TRADE LOG (recent simulated trades) ===")
    if bt_trades:
        for t in bt_trades:
            lines.append(
                f"{t['entry_date']} → {t['exit_date']} {_fmt_pct(float(t['return_pct']) if t['return_pct'] is not None else None)} "
                f"[{t['strategy_name']}] entry: {t['entry_reason'][:80]} | exit: {t['exit_reason'][:80]}"
            )
    else:
        lines.append("No backtest trade log rows for this ticker.")

    recos = load_recent_ai_recommendations(conn, ticker)
    lines.append("")
    lines.append("=== PAST AI RECOMMENDATIONS ===")
    if recos:
        for r in recos:
            lines.append(f"{r['trade_date']} {r['decision']} (bucket={r['bucket']})")
    else:
        lines.append("No prior AI recommendation history.")

    lines.append("")
    lines.append("=== MANDATORY TRADING RULES (executor enforces these) ===")
    lines.append(f"- Maximum {_fmt_inr(cap)} invested per stock (including adds).")
    lines.append(
        f"- Every live BUY has a mandatory {trailing_pct:.0f}% trailing stop; executor sells if LTP breaches it."
    )
    lines.append("- Wallet cash must never go negative; size buys within available cash.")
    lines.append(
        f"- Aim to harvest the best risk-adjusted exit within {exit_window_days} calendar days, "
        "using backtests, live trade history, weekly structure, and current profit."
    )
    lines.append(
        f"- The 90-day window is not a forced sell date; SELL/UNDERWEIGHT is allowed earlier "
        f"when analysis indicates peak profit or thesis-break risk."
    )
    lines.append(
        "- Before recommending SELL, review live trade history and backtest trade dates above; "
        "do not churn like losing backtest strategies with many small whipsaw trades."
    )
    lines.append(
        "- Analyse deeply: tie Rating to backtest evidence, live hold duration, and past AI stance consistency."
    )

    return "\n".join(lines)


def _deduct_transaction_charge(conn, *, ticker: str, action: str) -> float:
    """Withdraw flat transaction charge from wallet (ledger type WITHDRAWAL)."""
    charge = transaction_charge_for_action(action)
    if charge <= 0:
        return 0.0
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE wallet_accounts
               SET current_cash = current_cash - %s, updated_at = now()
             WHERE id = %s
            RETURNING current_cash
            """,
            (charge, ADMIN_WALLET_ID),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Wallet not found: {ADMIN_WALLET_ID}")
        cash_after = float(row[0])
        cur.execute(
            """
            INSERT INTO wallet_transactions (wallet_id, type, amount, balance_after, note)
            VALUES (%s, 'WITHDRAWAL', %s, %s, %s)
            """,
            (
                ADMIN_WALLET_ID,
                charge,
                cash_after,
                f"txn_charge {action} {ticker.upper()}",
            ),
        )
    return charge


def _adjust_latest_sell_pnl(conn, ticker: str, charge: float) -> float | None:
    """Subtract sell-leg charge from the most recent SELL row (net realized PnL)."""
    if charge <= 0:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT realized_pnl FROM portfolio_trades
                WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s) AND action = 'SELL'
                ORDER BY trade_time DESC LIMIT 1
                """,
                (ADMIN_WALLET_ID, ticker),
            )
            row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE portfolio_trades
               SET realized_pnl = realized_pnl - %s
             WHERE id = (
               SELECT id FROM portfolio_trades
               WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s) AND action = 'SELL'
               ORDER BY trade_time DESC LIMIT 1
             )
            RETURNING realized_pnl
            """,
            (charge, ADMIN_WALLET_ID, ticker),
        )
        row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def execute_trade(conn, *, ticker: str, action: str, quantity: float, price: float) -> float | None:
    """
    Execute paper trade via ``execute_wallet_trade``.

    Applies per-leg charges from ``BUY_TRANSACTION_CHARGE_INR`` / ``SELL_TRANSACTION_CHARGE_INR``
    (deducted from cash; SELL ``realized_pnl`` is net of sell charge).
    Returns net ``realized_pnl`` for SELL, else ``None``.
    """
    ticker = ticker.upper()
    charge = transaction_charge_for_action(action)
    if action == "BUY":
        cash = load_wallet_cash(conn)
        cost = quantity * price
        total = cost + charge
        min_cash = min_wallet_cash_reserve_inr()
        if total > max(0.0, cash - min_cash):
            raise ValueError(
                f"Insufficient cash: need {_fmt_inr(total)} "
                f"(trade {_fmt_inr(cost)} + charge {_fmt_inr(charge)}), "
                f"have {_fmt_inr(cash)} and must keep {_fmt_inr(min_cash)}"
            )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT execute_wallet_trade(%s::uuid, %s, %s, %s, %s)",
            (ADMIN_WALLET_ID, ticker, action, quantity, price),
        )
    net_pnl: float | None = None
    if charge > 0:
        _deduct_transaction_charge(conn, ticker=ticker, action=action)
    if action == "SELL":
        net_pnl = _adjust_latest_sell_pnl(conn, ticker, charge)
    conn.commit()
    return net_pnl
