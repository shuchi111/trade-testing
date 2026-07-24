"""Paper wallet helpers for AI recommendation + execution pipelines."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from trading_constraints import (
    min_wallet_cash_reserve_inr,
    transaction_charge_for_action,
)

logger = logging.getLogger(__name__)

ADMIN_WALLET_ID = "00000000-0000-0000-0000-000000000001"
ML_WALLET_ID = "00000000-0000-0000-0000-000000000003"
MAX_POSITION_INR = 25_000.0
MIN_WALLET_CASH_RESERVE_INR = 5_000.0
SWING_EXIT_WINDOW_DAYS = 90
TRAILING_STOP_LOSS_PCT = 5.0
# Treat remainder at or below this as a full exit so we DELETE the holding
# instead of writing quantity≈0 (trips portfolio_holdings_quantity_check).
# Needed for fractional crypto lots; 1e-6 is far below any whole-share NSE lot.
QUANTITY_EPSILON = 1e-6


def load_holding(conn, ticker: str, wallet_id: str = ADMIN_WALLET_ID) -> tuple[float, float]:
    """Return (quantity, avg_entry) for ticker in the given wallet."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT quantity, avg_entry
            FROM portfolio_holdings
            WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s)
            """,
            (wallet_id, ticker),
        )
        row = cur.fetchone()
    if not row:
        return 0.0, 0.0
    return float(row[0] or 0), float(row[1] or 0)


def load_holding_detail(conn, ticker: str) -> dict[str, Any]:
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
    except Exception:
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


def load_wallet_cash(conn, wallet_id: str = ADMIN_WALLET_ID) -> float:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_cash FROM wallet_accounts WHERE id = %s",
            (wallet_id,),
        )
        row = cur.fetchone()
    return float(row[0]) if row else 0.0


def load_latest_reference_prices(conn, tickers: list[str]) -> dict[str, float]:
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
    except Exception:
        return {}
    return {
        str(ticker).upper(): float(price)
        for ticker, price in rows
        if price is not None and float(price) > 0
    }


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


def days_held_for_entry(entry_time: Any, as_of: date) -> int | None:
    if not entry_time:
        return None
    if isinstance(entry_time, datetime):
        entry_time = entry_time.date()
    if not isinstance(entry_time, date):
        return None
    return max(0, (as_of - entry_time).days)


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
    except Exception:
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


def load_active_trailing_stop(conn, ticker: str) -> dict[str, Any] | None:
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
    except Exception:
        return None
    if not row:
        return None
    return {
        "id": str(row[0]),
        "quantity": float(row[1] or 0),
        "entry_price": float(row[2] or 0),
        "trailing_pct": float(row[3] or TRAILING_STOP_LOSS_PCT),
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
    pct = stop["trailing_pct"] or TRAILING_STOP_LOSS_PCT
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


def load_recent_wallet_trades(conn, limit: int = 10) -> list[dict[str, Any]]:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
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
    """Portfolio-wide context used by TradingAgents recommendations."""
    ticker = ticker.strip().upper()
    as_of = datetime.strptime(trade_date, "%Y-%m-%d").date()
    cash = load_wallet_cash(conn)
    all_holdings = load_all_holding_details(conn)
    prices = load_latest_reference_prices(conn, [h["ticker"] for h in all_holdings])
    if reference_price and reference_price > 0:
        prices[ticker] = reference_price

    total_cost = 0.0
    total_mark = 0.0
    lines: list[str] = [
        "=== WEBSITE CONTEXT COVERAGE ===",
        "Use every section below as if reviewing the website tabs before deciding: wallet, holdings, holding dates, live trades, backtests, prior AI history, active stops, and mandatory rules.",
        "",
        "=== LIVE PORTFOLIO ===",
    ]
    for h in all_holdings:
        mark = prices.get(h["ticker"]) or h["avg_entry"]
        total_cost += h["quantity"] * h["avg_entry"]
        total_mark += h["quantity"] * mark
    lines.append(
        f"Wallet cash: {_fmt_inr(cash)} | Open positions: {len(all_holdings)} | "
        f"Open cost {_fmt_inr(total_cost)} | Estimated holdings value {_fmt_inr(total_mark)} | "
        f"Estimated equity {_fmt_inr(cash + total_mark)}"
    )

    lines.append("")
    lines.append("=== ALL OPEN HOLDINGS (purchase date and days held) ===")
    if all_holdings:
        for h in all_holdings:
            mark = prices.get(h["ticker"]) or h["avg_entry"]
            value = h["quantity"] * mark
            cost = h["quantity"] * h["avg_entry"]
            entry = h.get("holding_since") or h.get("entry_time")
            held_days = days_held_for_entry(entry, as_of)
            entry_text = entry.isoformat() if hasattr(entry, "isoformat") else "unknown"
            days_text = f"{held_days} days" if held_days is not None else "days unknown"
            stop = load_active_trailing_stop(conn, h["ticker"])
            stop_text = f", trailing stop {_fmt_inr(stop['current_stop_price'])}" if stop else ""
            lines.append(
                f"{h['ticker']}: qty {h['quantity']:.0f}, avg {_fmt_inr(h['avg_entry'])}, "
                f"purchased {entry_text}, held {days_text}, mark {_fmt_inr(mark)}, "
                f"value {_fmt_inr(value)}, PnL {_fmt_inr(value - cost)}, "
                f"cap room {_fmt_inr(max(0.0, MAX_POSITION_INR - value))}{stop_text}"
            )
    else:
        lines.append("No open holdings in wallet.")

    focus = load_holding_detail(conn, ticker)
    focus_entry = focus.get("holding_since") or focus.get("entry_time")
    focus_days = days_held_for_entry(focus_entry, as_of)
    lines.append("")
    lines.append("=== CURRENT TICKER FOCUS ===")
    if focus["quantity"] > 0 and focus["avg_entry"] > 0:
        entry_text = focus_entry.isoformat() if hasattr(focus_entry, "isoformat") else "unknown"
        mark = reference_price or focus["avg_entry"]
        unrealized_pct = ((mark - focus["avg_entry"]) / focus["avg_entry"]) * 100
        lines.append(
            f"Hold: {focus['quantity']:.0f} {ticker} @ {_fmt_inr(focus['avg_entry'])} "
            f"(purchased {entry_text}, {focus_days if focus_days is not None else 'unknown'} days held, "
            f"{SWING_EXIT_WINDOW_DAYS}-day swing exit window)"
        )
        lines.append(f"Unrealized vs entry: {_fmt_pct(unrealized_pct)} at LTP {_fmt_inr(mark)}")
    else:
        lines.append(f"No open position in {ticker}.")

    live_trades = load_recent_portfolio_trades(conn, ticker)
    lines.append("")
    lines.append("=== LIVE TRADE HISTORY (most recent first) ===")
    if live_trades:
        for t in live_trades:
            pnl = "—" if t["realized_pnl"] is None else _fmt_inr(t["realized_pnl"])
            lines.append(
                f"{t['trade_time']} {t['action']} {t['quantity']:.0f} @ {_fmt_inr(t['price'])} "
                f"value {_fmt_inr(t['total_value'])} PnL {pnl}"
            )
    else:
        lines.append("No live paper trades recorded for this ticker.")

    trades = load_recent_wallet_trades(conn)
    lines.append("")
    lines.append("=== RECENT WALLET TRADES (all tickers) ===")
    if trades:
        for t in trades:
            pnl = "—" if t["realized_pnl"] is None else _fmt_inr(t["realized_pnl"])
            lines.append(
                f"{t['trade_time']} {t['ticker']} {t['action']} {t['quantity']:.0f} "
                f"@ {_fmt_inr(t['price'])} value {_fmt_inr(t['total_value'])} PnL {pnl}"
            )
    else:
        lines.append("No wallet-level trade history yet.")

    summaries = load_backtest_strategy_summaries(conn, ticker)
    lines.append("")
    lines.append("=== BACKTEST STRATEGY SUMMARY (per ticker) ===")
    bt_strategy = None
    if summaries:
        best = summaries[0]
        worst = summaries[-1] if len(summaries) > 1 else None
        for s in summaries:
            lines.append(
                f"{s['strategy_name']}: return {_fmt_pct(float(s['total_return_pct']) if s['total_return_pct'] is not None else None)} "
                f"trades={s['total_trades']} win={s['win_rate_pct']}% "
                f"maxDD={s['max_drawdown_pct']}% expectancy={s['expectancy_pct']}% "
                f"avgHold={s['avg_holding_days']}d ({s['date_from']} to {s['date_to']})"
            )
        bt_strategy = best["strategy_name"]
        if worst and worst["strategy_name"] != best["strategy_name"]:
            lines.append(
                f"Best backtest: {best['strategy_name']}; weakest: {worst['strategy_name']} — "
                "avoid repeating high-churn losing patterns."
            )
    else:
        lines.append("No backtest results in database — run Backtest Lab for this ticker first.")

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
    lines.append("=== HOLDINGS DISCIPLINE CHECKLIST (complete before any BUY) ===")
    focus_qty = float(focus.get("quantity") or 0)
    if focus_qty > 0:
        lines.append(
            f"Already holding {ticker}: do NOT recommend fresh BUY unless OVERWEIGHT with "
            "explicit cap room and a repaired weekly thesis."
        )
    else:
        lines.append(
            f"No open lot in {ticker}: new BUY only if wallet reserve, per-stock cap, "
            "and lessons/cool-off allow it."
        )
    try:
        from trade_lessons import format_lessons_block

        lines.append("")
        lines.append(format_lessons_block(conn, ticker=ticker))
    except Exception as exc:
        logger.warning("format_lessons_block failed: %s", exc)
        lines.append("")
        lines.append("=== LESSONS FROM PAST MISTAKES ===")
        lines.append("Lessons unavailable this run — still avoid revenge trades after losses.")

    lines.append("")
    lines.append("=== MANDATORY TRADING RULES AND STRATEGY CONTEXT ===")
    lines.append(f"- Maximum {_fmt_inr(MAX_POSITION_INR)} invested per stock; keep at least {_fmt_inr(MIN_WALLET_CASH_RESERVE_INR)} cash.")
    lines.append(f"- Every live BUY has a mandatory {TRAILING_STOP_LOSS_PCT:.0f}% trailing stop.")
    lines.append(
        f"- Aim to harvest the best risk-adjusted exit within {SWING_EXIT_WINDOW_DAYS} calendar days; "
        "this is not a forced sell date and not a minimum hold."
    )
    lines.append("- Before SELL, review all holdings, wallet cash, live trades, backtests, and past AI stance consistency.")
    lines.append("- After a realized loss on this ticker, cool off before re-buying (executor enforces cooldown).")
    return "\n".join(lines)


def _deduct_transaction_charge(
    conn, *, ticker: str, action: str, wallet_id: str = ADMIN_WALLET_ID
) -> float:
    """Withdraw flat transaction charge from wallet (ledger type WITHDRAWAL).

    Parameters
    ----------
    conn
        Open psycopg2 connection. UPDATE + INSERT run in the caller's open
        transaction; ``execute_trade`` commits after this returns. On failure
        the caller must ``rollback()`` so cash and ledger stay consistent.
    ticker
        Instrument symbol included in the ledger ``note``.
    action
        Trade side (``BUY`` / ``SELL``); selects the charge via
        ``transaction_charge_for_action``.
    wallet_id
        Target paper wallet UUID.

    Returns
    -------
    float
        Charge amount deducted (``0.0`` when the side charge is disabled).

    Raises
    ------
    ValueError
        If no ``wallet_accounts`` row exists for ``wallet_id``.

    Examples
    --------
    >>> # charge = _deduct_transaction_charge(conn, ticker="INFY.NS", action="SELL")
    """
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
            (charge, wallet_id),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Wallet account row missing for id={wallet_id}")
        cash_after = float(row[0])
        note = (
            f"Sell fee ₹{charge:,.0f} · {ticker.upper()}"
            if action.upper() == "SELL"
            else f"Buy fee ₹{charge:,.0f} · {ticker.upper()}"
        )
        cur.execute(
            """
            INSERT INTO wallet_transactions (wallet_id, type, amount, balance_after, note)
            VALUES (%s, 'WITHDRAWAL', %s, %s, %s)
            """,
            (
                wallet_id,
                charge,
                cash_after,
                note,
            ),
        )
    return charge


def _adjust_latest_sell_pnl(
    conn, ticker: str, charge: float, wallet_id: str = ADMIN_WALLET_ID
) -> float | None:
    """Subtract sell-leg charge from the most recent SELL row (net realized PnL).

    Parameters
    ----------
    conn
        Open psycopg2 connection in the same transaction as the preceding sell.
    ticker
        Instrument whose latest ``portfolio_trades`` SELL row is adjusted.
    charge
        INR to subtract from ``realized_pnl``. When ``0``, only reads PnL.
    wallet_id
        Paper wallet UUID.

    Returns
    -------
    float | None
        Net ``realized_pnl`` after adjustment, or ``None`` if no SELL row /
        null PnL.

    Notes
    -----
    Must run **after** the SELL row is inserted (by ``execute_wallet_trade`` or
    ``_execute_sell_without_zero_holding_update``). ``execute_trade`` always
    calls this only after that insert in the same uncommitted transaction.
    """
    # Safe only when called after the SELL insert in execute_trade (same txn).
    if charge <= 0:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT realized_pnl FROM portfolio_trades
                WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s) AND action = 'SELL'
                ORDER BY trade_time DESC LIMIT 1
                """,
                (wallet_id, ticker),
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
            (charge, wallet_id, ticker),
        )
        row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _is_holding_quantity_check_violation(exc: BaseException) -> bool:
    """Return True only for the holdings quantity CHECK constraint failure.

    Parameters
    ----------
    exc
        Exception raised while executing a SELL.

    Returns
    -------
    bool
        ``True`` when the error message names
        ``portfolio_holdings_quantity_check`` (not other CHECK constraints).
    """
    message = str(exc).lower()
    return "portfolio_holdings_quantity_check" in message


def _execute_sell_without_zero_holding_update(
    conn,
    *,
    ticker: str,
    quantity: float,
    price: float,
    wallet_id: str = ADMIN_WALLET_ID,
) -> None:
    """Execute a SELL without writing a zero holding quantity.

    Used for full exits (or float dust under ``QUANTITY_EPSILON``) and as a
    fallback when ``execute_wallet_trade`` trips
    ``portfolio_holdings_quantity_check``.

    Parameters
    ----------
    conn
        Open psycopg2 connection; caller commits or rolls back.
    ticker
        Exchange-qualified instrument symbol.
    quantity
        Shares to sell; must not exceed the current holding.
    price
        Execution reference price per share.
    wallet_id
        Paper wallet UUID.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If there is no open position, ``quantity`` exceeds held amount, or the
        wallet row is missing.

    Notes
    -----
    Weighted-average cost basis: on a **partial** sell, ``avg_entry`` is
    intentionally left unchanged (same as ``execute_wallet_trade``). Only
    ``quantity`` is reduced. Realized PnL uses the existing ``avg_entry`` for
    the sold lot; remaining shares keep that same average.
    """
    total = quantity * price
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT avg_entry, quantity
            FROM portfolio_holdings
            WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s)
            FOR UPDATE
            """,
            (wallet_id, ticker),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"No open position to sell for {ticker}")

        avg_entry = float(row[0] or 0)
        held_qty = float(row[1] or 0)
        if quantity > held_qty:
            raise ValueError(
                f"Cannot sell {quantity:g} {ticker}; current holding is {held_qty:g}"
            )

        remaining_qty = held_qty - quantity
        realized = (price - avg_entry) * quantity
        cur.execute(
            """
            UPDATE wallet_accounts
               SET current_cash = current_cash + %s, updated_at = now()
             WHERE id = %s
            RETURNING current_cash
            """,
            (total, wallet_id),
        )
        cash_row = cur.fetchone()
        if not cash_row:
            raise ValueError(f"Wallet account row missing for id={wallet_id}")

        if remaining_qty <= 0:
            cur.execute(
                "DELETE FROM portfolio_holdings WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s)",
                (wallet_id, ticker),
            )
            cur.execute(
                """
                UPDATE portfolio_trailing_stops
                   SET status = 'CANCELLED', closed_at = now(), updated_at = now()
                 WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s) AND status = 'ACTIVE'
                """,
                (wallet_id, ticker),
            )
        else:
            # Avg-cost accounting: keep avg_entry; only reduce quantity
            # (matches execute_wallet_trade partial-sell path).
            cur.execute(
                """
                UPDATE portfolio_holdings
                   SET quantity = %s, updated_at = now()
                 WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s)
                """,
                (remaining_qty, wallet_id, ticker),
            )
            cur.execute(
                """
                UPDATE portfolio_trailing_stops
                   SET quantity = %s, updated_at = now()
                 WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s) AND status = 'ACTIVE'
                """,
                (remaining_qty, wallet_id, ticker),
            )

        cur.execute(
            """
            INSERT INTO portfolio_trades (
              wallet_id, ticker, action, quantity, price, total_value, realized_pnl
            )
            VALUES (%s, %s, 'SELL', %s, %s, %s, %s)
            """,
            (wallet_id, ticker, quantity, price, total, realized),
        )
        cur.execute(
            """
            INSERT INTO wallet_transactions (wallet_id, type, amount, balance_after, note)
            VALUES (%s, 'SELL', %s, %s, %s)
            """,
            (wallet_id, total, cash_row[0], ticker),
        )


def execute_trade(
    conn,
    *,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    wallet_id: str = ADMIN_WALLET_ID,
) -> float | None:
    """Execute a paper trade, deduct per-leg charge, and net SELL PnL.

    Parameters
    ----------
    conn
        Open psycopg2 connection. All steps share one transaction; this
        function commits on success. Callers should not assume intermediate
        state is durable before return.
    ticker
        Exchange-qualified instrument symbol.
    action
        ``BUY`` or ``SELL``.
    quantity
        Share quantity (caller applies whole/fractional policy).
    price
        Validated execution reference price.
    wallet_id
        Paper wallet UUID.

    Returns
    -------
    float | None
        For ``SELL``: net ``realized_pnl`` after the sell-leg charge (may be
        ``None`` only if the trade row has null PnL). For ``BUY``: always
        ``None``. Failures raise; they do not return ``None``.

    Raises
    ------
    ValueError
        Insufficient cash for ``BUY`` (after charge + reserve), no position /
        oversize for ``SELL``, or missing wallet row during charge deduction.
    """
    ticker = ticker.upper()
    charge = transaction_charge_for_action(action)
    use_manual_sell = False
    if action == "BUY":
        cash = load_wallet_cash(conn, wallet_id=wallet_id)
        cost = quantity * price
        total = cost + charge
        min_cash = min_wallet_cash_reserve_inr()
        if total > max(0.0, cash - min_cash):
            raise ValueError(
                f"Insufficient cash: need {_fmt_inr(total)} "
                f"(trade {_fmt_inr(cost)} + charge {_fmt_inr(charge)}), "
                f"have {_fmt_inr(cash)} and must keep {_fmt_inr(min_cash)}"
            )
    elif action == "SELL":
        held_qty, _ = load_holding(conn, ticker, wallet_id=wallet_id)
        if held_qty <= 0:
            raise ValueError(f"No open position to sell for {ticker}")
        if quantity > held_qty:
            raise ValueError(
                f"Cannot sell {quantity:g} {ticker}; current holding is {held_qty:g}"
            )
        use_manual_sell = (held_qty - quantity) <= QUANTITY_EPSILON
    if use_manual_sell:
        _execute_sell_without_zero_holding_update(
            conn, ticker=ticker, quantity=quantity, price=price, wallet_id=wallet_id
        )
    else:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT execute_wallet_trade(%s::uuid, %s, %s, %s, %s)",
                    (wallet_id, ticker, action, quantity, price),
                )
        except Exception as exc:
            if action == "SELL" and _is_holding_quantity_check_violation(exc):
                conn.rollback()
                _execute_sell_without_zero_holding_update(
                    conn,
                    ticker=ticker,
                    quantity=quantity,
                    price=price,
                    wallet_id=wallet_id,
                )
            else:
                raise
    net_pnl: float | None = None
    if charge > 0:
        _deduct_transaction_charge(
            conn, ticker=ticker, action=action, wallet_id=wallet_id
        )
    if action == "SELL":
        net_pnl = _adjust_latest_sell_pnl(conn, ticker, charge, wallet_id=wallet_id)
    conn.commit()
    return net_pnl
