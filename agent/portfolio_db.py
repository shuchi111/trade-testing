"""Paper wallet helpers for AI recommendation + execution pipelines."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from trading_constraints import (
    buy_transaction_charge_inr,
    max_position_inr,
    min_hold_days,
    sell_transaction_charge_inr,
    thesis_break_loss_pct,
    transaction_charge_for_action,
)

ADMIN_WALLET_ID = "00000000-0000-0000-0000-000000000001"


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
        return {"quantity": 0.0, "avg_entry": 0.0, "entry_time": None}
    entry_time = row[2]
    if isinstance(entry_time, datetime):
        entry_time = entry_time.date()
    return {
        "quantity": float(row[0] or 0),
        "avg_entry": float(row[1] or 0),
        "entry_time": entry_time,
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
    except Exception:
        pass
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
    except Exception:
        pass
    return None


def days_held(conn, ticker: str, as_of: date) -> int | None:
    """Calendar days since position entry (holdings.entry_time or last BUY)."""
    detail = load_holding_detail(conn, ticker)
    entry = detail.get("entry_time")
    if detail["quantity"] <= 0:
        return None
    if not entry:
        entry = last_live_buy_date(conn, ticker)
    if not entry:
        return None
    return max(0, (as_of - entry).days)


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
    """Rich portfolio + history context for LLM agents."""
    ticker = ticker.strip().upper()
    as_of = datetime.strptime(trade_date, "%Y-%m-%d").date()
    cap = max_position_inr()
    min_days = min_hold_days()
    stop_pct = thesis_break_loss_pct()

    detail = load_holding_detail(conn, ticker)
    qty = detail["quantity"]
    avg_entry = detail["avg_entry"]
    cash = load_wallet_cash(conn)

    current_value = qty * reference_price if reference_price and qty > 0 else qty * avg_entry if qty > 0 else 0.0
    room_to_cap = max(0.0, cap - current_value)
    held_days = days_held(conn, ticker, as_of)

    lines: list[str] = ["=== LIVE PORTFOLIO ==="]
    if qty > 0 and avg_entry > 0:
        unrealized_pct = None
        if reference_price and avg_entry > 0:
            unrealized_pct = (reference_price - avg_entry) / avg_entry * 100.0
        entry_str = detail["entry_time"].isoformat() if detail.get("entry_time") else "unknown"
        hold_str = f"{held_days} days held" if held_days is not None else "hold duration unknown"
        lines.append(
            f"Hold: {qty:.0f} {ticker} @ {_fmt_inr(avg_entry)} "
            f"(opened {entry_str}, {hold_str}, {min_days}-day minimum hold policy)"
        )
        if unrealized_pct is not None:
            lines.append(f"Unrealized vs entry: {_fmt_pct(unrealized_pct)} at LTP {_fmt_inr(reference_price)}")
    else:
        lines.append(f"No open position in {ticker}.")

    lines.append(
        f"Wallet cash: {_fmt_inr(cash)} | Per-stock cap: {_fmt_inr(cap)} | "
        f"Room to add on this name: {_fmt_inr(room_to_cap)} (wallet must never go negative)"
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
    lines.append("- Wallet cash must never go negative; size buys within available cash.")
    lines.append(
        f"- After a live BUY, minimum {min_days} calendar days before SELL/UNDERWEIGHT "
        f"unless unrealized loss reaches about -{stop_pct:.0f}% versus average entry (thesis break)."
    )
    lines.append(
        "- Before recommending SELL, review live trade history and backtest trade dates above; "
        "do not churn like losing backtest strategies with many small whipsaw trades."
    )
    lines.append(
        "- Analyse deeply: tie Rating to backtest evidence, live hold duration, and past AI stance consistency."
    )

    return "\n".join(lines)


def can_sell_under_min_hold(
    conn,
    ticker: str,
    trade_date: str,
    avg_entry: float,
    current_price: float,
) -> tuple[bool, str | None]:
    """Return (allowed, skip_reason). Allows early exit on thesis-break loss."""
    as_of = datetime.strptime(trade_date, "%Y-%m-%d").date()
    held = days_held(conn, ticker, as_of)
    if held is None:
        return True, None
    min_days = min_hold_days()
    if held >= min_days:
        return True, None
    if avg_entry > 0 and current_price > 0:
        loss_pct = (current_price - avg_entry) / avg_entry * 100.0
        if loss_pct <= -thesis_break_loss_pct():
            return True, None
    return False, "min_hold_period"


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
        if total > cash:
            raise ValueError(
                f"Insufficient cash: need {_fmt_inr(total)} "
                f"(trade {_fmt_inr(cost)} + charge {_fmt_inr(charge)}), have {_fmt_inr(cash)}"
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
