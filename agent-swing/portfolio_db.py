"""Paper wallet helpers for AI recommendation + execution pipelines."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

ADMIN_WALLET_ID = "00000000-0000-0000-0000-000000000001"
ML_WALLET_ID = "00000000-0000-0000-0000-000000000003"
MAX_POSITION_INR = 25_000.0
MIN_WALLET_CASH_RESERVE_INR = 5_000.0
SWING_EXIT_WINDOW_DAYS = 90
TRAILING_STOP_LOSS_PCT = 5.0


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
    """Prior AI stances with signal metrics + short PM excerpt when columns exist."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_date, decision, bucket, reference_price,
                       signal_action, target_price, stop_loss_price,
                       risk_reward_ratio, ai_confidence_pct, final_trade_decision
                FROM ai_recommendation_history
                WHERE UPPER(ticker) = UPPER(%s)
                ORDER BY trade_date DESC, computed_at DESC
                LIMIT %s
                """,
                (ticker, limit),
            )
            rows = cur.fetchall()
    except Exception:
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
            return [
                {
                    "trade_date": r[0],
                    "decision": r[1] or "",
                    "bucket": r[2] or "",
                    "reference_price": None if r[3] is None else float(r[3]),
                    "signal_action": "",
                    "target_price": None,
                    "stop_loss_price": None,
                    "risk_reward_ratio": None,
                    "ai_confidence_pct": None,
                    "excerpt": "",
                }
                for r in rows
            ]
        except Exception:
            return []

    out: list[dict[str, Any]] = []
    for r in rows:
        pm = str(r[9] or "").strip() if len(r) > 9 else ""
        excerpt = " ".join(pm.split())
        if len(excerpt) > 220:
            excerpt = excerpt[:217] + "..."
        out.append(
            {
                "trade_date": r[0],
                "decision": r[1] or "",
                "bucket": r[2] or "",
                "reference_price": None if r[3] is None else float(r[3]),
                "signal_action": (r[4] or "") if len(r) > 4 else "",
                "target_price": None if len(r) <= 5 or r[5] is None else float(r[5]),
                "stop_loss_price": None if len(r) <= 6 or r[6] is None else float(r[6]),
                "risk_reward_ratio": None if len(r) <= 7 or r[7] is None else float(r[7]),
                "ai_confidence_pct": None if len(r) <= 8 or r[8] is None else float(r[8]),
                "excerpt": excerpt,
            }
        )
    return out


def load_recent_ai_trade_executions(
    conn, ticker: str, *, limit: int = 8, include_dry_run: bool = False
) -> list[dict[str, Any]]:
    """Recent autonomous executor fills / skips for this ticker."""
    try:
        with conn.cursor() as cur:
            if include_dry_run:
                cur.execute(
                    """
                    SELECT trade_date, decision, action_taken, quantity, price, pnl,
                           skip_reason, dry_run, created_at
                    FROM ai_trade_executions
                    WHERE UPPER(ticker) = UPPER(%s)
                    ORDER BY trade_date DESC, created_at DESC
                    LIMIT %s
                    """,
                    (ticker, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT trade_date, decision, action_taken, quantity, price, pnl,
                           skip_reason, dry_run, created_at
                    FROM ai_trade_executions
                    WHERE UPPER(ticker) = UPPER(%s) AND dry_run = false
                    ORDER BY trade_date DESC, created_at DESC
                    LIMIT %s
                    """,
                    (ticker, limit),
                )
            rows = cur.fetchall()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "trade_date": r[0],
                "decision": r[1] or "",
                "action_taken": r[2] or "",
                "quantity": None if r[3] is None else float(r[3]),
                "price": None if r[4] is None else float(r[4]),
                "pnl": None if r[5] is None else float(r[5]),
                "skip_reason": r[6] or "",
                "dry_run": bool(r[7]),
                "created_at": r[8],
            }
        )
    return out


def load_portfolio_trade_quality(conn, wallet_id: str = ADMIN_WALLET_ID) -> dict[str, Any]:
    """Book-level closed-trade quality from portfolio_trades (SELL rows with PnL)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  COUNT(*) FILTER (WHERE action = 'SELL' AND realized_pnl IS NOT NULL),
                  COUNT(*) FILTER (WHERE action = 'SELL' AND realized_pnl > 0),
                  COUNT(*) FILTER (WHERE action = 'SELL' AND realized_pnl < 0),
                  COALESCE(SUM(realized_pnl) FILTER (WHERE action = 'SELL'), 0),
                  COALESCE(AVG(realized_pnl) FILTER (WHERE action = 'SELL' AND realized_pnl IS NOT NULL), 0),
                  COALESCE(SUM(realized_pnl) FILTER (WHERE action = 'SELL' AND realized_pnl > 0), 0),
                  COALESCE(SUM(ABS(realized_pnl)) FILTER (WHERE action = 'SELL' AND realized_pnl < 0), 0)
                FROM portfolio_trades
                WHERE wallet_id = %s
                """,
                (wallet_id,),
            )
            row = cur.fetchone()
    except Exception:
        return {}
    if not row:
        return {}
    closed = int(row[0] or 0)
    wins = int(row[1] or 0)
    losses = int(row[2] or 0)
    realized = float(row[3] or 0)
    expectancy = float(row[4] or 0)
    gross_profit = float(row[5] or 0)
    gross_loss = float(row[6] or 0)
    win_rate = (wins / closed * 100.0) if closed else None
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit <= 0 else float("inf"))
    return {
        "closed_trades": closed,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "realized_pnl": realized,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }


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
        "Use every section below as if reviewing the website tabs before deciding: wallet, holdings, holding dates, live trades, AI executions, backtests, prior AI history, active stops, and mandatory rules.",
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

    quality = load_portfolio_trade_quality(conn)
    if quality.get("closed_trades"):
        pf = quality.get("profit_factor")
        if pf is None:
            pf_text = "n/a"
        elif pf == float("inf"):
            pf_text = "∞"
        else:
            pf_text = f"{pf:.2f}"
        lines.append(
            f"Trade quality (closed SELLs): closed={quality['closed_trades']} "
            f"wins={quality['wins']} losses={quality['losses']} "
            f"win_rate={_fmt_pct(quality.get('win_rate_pct')).replace('+', '')} "
            f"realised_PnL={_fmt_inr(quality.get('realized_pnl'))} "
            f"expectancy/trade={_fmt_inr(quality.get('expectancy'))} "
            f"profit_factor={pf_text}"
        )
    else:
        lines.append("Trade quality: no closed SELL PnL rows yet — treat edge as unproven.")

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

    executions = load_recent_ai_trade_executions(conn, ticker, limit=8)
    lines.append("")
    lines.append("=== AI EXECUTION HISTORY (autonomous executor) ===")
    if executions:
        for e in executions:
            qty = "—" if e["quantity"] is None else f"{e['quantity']:.0f}"
            price = "—" if e["price"] is None else _fmt_inr(e["price"])
            pnl = "—" if e["pnl"] is None else _fmt_inr(e["pnl"])
            skip = f" skip={e['skip_reason']}" if e.get("skip_reason") else ""
            lines.append(
                f"{e['trade_date']} decision={e['decision'] or '—'} "
                f"action={e['action_taken']} qty={qty} @ {price} PnL={pnl}{skip}"
            )
    else:
        lines.append("No AI execution rows yet for this ticker.")

    summaries = load_backtest_strategy_summaries(conn, ticker)
    lines.append("")
    lines.append("=== BACKTEST STRATEGY SUMMARY (per ticker) ===")
    best_name = None
    worst_name = None
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
        best_name = best["strategy_name"]
        if worst and worst["strategy_name"] != best["strategy_name"]:
            worst_name = worst["strategy_name"]
            lines.append(
                f"Best backtest: {best_name}; weakest: {worst_name} — "
                "avoid repeating high-churn losing patterns."
            )
    else:
        lines.append("No backtest results in database — run Backtest Lab for this ticker first.")

    lines.append("")
    lines.append("=== BACKTEST TRADE LOG (recent simulated trades) ===")
    bt_best = load_backtest_trades(conn, ticker, strategy_name=best_name, limit=5) if best_name else []
    bt_worst = (
        load_backtest_trades(conn, ticker, strategy_name=worst_name, limit=4)
        if worst_name
        else []
    )
    if not bt_best and not bt_worst:
        bt_best = load_backtest_trades(conn, ticker, strategy_name=None, limit=6)
    if bt_best:
        lines.append(f"-- Best / primary strategy sample ({best_name or 'mixed'}) --")
        for t in bt_best:
            lines.append(
                f"{t['entry_date']} → {t['exit_date']} {_fmt_pct(float(t['return_pct']) if t['return_pct'] is not None else None)} "
                f"[{t['strategy_name']}] entry: {t['entry_reason'][:80]} | exit: {t['exit_reason'][:80]}"
            )
    if bt_worst:
        lines.append(f"-- Weakest strategy sample ({worst_name}) — churn / loss patterns --")
        for t in bt_worst:
            lines.append(
                f"{t['entry_date']} → {t['exit_date']} {_fmt_pct(float(t['return_pct']) if t['return_pct'] is not None else None)} "
                f"[{t['strategy_name']}] entry: {t['entry_reason'][:80]} | exit: {t['exit_reason'][:80]}"
            )
    if not bt_best and not bt_worst:
        lines.append("No backtest trade log rows for this ticker.")

    recos = load_recent_ai_recommendations(conn, ticker)
    lines.append("")
    lines.append("=== PAST AI RECOMMENDATIONS ===")
    if recos:
        for r in recos:
            bits = [f"{r['trade_date']} {r['decision']} (bucket={r['bucket']})"]
            if r.get("signal_action"):
                bits.append(f"action={r['signal_action']}")
            if r.get("reference_price") is not None:
                bits.append(f"ref={_fmt_inr(r['reference_price'])}")
            if r.get("target_price") is not None:
                bits.append(f"target={_fmt_inr(r['target_price'])}")
            if r.get("stop_loss_price") is not None:
                bits.append(f"stop={_fmt_inr(r['stop_loss_price'])}")
            if r.get("risk_reward_ratio") is not None:
                bits.append(f"R:R={r['risk_reward_ratio']:.2f}")
            if r.get("ai_confidence_pct") is not None:
                bits.append(f"conf={r['ai_confidence_pct']:.0f}%")
            lines.append(" | ".join(bits))
            if r.get("excerpt"):
                lines.append(f"  excerpt: {r['excerpt']}")
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

    # Claude Skills Pack (5 screeners + TA/Nifty/VIX/trade plan) — BEFORE any agent signal.
    lines.append("")
    try:
        from tradingagents.agents.utils.claude_skills_pack import build_claude_skills_context

        lines.append(build_claude_skills_context(ticker))
    except Exception as exc:
        logger.warning("claude skills pack failed for %s: %s", ticker, exc)
        lines.append("=== CLAUDE SKILLS PACK (observe BEFORE any Buy/Sell/Hold signal) ===")
        lines.append(f"Skills pack unavailable: {exc}")
        lines.append("Default to HOLD unless other evidence is overwhelming.")
        lines.append("=== END CLAUDE SKILLS PACK ===")

    lines.append("")
    lines.append("=== MANDATORY TRADING RULES AND STRATEGY CONTEXT ===")
    lines.append(f"- Maximum {_fmt_inr(MAX_POSITION_INR)} invested per stock; keep at least {_fmt_inr(MIN_WALLET_CASH_RESERVE_INR)} cash.")
    lines.append(f"- Every live BUY has a mandatory {TRAILING_STOP_LOSS_PCT:.0f}% trailing stop.")
    lines.append(
        f"- Aim to harvest the best risk-adjusted exit within {SWING_EXIT_WINDOW_DAYS} calendar days; "
        "this is not a forced sell date and not a minimum hold."
    )
    lines.append("- Before SELL, review all holdings, wallet cash, live trades, AI executions, backtests, Claude Skills Pack consensus, and past AI stance consistency.")
    lines.append("- After a realized loss on this ticker, cool off before re-buying (executor enforces cooldown).")
    return "\n".join(lines)


def execute_trade(
    conn,
    *,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    wallet_id: str = ADMIN_WALLET_ID,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT execute_wallet_trade(%s::uuid, %s, %s, %s, %s)",
            (wallet_id, ticker.upper(), action, quantity, price),
        )
    conn.commit()
