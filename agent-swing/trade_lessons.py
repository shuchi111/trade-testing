"""Persistent lessons from closed paper trades — survives ephemeral cron runners.

Lessons are stored in Postgres ``ai_trade_lessons`` and injected into:
  - ``build_analysis_context`` (prompt context every run)
  - ``FinancialSituationMemory`` (BM25 past-reflections hooks)
  - ``execute_ai_trades`` BUY gates (hard skip after recent same-ticker losses)
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

ADMIN_WALLET_ID = "00000000-0000-0000-0000-000000000001"

# Cool-off after a realized loss on the same ticker before allowing a new BUY.
RECENT_LOSS_COOLDOWN_DAYS = 10
# Minimum absolute INR loss that counts as a "mistake" for cool-off.
MIN_LOSS_INR_FOR_COOLDOWN = 100.0


def _rollback_after_optional_query_error(conn, label: str, exc: Exception) -> None:
    try:
        conn.rollback()
    except Exception as rollback_err:
        logger.warning("%s failed: %s; rollback failed: %s", label, exc, rollback_err)
        return
    logger.warning("%s failed: %s", label, exc)


def ensure_lessons_table(conn) -> None:
    """Create ``ai_trade_lessons`` if the migration has not been applied yet."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_trade_lessons (
              id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
              wallet_id         UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000001',
              ticker            TEXT NOT NULL,
              trade_date        DATE NOT NULL,
              outcome           TEXT NOT NULL DEFAULT 'loss',
              realized_pnl      NUMERIC(18, 4),
              situation         TEXT NOT NULL DEFAULT '',
              lesson            TEXT NOT NULL,
              source_trade_id   UUID,
              source_execution_id UUID,
              decision_at_entry TEXT NOT NULL DEFAULT '',
              created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_trade_lessons_dedupe
              ON ai_trade_lessons (ticker, trade_date, outcome, realized_pnl)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ai_trade_lessons_ticker_date
              ON ai_trade_lessons (ticker, trade_date DESC)
            """
        )
    conn.commit()


def _fmt_inr(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"Rs.{value:,.2f}"


def load_closed_sells_for_reflection(
    conn, *, lookback_days: int = 45, limit: int = 40
) -> list[dict[str, Any]]:
    """Recent closed SELL rows from the live ledger and AI executions."""
    out: list[dict[str, Any]] = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, ticker, trade_time, quantity, price, total_value, realized_pnl
                FROM portfolio_trades
                WHERE wallet_id = %s
                  AND action = 'SELL'
                  AND realized_pnl IS NOT NULL
                  AND trade_time >= (CURRENT_DATE - (%s || ' days')::interval)
                ORDER BY trade_time DESC
                LIMIT %s
                """,
                (ADMIN_WALLET_ID, lookback_days, limit),
            )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_closed_sells_for_reflection", exc)
        rows = []

    for row in rows:
        trade_time = row[2]
        trade_date = trade_time.date() if isinstance(trade_time, datetime) else trade_time
        pnl = float(row[6])
        out.append(
            {
                "id": str(row[0]) if row[0] else None,
                "ticker": str(row[1]).upper(),
                "trade_date": trade_date,
                "quantity": float(row[3] or 0),
                "price": float(row[4] or 0),
                "total_value": float(row[5] or 0),
                "realized_pnl": pnl,
                "outcome": "win" if pnl > 0 else "loss" if pnl < 0 else "flat",
            }
        )

    # Supplement from executor log when portfolio_trades is thin / missing pnl
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, ticker, trade_date, quantity, price, pnl
                FROM ai_trade_executions
                WHERE wallet_id = %s
                  AND action_taken = 'SELL'
                  AND dry_run = false
                  AND pnl IS NOT NULL
                  AND trade_date >= (CURRENT_DATE - (%s || ' days')::interval)
                ORDER BY trade_date DESC, created_at DESC
                LIMIT %s
                """,
                (ADMIN_WALLET_ID, lookback_days, limit),
            )
            exec_rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_closed_sells_executions", exc)
        exec_rows = []

    seen = {(t["ticker"], str(t["trade_date"]), float(t["realized_pnl"])) for t in out}
    for row in exec_rows:
        ticker = str(row[1]).upper()
        trade_date = row[2]
        if isinstance(trade_date, datetime):
            trade_date = trade_date.date()
        pnl = float(row[5])
        key = (ticker, str(trade_date), pnl)
        if key in seen:
            continue
        seen.add(key)
        qty = float(row[3] or 0)
        price = float(row[4] or 0)
        out.append(
            {
                "id": str(row[0]) if row[0] else None,
                "ticker": ticker,
                "trade_date": trade_date,
                "quantity": qty,
                "price": price,
                "total_value": qty * price,
                "realized_pnl": pnl,
                "outcome": "win" if pnl > 0 else "loss" if pnl < 0 else "flat",
            }
        )
    return out[:limit]


def _prior_buy_decision(conn, ticker: str, before: date) -> str:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT decision, bucket
                FROM ai_recommendation_history
                WHERE UPPER(ticker) = UPPER(%s)
                  AND trade_date <= %s::date
                  AND LOWER(bucket) IN ('buy', 'unknown')
                ORDER BY trade_date DESC, computed_at DESC
                LIMIT 1
                """,
                (ticker, before.isoformat()),
            )
            row = cur.fetchone()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "_prior_buy_decision", exc)
        return ""
    if not row:
        return ""
    return f"{row[0] or ''} (bucket={row[1] or ''})".strip()


def _prior_report_snippet(conn, ticker: str, before: date, max_chars: int = 900) -> str:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(full_report, final_trade_decision, decision)
                FROM ai_recommendation_history
                WHERE UPPER(ticker) = UPPER(%s) AND trade_date <= %s::date
                ORDER BY trade_date DESC, computed_at DESC
                LIMIT 1
                """,
                (ticker, before.isoformat()),
            )
            row = cur.fetchone()
    except Exception as exc:
        # full_report may be missing on older schemas — fall back quietly
        _rollback_after_optional_query_error(conn, "_prior_report_snippet", exc)
        return ""
    if not row or not row[0]:
        return ""
    text = re.sub(r"\s+", " ", str(row[0])).strip()
    return text[:max_chars]


def build_rule_lesson(trade: dict[str, Any], *, entry_decision: str, report_snip: str) -> tuple[str, str]:
    """Deterministic veteran-style lesson (no LLM required)."""
    ticker = trade["ticker"]
    pnl = float(trade["realized_pnl"])
    outcome = trade["outcome"]
    qty = float(trade["quantity"])
    price = float(trade["price"])
    trade_date = trade["trade_date"]

    situation = (
        f"{ticker} closed {outcome} on {trade_date}: sold {qty:.0f} @ {_fmt_inr(price)} "
        f"realized PnL {_fmt_inr(pnl)}. Entry stance was: {entry_decision or 'unknown'}. "
        f"Prior report excerpt: {report_snip or 'n/a'}"
    )

    if outcome == "loss":
        lesson = (
            f"MISTAKE LESSON ({ticker}): Realized loss {_fmt_inr(pnl)} on {trade_date}. "
            "Do NOT re-buy this name until weekly structure repairs and risk/reward is clearly "
            f">= 1.50 after the mandatory 5% trail. Cool-off {RECENT_LOSS_COOLDOWN_DAYS} days. "
            "Prefer HOLD/UNDERWEIGHT over revenge BUY. Size only inside remaining per-stock cap "
            "and wallet cash reserve. Review holdings first — never add to a broken thesis."
        )
    elif outcome == "win":
        lesson = (
            f"WIN LESSON ({ticker}): Realized gain {_fmt_inr(pnl)} on {trade_date}. "
            "Repeat only when the same setup quality is present: clear weekly trend, "
            "R:R >= 1.50, and holdings capacity under the Rs.25,000 cap. Do not overtrade a win."
        )
    else:
        lesson = (
            f"FLAT EXIT ({ticker}) on {trade_date}: fees and opportunity cost matter. "
            "Avoid low-conviction churn; wait for a cleaner swing setup."
        )
    return situation, lesson


def upsert_lesson(
    conn,
    *,
    ticker: str,
    trade_date: date,
    outcome: str,
    realized_pnl: float | None,
    situation: str,
    lesson: str,
    source_trade_id: str | None = None,
    decision_at_entry: str = "",
) -> bool:
    """Insert a lesson if not already stored. Returns True when a row was written."""
    ensure_lessons_table(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_trade_lessons (
                  wallet_id, ticker, trade_date, outcome, realized_pnl,
                  situation, lesson, source_trade_id, decision_at_entry
                ) VALUES (%s, %s, %s::date, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, trade_date, outcome, realized_pnl) DO NOTHING
                """,
                (
                    ADMIN_WALLET_ID,
                    ticker.upper(),
                    trade_date.isoformat() if isinstance(trade_date, date) else str(trade_date),
                    outcome,
                    realized_pnl,
                    situation[:8000],
                    lesson[:8000],
                    source_trade_id,
                    (decision_at_entry or "")[:500],
                ),
            )
            written = cur.rowcount > 0
        conn.commit()
        return written
    except Exception as exc:
        # Unique index name may differ if only CREATE TABLE ran — try without conflict target
        try:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ai_trade_lessons (
                      wallet_id, ticker, trade_date, outcome, realized_pnl,
                      situation, lesson, source_trade_id, decision_at_entry
                    )
                    SELECT %s, %s, %s::date, %s, %s, %s, %s, %s, %s
                    WHERE NOT EXISTS (
                      SELECT 1 FROM ai_trade_lessons
                      WHERE UPPER(ticker) = UPPER(%s)
                        AND trade_date = %s::date
                        AND outcome = %s
                        AND COALESCE(realized_pnl, 0) = COALESCE(%s, 0)
                    )
                    """,
                    (
                        ADMIN_WALLET_ID,
                        ticker.upper(),
                        trade_date.isoformat() if isinstance(trade_date, date) else str(trade_date),
                        outcome,
                        realized_pnl,
                        situation[:8000],
                        lesson[:8000],
                        source_trade_id,
                        (decision_at_entry or "")[:500],
                        ticker,
                        trade_date.isoformat() if isinstance(trade_date, date) else str(trade_date),
                        outcome,
                        realized_pnl,
                    ),
                )
                written = cur.rowcount > 0
            conn.commit()
            return written
        except Exception as exc2:
            _rollback_after_optional_query_error(conn, "upsert_lesson", exc2)
            logger.warning("upsert_lesson failed for %s: %s (first: %s)", ticker, exc2, exc)
            return False


def harvest_lessons_from_closed_trades(
    conn, *, lookback_days: int = 45, limit: int = 40
) -> dict[str, Any]:
    """Score recent closed sells and persist rule-based lessons."""
    ensure_lessons_table(conn)
    sells = load_closed_sells_for_reflection(conn, lookback_days=lookback_days, limit=limit)
    written = 0
    skipped = 0
    for trade in sells:
        entry_decision = _prior_buy_decision(conn, trade["ticker"], trade["trade_date"])
        report_snip = _prior_report_snippet(conn, trade["ticker"], trade["trade_date"])
        situation, lesson = build_rule_lesson(
            trade, entry_decision=entry_decision, report_snip=report_snip
        )
        ok = upsert_lesson(
            conn,
            ticker=trade["ticker"],
            trade_date=trade["trade_date"],
            outcome=trade["outcome"],
            realized_pnl=trade["realized_pnl"],
            situation=situation,
            lesson=lesson,
            source_trade_id=trade.get("id"),
            decision_at_entry=entry_decision,
        )
        if ok:
            written += 1
        else:
            skipped += 1
    return {"ok": True, "closed_sells": len(sells), "written": written, "skipped": skipped}


def load_lessons(
    conn,
    *,
    ticker: str | None = None,
    limit: int = 12,
    losses_only: bool = False,
) -> list[dict[str, Any]]:
    ensure_lessons_table(conn)
    try:
        with conn.cursor() as cur:
            if ticker and losses_only:
                cur.execute(
                    """
                    SELECT ticker, trade_date, outcome, realized_pnl, situation, lesson,
                           decision_at_entry, created_at
                    FROM ai_trade_lessons
                    WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s) AND outcome = 'loss'
                    ORDER BY trade_date DESC, created_at DESC
                    LIMIT %s
                    """,
                    (ADMIN_WALLET_ID, ticker, limit),
                )
            elif ticker:
                cur.execute(
                    """
                    SELECT ticker, trade_date, outcome, realized_pnl, situation, lesson,
                           decision_at_entry, created_at
                    FROM ai_trade_lessons
                    WHERE wallet_id = %s AND UPPER(ticker) = UPPER(%s)
                    ORDER BY trade_date DESC, created_at DESC
                    LIMIT %s
                    """,
                    (ADMIN_WALLET_ID, ticker, limit),
                )
            elif losses_only:
                cur.execute(
                    """
                    SELECT ticker, trade_date, outcome, realized_pnl, situation, lesson,
                           decision_at_entry, created_at
                    FROM ai_trade_lessons
                    WHERE wallet_id = %s AND outcome = 'loss'
                    ORDER BY trade_date DESC, created_at DESC
                    LIMIT %s
                    """,
                    (ADMIN_WALLET_ID, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT ticker, trade_date, outcome, realized_pnl, situation, lesson,
                           decision_at_entry, created_at
                    FROM ai_trade_lessons
                    WHERE wallet_id = %s
                    ORDER BY trade_date DESC, created_at DESC
                    LIMIT %s
                    """,
                    (ADMIN_WALLET_ID, limit),
                )
            rows = cur.fetchall()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "load_lessons", exc)
        return []

    return [
        {
            "ticker": str(r[0]).upper(),
            "trade_date": r[1],
            "outcome": r[2] or "",
            "realized_pnl": None if r[3] is None else float(r[3]),
            "situation": r[4] or "",
            "lesson": r[5] or "",
            "decision_at_entry": r[6] or "",
            "created_at": r[7],
        }
        for r in rows
    ]


_HARVESTED_THIS_PROCESS = False


def format_lessons_block(
    conn,
    *,
    ticker: str,
    ticker_limit: int = 6,
    wallet_limit: int = 8,
) -> str:
    """Text block for portfolio context — holdings-aware scar tissue."""
    global _HARVESTED_THIS_PROCESS
    if not _HARVESTED_THIS_PROCESS:
        try:
            harvest_lessons_from_closed_trades(conn, lookback_days=45, limit=30)
            _HARVESTED_THIS_PROCESS = True
        except Exception as exc:
            logger.warning("harvest during format_lessons_block failed: %s", exc)

    ticker_lessons = load_lessons(conn, ticker=ticker, limit=ticker_limit)
    wallet_losses = load_lessons(conn, ticker=None, limit=wallet_limit, losses_only=True)

    lines = [
        "=== LESSONS FROM PAST MISTAKES (MUST OBEY) ===",
        "Act as a 20+ year swing trader: protect capital first, never revenge-trade, "
        "check ALL open holdings before any new BUY, and only take setups with clear "
        "weekly edge and R:R >= 1.50 after the 5% trailing stop.",
    ]

    if ticker_lessons:
        lines.append(f"Lessons specific to {ticker.upper()}:")
        for L in ticker_lessons:
            pnl = _fmt_inr(L["realized_pnl"])
            lines.append(
                f"- [{L['outcome'].upper()} {L['trade_date']}] PnL {pnl}: {L['lesson']}"
            )
    else:
        lines.append(f"No stored lessons yet for {ticker.upper()}.")

    if wallet_losses:
        lines.append("Recent wallet-wide losses (do not repeat patterns):")
        for L in wallet_losses:
            if L["ticker"].upper() == ticker.upper():
                continue
            pnl = _fmt_inr(L["realized_pnl"])
            lines.append(
                f"- {L['ticker']} [{L['trade_date']}] PnL {pnl}: {L['lesson'][:220]}"
            )

    lines.append(
        "If a lesson says cool-off or broken thesis, default to HOLD unless structure "
        "clearly repaired AND holdings capacity exists under the per-stock cap."
    )
    return "\n".join(lines)


def seed_agent_memories(ta: Any, lessons: list[dict[str, Any]]) -> int:
    """Push durable lessons into in-process BM25 memories for this run."""
    if not lessons:
        return 0
    pairs = [
        (L.get("situation") or L.get("lesson") or "", L.get("lesson") or "")
        for L in lessons
        if L.get("lesson")
    ]
    pairs = [(s, lesson) for s, lesson in pairs if lesson]
    if not pairs:
        return 0
    for mem_name in (
        "bull_memory",
        "bear_memory",
        "trader_memory",
        "invest_judge_memory",
        "portfolio_manager_memory",
    ):
        mem = getattr(ta, mem_name, None)
        if mem is not None and hasattr(mem, "add_situations"):
            mem.add_situations(pairs)
    return len(pairs)


def recent_loss_blocks_buy(
    conn,
    ticker: str,
    *,
    as_of: date | None = None,
    cooldown_days: int = RECENT_LOSS_COOLDOWN_DAYS,
) -> tuple[bool, str]:
    """Hard gate: skip new BUY after a meaningful same-ticker loss inside cool-off."""
    as_of = as_of or date.today()
    cutoff = as_of - timedelta(days=cooldown_days)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_time, realized_pnl
                FROM portfolio_trades
                WHERE wallet_id = %s
                  AND UPPER(ticker) = UPPER(%s)
                  AND action = 'SELL'
                  AND realized_pnl IS NOT NULL
                  AND realized_pnl < %s
                  AND trade_time::date >= %s::date
                ORDER BY trade_time DESC
                LIMIT 1
                """,
                (ADMIN_WALLET_ID, ticker, -MIN_LOSS_INR_FOR_COOLDOWN, cutoff.isoformat()),
            )
            row = cur.fetchone()
            if not row:
                cur.execute(
                    """
                    SELECT trade_date, pnl
                    FROM ai_trade_executions
                    WHERE wallet_id = %s
                      AND UPPER(ticker) = UPPER(%s)
                      AND action_taken = 'SELL'
                      AND dry_run = false
                      AND pnl IS NOT NULL
                      AND pnl < %s
                      AND trade_date >= %s::date
                    ORDER BY trade_date DESC, created_at DESC
                    LIMIT 1
                    """,
                    (
                        ADMIN_WALLET_ID,
                        ticker,
                        -MIN_LOSS_INR_FOR_COOLDOWN,
                        cutoff.isoformat(),
                    ),
                )
                row = cur.fetchone()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "recent_loss_blocks_buy", exc)
        return False, ""

    if not row:
        return False, ""
    trade_time, pnl = row[0], float(row[1])
    td = trade_time.date() if isinstance(trade_time, datetime) else trade_time
    return True, (
        f"recent_loss_cooldown:{ticker}:{td}:pnl={pnl:.2f}:"
        f"wait_{cooldown_days}d"
    )


def portfolio_quality_blocks_new_risk(conn) -> tuple[bool, str]:
    """Soft capital-protection gate when live expectancy is deeply negative."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  COUNT(*) FILTER (WHERE action = 'SELL' AND realized_pnl > 0) AS winning_trades,
                  COUNT(*) FILTER (WHERE action = 'SELL' AND realized_pnl < 0) AS losing_trades,
                  COALESCE(SUM(realized_pnl) FILTER (WHERE action = 'SELL' AND realized_pnl > 0), 0) AS gross_profit,
                  COALESCE(ABS(SUM(realized_pnl) FILTER (WHERE action = 'SELL' AND realized_pnl < 0)), 0) AS gross_loss
                FROM portfolio_trades
                WHERE wallet_id = %s
                """,
                (ADMIN_WALLET_ID,),
            )
            row = cur.fetchone()
    except Exception as exc:
        _rollback_after_optional_query_error(conn, "portfolio_quality_blocks_new_risk", exc)
        return False, ""
    if not row:
        return False, ""
    winning_trades = int(row[0] or 0)
    losing_trades = int(row[1] or 0)
    closed = winning_trades + losing_trades
    if closed < 5:
        return False, ""
    gross_profit = float(row[2] or 0)
    gross_loss = float(row[3] or 0)
    avg_win = gross_profit / winning_trades if winning_trades else 0.0
    avg_loss = gross_loss / losing_trades if losing_trades else 0.0
    win_rate = (winning_trades / closed * 100.0) if closed else 0.0
    expectancy = (win_rate / 100.0) * avg_win - (1 - win_rate / 100.0) * avg_loss
    if win_rate < 35.0 and expectancy < -200.0:
        return True, (
            f"portfolio_quality_gate:win_rate={win_rate:.1f}%:"
            f"expectancy={expectancy:.0f}"
        )
    return False, ""
