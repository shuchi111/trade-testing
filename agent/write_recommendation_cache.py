"""
Run TradingAgents propagate for one ticker and upsert into PostgreSQL (Supabase).

Stores ``reference_price`` as the latest split-adjusted close from yfinance when
available (else raw close); keep fetch_last_close in sync for refresh compares.

Used locally and from GitHub Actions. Requires DATABASE_URL and one of:
Z_API_KEY, GLM_API_KEY, or ANTHROPIC_AUTH_TOKEN (with LLM_BACKEND_URL or ANTHROPIC_BASE_URL).

  python write_recommendation_cache.py --ticker TCS.NS \\
      [--trade-date YYYY-MM-DD] [--source github_action_manual]

Env:
    YFINANCE_HISTORY_PERIOD — passed to yfinance ``history(period=...)`` (default 10d).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
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

from db_url import resolve_psycopg2_url
from market_date import adjust_to_last_trading_day, ist_today
from portfolio_db import build_analysis_context, load_holding
from recommendation_bucket import recommendation_bucket
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger("write_recommendation_cache")


def _build_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = os.getenv("LLM_PROVIDER", "glm").strip()
    backend = (
        (os.getenv("LLM_BACKEND_URL") or "").strip()
        or (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
        or "https://api.z.ai/api/paas/v4/"
    )
    config["backend_url"] = backend.rstrip("/")
    config["deep_think_llm"] = os.getenv("DEEP_THINK_LLM", "glm-5.2")
    config["quick_think_llm"] = os.getenv("QUICK_THINK_LLM", "glm-5.2")
    config["max_debate_rounds"] = int(os.getenv("MAX_DEBATE_ROUNDS", "1"))
    config["max_recur_limit"] = int(
        os.getenv("MAX_RECUR_LIMIT", str(config.get("max_recur_limit", 1000)))
    )
    api_key = (
        (os.getenv("Z_API_KEY") or "").strip()
        or (os.getenv("GLM_API_KEY") or "").strip()
        or (os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()
        or (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        or ""
    )
    config["api_key"] = api_key
    config["data_vendors"] = {
        "core_stock_apis": os.getenv("DATA_VENDOR_STOCKS", "yfinance"),
        "technical_indicators": os.getenv("DATA_VENDOR_INDICATORS", "yfinance"),
        "fundamental_data": os.getenv("DATA_VENDOR_FUNDAMENTALS", "yfinance"),
        "news_data": os.getenv("DATA_VENDOR_NEWS", "yfinance"),
    }
    return config


def fetch_last_close(symbol: str, period: str | None = None) -> float | None:
    """Return last quoted close for ``symbol``: split-adjusted if available."""
    import yfinance as yf

    hist_period = (
        (period.strip() if period else None)
        or os.getenv("YFINANCE_HISTORY_PERIOD", "").strip()
        or "10d"
    )

    sym = symbol.upper()
    try:
        t = yf.Ticker(sym)
        hist = t.history(period=hist_period)
        if hist is None or hist.empty:
            return None
        adj = hist["Adj Close"] if "Adj Close" in hist.columns else hist["Close"]
        val = adj.iloc[-1]
        out = float(val) if val is not None else None
        return out if out is not None and out > 0 else None
    except Exception as e:
        logger.error("fetch_last_close %s failed: %s", sym, e)
        return None


def _portfolio_context(
    conn,
    ticker: str,
    trade_date: str,
    holding_qty: float,
    holding_entry: float,
    reference_price: float | None = None,
) -> str:
    """Build rich portfolio + backtest + trade history context for LLM agents."""
    try:
        return build_analysis_context(
            conn,
            ticker,
            trade_date=trade_date,
            reference_price=reference_price,
        )
    except Exception as exc:
        logger.warning("build_analysis_context failed for %s: %s", ticker, exc)
        if holding_qty > 0 and holding_entry > 0:
            return (
                f"You currently hold {holding_qty:.0f} units of {ticker} "
                f"at an average entry price of {holding_entry:,.2f}. "
                "Use this average entry as the percentage basis for swing framing."
            )
        return (
            f"Portfolio tracker: no quantity held reported for {ticker}. "
            "Use fundamentals and risk only."
        )


def upsert_cache_row(
    conn,
    *,
    ticker: str,
    trade_date: str,
    decision: str,
    final_trade_decision: str,
    reference_price: float | None,
    holding_quantity: float,
    holding_avg_entry: float,
    source: str,
) -> None:
    """Append a recommendation row into ``ai_recommendation_cache``.

    Always inserts — no upsert, no overwrite. Each cron run creates a new row
    keyed by its auto-generated UUID.

    Parameters
    ----------
    conn : psycopg2.connection
        Active database connection. Committed on success.
    ticker : str
        Stock ticker symbol (e.g. ``"TCS.NS"``). Uppercased internally.
    trade_date : str
        Date string in ``YYYY-MM-DD`` format.
    decision : str
        Raw LLM decision text.
    final_trade_decision : str
        Processed final trade decision string.
    reference_price : float | None
        Latest split-adjusted close from yfinance, or ``None``.
    holding_quantity : float
        Current position quantity (0 if not held).
    holding_avg_entry : float
        Average entry price for the position (0 if not held).
    source : str
        Provenance tag (e.g. ``"github_action"``).
    """
    sql = """
    INSERT INTO ai_recommendation_cache (
      ticker, trade_date, decision, final_trade_decision, reference_price,
      holding_quantity, holding_avg_entry, source, computed_at
    )
    VALUES (%s, %s::date, %s, %s, %s, %s, %s, %s, now());
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                ticker.upper(),
                trade_date,
                decision or "",
                final_trade_decision or "",
                reference_price,
                holding_quantity,
                holding_avg_entry,
                source,
            ),
        )
    if _is_buy_signal(decision):
        _insert_buy_signal(
            conn,
            ticker=ticker,
            trade_date=trade_date,
            decision=decision,
            reference_price=reference_price,
            source=source,
        )
    _insert_history(
        conn,
        ticker=ticker,
        trade_date=trade_date,
        decision=decision,
        final_trade_decision=final_trade_decision,
        reference_price=reference_price,
        holding_quantity=holding_quantity,
        holding_avg_entry=holding_avg_entry,
        source=source,
    )
    conn.commit()


def _insert_history(
    conn,
    *,
    ticker: str,
    trade_date: str,
    decision: str,
    final_trade_decision: str,
    reference_price: float | None,
    holding_quantity: float,
    holding_avg_entry: float,
    source: str,
) -> None:
    bucket = recommendation_bucket(decision)
    sql = """
    INSERT INTO ai_recommendation_history (
      ticker, trade_date, decision, final_trade_decision, bucket,
      reference_price, holding_quantity, holding_avg_entry, source, computed_at
    )
    VALUES (%s, %s::date, %s, %s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (ticker, trade_date, decision) DO NOTHING;
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                ticker.upper(),
                trade_date,
                decision or "",
                final_trade_decision or "",
                bucket,
                reference_price,
                holding_quantity,
                holding_avg_entry,
                source,
            ),
        )


_BUY_KEYWORDS = frozenset({"BUY", "OVERWEIGHT"})
_SELL_KEYWORDS = frozenset({"SELL", "UNDERWEIGHT"})


def _is_buy_signal(decision: str | None) -> bool:
    """Match BUY/OVERWEIGHT anywhere in the decision string.

    Handles formats like ``"BUY"``, ``"OVERWEIGHT"``, ``"RECOMMENDATION: BUY"``,
    ``"ACTION=BUY"``, ``"BUY **strong**"`` etc. Aligned with the TypeScript
    ``recommendationBucket()`` in ``lib/ai-recommendation/decision-bucket.ts``.
    """
    if not decision:
        return False
    u = decision.strip().upper()
    # Exact full match
    if u in _BUY_KEYWORDS:
        return True
    # First word match
    first = u.split()[0] if u else ""
    if first in _BUY_KEYWORDS:
        return True
    # Keyword appears anywhere (handles "RECOMMENDATION: BUY", "ACTION=BUY" etc.)
    for kw in _BUY_KEYWORDS:
        if kw in u:
            return True
    return False


def _insert_buy_signal(
    conn,
    *,
    ticker: str,
    trade_date: str,
    decision: str,
    reference_price: float | None,
    source: str,
) -> None:
    sql = """
    INSERT INTO ai_buy_signals (ticker, trade_date, decision, reference_price, source, computed_at)
    VALUES (%s, %s::date, %s, %s, %s, now());
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker.upper(), trade_date, decision or "", reference_price, source))


def run_single_recommendation(
    *,
    ticker: str,
    trade_date: str,
    holding_quantity: float = 0.0,
    holding_avg_entry: float = 0.0,
    source: str = "github_action",
    debug: bool = False,
    db_conn=None,
) -> dict:
    """
    Run LLM graph for one ticker and append to ``ai_recommendation_cache``.

    Always inserts a new row (append-only, no upsert). If the decision is
    BUY/OVERWEIGHT, also inserts into ``ai_buy_signals``.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"TCS.NS"``).
    trade_date : str
        Date string in ``YYYY-MM-DD`` format. Should be the last trading day.
    holding_quantity : float
        Current position quantity (0 if not held). Must be >= 0.
    holding_avg_entry : float
        Average entry price (0 if not held). Must be >= 0.
    source : str
        Provenance tag (e.g. ``"github_action"``).
    debug : bool
        Enable verbose LLM graph output.
    db_conn
        Optional open psycopg2 connection; if omitted, opens and closes one
        (caller may pass a shared connection for batch jobs).

    Returns
    -------
    dict
        ``{"ok": bool, "ticker": str, ...}`` with ``error`` when failed.
    """
    ticker = ticker.strip().upper()
    cfg = _build_config()
    if not cfg.get("api_key"):
        return {"ok": False, "error": "Missing Z_API_KEY or GLM_API_KEY", "ticker": ticker}

    db_url = resolve_psycopg2_url()
    if not db_url:
        return {"ok": False, "error": "Missing DIRECT_URL or DATABASE_URL", "ticker": ticker}

    if holding_quantity < 0 or holding_avg_entry < 0:
        return {"ok": False, "error": "holding_quantity and holding_avg_entry must be >= 0", "ticker": ticker}

    reference_price = fetch_last_close(ticker)

    owns_conn = db_conn is None
    conn = psycopg2.connect(db_url) if owns_conn else db_conn
    try:
        portfolio_context = _portfolio_context(
            conn,
            ticker,
            trade_date,
            holding_quantity,
            holding_avg_entry,
            reference_price=reference_price,
        )

        try:
            ta = TradingAgentsGraph(debug=debug, config=cfg)
            final_state, decision = ta.propagate(
                ticker, trade_date, portfolio_context=portfolio_context
            )
        except Exception as e:
            return {"ok": False, "error": str(e), "ticker": ticker, "trade_date": trade_date}

        final_trade_decision = final_state.get("final_trade_decision") or ""
        upsert_cache_row(
            conn,
            ticker=ticker,
            trade_date=trade_date,
            decision=str(decision or ""),
            final_trade_decision=str(final_trade_decision),
            reference_price=reference_price,
            holding_quantity=float(holding_quantity),
            holding_avg_entry=float(holding_avg_entry),
            source=source,
        )
        if owns_conn:
            try:
                from execute_ai_trades import decide_and_execute, load_settings

                settings = load_settings(conn)
                exec_out = decide_and_execute(
                    conn,
                    ticker=ticker,
                    trade_date=trade_date,
                    dry_run=settings.get("dry_run", False),
                    settings=settings,
                )
                logger.info("Auto-execute %s: %s", ticker, exec_out)
            except Exception as exec_err:
                logger.warning("Auto-execute failed for %s: %s", ticker, exec_err)
    except Exception as e:
        try:
            conn.rollback()
        except Exception as rollback_err:
            logger.error("Rollback failed after write error: %s", rollback_err)
        return {"ok": False, "error": f"Database write failed: {e}", "ticker": ticker}
    finally:
        if owns_conn:
            conn.close()

    logger.info(
        "INSERT %s trade_date=%s decision=%r ref_price=%s",
        ticker, trade_date, decision, reference_price,
    )

    return {
        "ok": True,
        "ticker": ticker,
        "trade_date": trade_date,
        "decision": decision,
        "reference_price": reference_price,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(description="Precompute recommendation and upsert ai_recommendation_cache")
    p.add_argument("--ticker", required=True, help="Symbol e.g. TCS.NS, BTC-USD")
    p.add_argument("--trade-date", default="", help="YYYY-MM-DD (default: last trading day)")
    p.add_argument("--holding-qty", type=float, default=None)
    p.add_argument("--holding-entry", type=float, default=None)
    p.add_argument(
        "--from-portfolio",
        action="store_true",
        help="Load holding qty/entry from portfolio_holdings (admin wallet)",
    )
    p.add_argument(
        "--source",
        default="github_action_manual",
        help="Row provenance stored in ai_recommendation_cache.source",
    )
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()
    td = args.trade_date.strip()
    if not td:
        raw = ist_today()
        adjusted = adjust_to_last_trading_day(raw)
        if adjusted != raw:
            logger.info("trade_date %s is a weekend, adjusted to %s", raw, adjusted)
        td = adjusted.strftime("%Y-%m-%d")
    else:
        try:
            parsed = datetime.strptime(td, "%Y-%m-%d").date()
            adjusted = adjust_to_last_trading_day(parsed)
            if adjusted != parsed:
                logger.info("trade_date %s is a weekend, adjusted to %s", parsed, adjusted)
            td = adjusted.strftime("%Y-%m-%d")
        except ValueError:
            logger.error("Invalid trade_date format: %s. Expected YYYY-MM-DD.", td)
            sys.exit(1)

    holding_qty = 0.0 if args.holding_qty is None else args.holding_qty
    holding_entry = 0.0 if args.holding_entry is None else args.holding_entry
    db_conn = None
    if args.from_portfolio:
        db_url = resolve_psycopg2_url()
        if db_url:
            db_conn = psycopg2.connect(db_url)
            try:
                holding_qty, holding_entry = load_holding(db_conn, args.ticker)
            except Exception:
                db_conn.close()
                db_conn = None

    try:
        out = run_single_recommendation(
            ticker=args.ticker,
            trade_date=td,
            holding_quantity=holding_qty,
            holding_avg_entry=holding_entry,
            source=args.source,
            debug=args.debug,
            db_conn=db_conn,
        )
    finally:
        if db_conn is not None:
            db_conn.close()
    if not out.get("ok"):
        logger.error("%s", out)
        sys.exit(1)


if __name__ == "__main__":
    main()
