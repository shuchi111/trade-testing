"""
Run TradingAgents propagate for one ticker and upsert into PostgreSQL (Supabase).

Stores ``reference_price`` as the latest split-adjusted close from yfinance when
available (else raw close); scheduled runs fail if no real price is available.
Keep fetch_last_close in sync for refresh compares.

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
import re
import sys
import time
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

import psycopg2  # type: ignore[reportMissingModuleSource]
import yfinance as yf  # type: ignore[reportMissingImports]

from canonical_decision import coerce_decision_for_holdings, resolve_canonical_decision
from db_url import resolve_psycopg2_url
from execute_ai_trades import decide_and_execute, load_settings
from market_date import adjust_to_last_trading_day, ist_today
from portfolio_db import build_analysis_context, load_holding
from recommendation_bucket import recommendation_bucket
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.market_data_validator import require_fresh_market_snapshot
from tradingagents.graph.signal_processing import is_transient_propagate_error
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger("write_recommendation_cache")


_ZAI_ANTHROPIC_URL = "https://api.z.ai/api/anthropic"
DEFAULT_TRAILING_STOP_PCT = 5.0
DEFAULT_MIN_RISK_REWARD = 1.5


def _build_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    backend = (
        (os.getenv("LLM_BACKEND_URL") or "").strip()
        or (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
        or _ZAI_ANTHROPIC_URL
    ).rstrip("/")
    # Z.ai bills GLM on the Anthropic gateway (/anthropic/v1/messages), not /paas/v4.
    if provider == "glm" or "/paas/" in backend:
        provider = "anthropic"
        backend = _ZAI_ANTHROPIC_URL
    config["llm_provider"] = provider
    config["backend_url"] = backend
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


def _sleep_with_heartbeat(seconds: float, label: str) -> None:
    """Sleep in chunks so CircleCI sees log output during long Z.ai backoff waits."""
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        chunk = min(remaining, 30.0)
        time.sleep(chunk)
        remaining -= chunk
        if remaining > 0:
            logger.info("%s: %.0fs remaining", label, remaining)


def _propagate_with_retry(
    ta: TradingAgentsGraph,
    ticker: str,
    trade_date: str,
    portfolio_context: str,
):
    """Retry full graph run on Z.ai gateway overload (HTTP 529 / code 1305)."""
    max_attempts = int(os.getenv("PROPAGATE_MAX_ATTEMPTS", "5"))
    base_delay = float(os.getenv("PROPAGATE_RETRY_DELAY_SEC", "120"))
    max_delay = float(os.getenv("PROPAGATE_RETRY_MAX_DELAY_SEC", "300"))
    last_err: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return ta.propagate(ticker, trade_date, portfolio_context=portfolio_context)
        except Exception as e:
            last_err = e
            if attempt >= max_attempts - 1 or not is_transient_propagate_error(e):
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            logger.warning(
                "propagate retry %s/%s for %s in %.0fs: %s",
                attempt + 1,
                max_attempts - 1,
                ticker,
                delay,
                e,
            )
            _sleep_with_heartbeat(delay, f"propagate backoff {ticker}")

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"propagate failed for {ticker}")


def fetch_last_close(symbol: str, period: str | None = None) -> float | None:
    """Return last quoted close for ``symbol``: split-adjusted if available."""
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
        try:
            conn.rollback()
        except Exception as rollback_err:
            logger.error("Rollback failed after context error for %s: %s", ticker, rollback_err)
        raise RuntimeError(f"Real portfolio context unavailable for {ticker}: {exc}") from exc


def _first_number_match(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1).replace(",", ""))
        except (TypeError, ValueError):
            continue
    return None


def _positive(value: float | None) -> float | None:
    """Return ``value`` when strictly positive; otherwise treat as missing.

    Parameters
    ----------
    value
        Parsed target/stop level from LLM output, or ``None``.

    Returns
    -------
    float | None
        The input when ``value > 0``; otherwise ``None`` so fallback logic applies
        and zero/negative levels are never persisted.

    Raises
    ------
    None
        Non-numeric inputs are coerced via ``float()``; ``TypeError`` and
        ``ValueError`` from coercion return ``None`` instead of raising.

    Notes
    -----
    A literal ``0`` is valid to the regex but must not be stored: it silently
    defeats the 5% stop / 1.5R target fallback and inflates risk to
    ``entry - 0 = entry``.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _extract_signal_metrics(
    decision: str,
    final_trade_decision: str,
    reference_price: float | None,
) -> dict[str, float | str | None]:
    text = f"{decision or ''}\n{final_trade_decision or ''}"
    action = recommendation_bucket(decision).upper()
    entry = reference_price if reference_price and reference_price > 0 else None
    # Positive-only: a parsed 0/negative is treated as "not found" so the
    # fallbacks below apply and 0 is never stored.
    target = _positive(_first_number_match(
        text,
        [r"\b(?:target|take\s*profit|tp)\s*(?:price)?\s*[:=@-]?\s*(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)"],
    ))
    stop = _positive(_first_number_match(
        text,
        [r"\b(?:stop\s*loss|stoploss|stop|sl)\s*(?:price)?\s*[:=@-]?\s*(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)"],
    ))
    confidence = _first_number_match(
        text,
        [
            r"\b(?:confidence|probability|conviction)\s*[:=@-]?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%",
            r"\b([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(?:confidence|probability|conviction)\b",
        ],
    )

    if entry is not None and stop is None:
        stop = entry * (1 - DEFAULT_TRAILING_STOP_PCT / 100.0)
    risk = max(0.0, entry - stop) if entry is not None and stop is not None else None
    if entry is not None and target is None and action == "BUY" and risk and risk > 0:
        target = entry + risk * DEFAULT_MIN_RISK_REWARD
    reward = max(0.0, target - entry) if entry is not None and target is not None else None
    risk_reward = reward / risk if risk and risk > 0 and reward is not None else None
    if confidence is not None and not (0 <= confidence <= 100):
        confidence = None

    return {
        "signal_action": action,
        "target_price": target,
        "stop_loss_price": stop,
        "risk_amount": risk,
        "reward_amount": reward,
        "risk_reward_ratio": risk_reward,
        "ai_confidence_pct": confidence,
    }


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
    metrics = _extract_signal_metrics(decision, final_trade_decision, reference_price)
    sql = """
    INSERT INTO ai_recommendation_cache (
      ticker, trade_date, decision, final_trade_decision, reference_price,
      signal_action, target_price, stop_loss_price, risk_amount, reward_amount,
      risk_reward_ratio, ai_confidence_pct,
      holding_quantity, holding_avg_entry, source, computed_at
    )
    VALUES (%s, %s::date, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now());
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
                metrics["signal_action"],
                metrics["target_price"],
                metrics["stop_loss_price"],
                metrics["risk_amount"],
                metrics["reward_amount"],
                metrics["risk_reward_ratio"],
                metrics["ai_confidence_pct"],
                holding_quantity,
                holding_avg_entry,
                source,
            ),
        )
    if recommendation_bucket(decision) == "buy":
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
    metrics = _extract_signal_metrics(decision, final_trade_decision, reference_price)
    sql = """
    INSERT INTO ai_recommendation_history (
      ticker, trade_date, decision, final_trade_decision, bucket,
      reference_price, signal_action, target_price, stop_loss_price, risk_amount,
      reward_amount, risk_reward_ratio, ai_confidence_pct,
      holding_quantity, holding_avg_entry, source, computed_at
    )
    VALUES (%s, %s::date, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
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
                metrics["signal_action"],
                metrics["target_price"],
                metrics["stop_loss_price"],
                metrics["risk_amount"],
                metrics["reward_amount"],
                metrics["risk_reward_ratio"],
                metrics["ai_confidence_pct"],
                holding_quantity,
                holding_avg_entry,
                source,
            ),
        )




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
        return {
            "ok": False,
            "error": "Missing Z_API_KEY, GLM_API_KEY, or ANTHROPIC_AUTH_TOKEN",
            "ticker": ticker,
        }

    db_url = resolve_psycopg2_url()
    if not db_url:
        return {"ok": False, "error": "Missing DIRECT_URL or DATABASE_URL", "ticker": ticker}

    if holding_quantity < 0 or holding_avg_entry < 0:
        return {"ok": False, "error": "holding_quantity and holding_avg_entry must be >= 0", "ticker": ticker}

    try:
        snapshot = require_fresh_market_snapshot(ticker, trade_date)
        reference_price = snapshot.latest_close
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Verified fresh market data unavailable: {exc}",
            "ticker": ticker,
            "trade_date": trade_date,
        }

    owns_conn = db_conn is None
    conn = psycopg2.connect(db_url) if owns_conn else db_conn
    try:
        try:
            holding_quantity, holding_avg_entry = load_holding(conn, ticker)
        except Exception as hold_err:
            logger.warning(
                "Could not reload holdings for %s from DB; using caller values: %s",
                ticker,
                hold_err,
            )

        portfolio_context = _portfolio_context(
            conn,
            ticker,
            trade_date,
            holding_quantity,
            holding_avg_entry,
            reference_price=reference_price,
        )
        # Optional context queries may fall back after SQL errors; clear psycopg2's
        # transaction state before the LLM call and final recommendation writes.
        conn.rollback()

        try:
            ta = TradingAgentsGraph(debug=debug, config=cfg)
            final_state, decision = _propagate_with_retry(
                ta, ticker, trade_date, portfolio_context
            )
        except Exception as e:
            logger.exception("TradingAgentsGraph failed for %s trade_date=%s", ticker, trade_date)
            return {"ok": False, "error": str(e), "ticker": ticker, "trade_date": trade_date}

        final_trade_decision = final_state.get("final_trade_decision") or ""
        raw_decision = resolve_canonical_decision(
            str(decision or ""),
            str(final_trade_decision),
        )
        decision = coerce_decision_for_holdings(raw_decision, holding_quantity)
        if decision != raw_decision:
            logger.info(
                "Coerced %s decision %r -> %r (no open position, qty=%s)",
                ticker,
                raw_decision,
                decision,
                holding_quantity,
            )
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
        if owns_conn and not source.startswith("circleci"):
            try:
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
