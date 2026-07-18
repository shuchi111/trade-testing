"""Persist ML desk signals + run reports (psycopg2 + optional Supabase)."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ML_SETTINGS_ID = "00000000-0000-0000-0000-000000000004"

# Ensure agent root on path when imported as backtest.ml_desk_store
_AGENT = Path(__file__).resolve().parents[1]
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))


def _pg_conn():
    from db_url import resolve_psycopg2_url
    import psycopg2  # type: ignore[reportMissingModuleSource]

    url = resolve_psycopg2_url()
    if not url:
        raise RuntimeError("DIRECT_URL / DATABASE_URL required")
    return psycopg2.connect(url)


def load_min_pred_return_pct() -> float:
    try:
        conn = _pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT min_pred_return_pct FROM ml_trading_settings WHERE id = %s",
                    (ML_SETTINGS_ID,),
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    return float(row[0])
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("ml_trading_settings read failed: %s", exc)
    return 0.5


def upsert_signal_rows(
    signals: list[dict[str, Any]],
    *,
    min_pred_return_pct: float,
    source: str = "cron",
) -> dict[str, int]:
    ok_rows = [r for r in signals if r.get("ok")]
    buys = sorted(
        [r for r in ok_rows if str(r.get("side", "")).upper() == "BUY"],
        key=lambda r: float(r.get("predicted_return_pct") or -1e9),
        reverse=True,
    )
    rank_map = {str(r.get("ticker", "")).upper(): i + 1 for i, r in enumerate(buys)}
    n = max(len(buys), 1)
    now = datetime.now(timezone.utc)

    rows: list[dict[str, Any]] = []
    for r in ok_rows:
        ticker = str(r.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        side = str(r.get("side") or "HOLD").upper()
        signal = {"BUY": "buy", "SELL": "sell"}.get(side, "hold")
        pred = r.get("predicted_return_pct")
        pred_f = float(pred) if pred is not None else None
        passes = False
        if signal == "buy" and pred_f is not None:
            passes = pred_f >= min_pred_return_pct
        elif signal == "sell":
            passes = True
        rank = rank_map.get(ticker)
        percentile = round((n - rank + 1) / n * 100, 2) if rank else None
        raw = {
            "confidence": r.get("confidence"),
            "reference_price": r.get("reference_price"),
            "buy_threshold_pct": r.get("buy_threshold_pct"),
            "sell_threshold_pct": r.get("sell_threshold_pct"),
            "retrain_date": r.get("retrain_date"),
        }
        rows.append(
            {
                "ticker": ticker,
                "as_of": r.get("as_of"),
                "signal": signal,
                "pred_return_pct": pred_f,
                "rank": rank,
                "percentile": percentile,
                "horizon_days": int(r.get("horizon_days") or 5),
                "model_name": "lightgbm_alpha158",
                "feature_count": r.get("feature_count"),
                "train_rows": r.get("train_rows"),
                "passes_min_pred": passes,
                "min_pred_return_pct": min_pred_return_pct,
                "raw": raw,
                "source": source,
                "computed_at": now,
            }
        )

    if not rows:
        return {"cache": 0, "history": 0}

    conn = _pg_conn()
    written_cache = written_hist = 0
    try:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO ml_signal_cache (
                      ticker, as_of, signal, pred_return_pct, rank, percentile,
                      horizon_days, model_name, feature_count, train_rows,
                      passes_min_pred, min_pred_return_pct, raw, source, computed_at
                    ) VALUES (
                      %(ticker)s, %(as_of)s::date, %(signal)s, %(pred_return_pct)s,
                      %(rank)s, %(percentile)s, %(horizon_days)s, %(model_name)s,
                      %(feature_count)s, %(train_rows)s, %(passes_min_pred)s,
                      %(min_pred_return_pct)s, %(raw)s::jsonb, %(source)s, %(computed_at)s
                    )
                    ON CONFLICT (ticker) DO UPDATE SET
                      as_of = EXCLUDED.as_of,
                      signal = EXCLUDED.signal,
                      pred_return_pct = EXCLUDED.pred_return_pct,
                      rank = EXCLUDED.rank,
                      percentile = EXCLUDED.percentile,
                      horizon_days = EXCLUDED.horizon_days,
                      model_name = EXCLUDED.model_name,
                      feature_count = EXCLUDED.feature_count,
                      train_rows = EXCLUDED.train_rows,
                      passes_min_pred = EXCLUDED.passes_min_pred,
                      min_pred_return_pct = EXCLUDED.min_pred_return_pct,
                      raw = EXCLUDED.raw,
                      source = EXCLUDED.source,
                      computed_at = EXCLUDED.computed_at
                    """,
                    {**row, "raw": json.dumps(row["raw"])},
                )
                written_cache += 1
                cur.execute(
                    """
                    INSERT INTO ml_signal_history (
                      ticker, as_of, signal, pred_return_pct, rank, percentile,
                      horizon_days, model_name, feature_count, train_rows,
                      passes_min_pred, min_pred_return_pct, raw, source, computed_at
                    ) VALUES (
                      %(ticker)s, %(as_of)s::date, %(signal)s, %(pred_return_pct)s,
                      %(rank)s, %(percentile)s, %(horizon_days)s, %(model_name)s,
                      %(feature_count)s, %(train_rows)s, %(passes_min_pred)s,
                      %(min_pred_return_pct)s, %(raw)s::jsonb, %(source)s, %(computed_at)s
                    )
                    ON CONFLICT (ticker, as_of, signal, model_name) DO UPDATE SET
                      pred_return_pct = EXCLUDED.pred_return_pct,
                      rank = EXCLUDED.rank,
                      percentile = EXCLUDED.percentile,
                      passes_min_pred = EXCLUDED.passes_min_pred,
                      min_pred_return_pct = EXCLUDED.min_pred_return_pct,
                      raw = EXCLUDED.raw,
                      source = EXCLUDED.source,
                      computed_at = EXCLUDED.computed_at
                    """,
                    {**row, "raw": json.dumps(row["raw"])},
                )
                written_hist += 1
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("signal upsert failed: %s", exc)
        raise
    finally:
        conn.close()

    return {"cache": written_cache, "history": written_hist}


def insert_run_report(payload: dict[str, Any]) -> str | None:
    summary = payload.get("summary") or {}
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ml_run_reports (
                  ran_at, duration_sec, mode, as_of, tickers_scored,
                  buys, sells, holds, executed_buys, executed_sells, skipped,
                  wallet_cash_before, wallet_cash_after, summary, error, source
                ) VALUES (
                  %s, %s, %s, %s::date, %s,
                  %s, %s, %s, %s, %s, %s,
                  %s, %s, %s::jsonb, %s, %s
                )
                RETURNING id
                """,
                (
                    payload.get("ran_at") or datetime.now(timezone.utc).isoformat(),
                    payload.get("duration_sec"),
                    payload.get("mode") or "universe",
                    payload.get("as_of"),
                    int(summary.get("total") or summary.get("ok") or 0),
                    int(summary.get("buy") or 0),
                    int(summary.get("sell") or 0),
                    int(summary.get("hold") or 0),
                    int(payload.get("executed_buys") or 0),
                    int(payload.get("executed_sells") or 0),
                    int(payload.get("skipped") or 0),
                    payload.get("wallet_cash_before"),
                    payload.get("wallet_cash_after"),
                    json.dumps(
                        {
                            "store": payload.get("store_counts"),
                            "failed": summary.get("failed"),
                            "min_pred_return_pct": payload.get("min_pred_return_pct"),
                        }
                    ),
                    payload.get("error"),
                    payload.get("source") or "cron",
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        conn.rollback()
        logger.warning("ml_run_reports insert failed: %s", exc)
        return None
    finally:
        conn.close()
