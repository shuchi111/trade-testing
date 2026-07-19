"""Persist / load All-Ticker ML signal scans."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from .config import SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "universe_signals"
CACHE_FILE = CACHE_DIR / "latest.json"


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def store_universe_signals_local(payload: dict) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_FILE
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return str(path)


def load_universe_signals_local() -> dict | None:
    if not CACHE_FILE.exists():
        return None
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Local universe cache read failed: %s", exc)
        return None


def store_universe_signals_db(payload: dict) -> str | None:
    """Insert into ml_universe_signal_runs. Returns id or None if table missing."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from .db_store import get_supabase

        sb = get_supabase()
        row = {
            "ran_at": payload.get("ran_at") or datetime.now(timezone.utc).isoformat(),
            "duration_sec": payload.get("duration_sec"),
            "as_of": payload.get("as_of"),
            "with_backtest": bool(payload.get("with_backtest")),
            "summary": payload.get("summary") or {},
            "backtest_summary": payload.get("backtest_summary"),
            "payload": _json_safe(payload),
        }
        resp = sb.table("ml_universe_signal_runs").insert(row).execute()
        return resp.data[0]["id"] if resp.data else None
    except Exception as exc:
        msg = str(exc).lower()
        if "ml_universe_signal_runs" in msg or "pgrst" in msg or "does not exist" in msg:
            logger.warning(
                "ml_universe_signal_runs missing — apply migration 015. Using local cache only."
            )
            return None
        logger.warning("DB store universe signals failed: %s", exc)
        return None


def load_universe_signals_db() -> dict | None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from .db_store import get_supabase

        sb = get_supabase()
        resp = (
            sb.table("ml_universe_signal_runs")
            .select("payload, ran_at, duration_sec, id, with_backtest, summary, backtest_summary, as_of")
            .order("ran_at", desc=True)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return None
        row = resp.data[0]
        payload = row.get("payload") or {}
        if isinstance(payload, str):
            payload = json.loads(payload)
        payload["ok"] = True
        payload["cached"] = True
        payload["ran_at"] = row.get("ran_at") or payload.get("ran_at")
        payload["duration_sec"] = row.get("duration_sec") or payload.get("duration_sec")
        payload["with_backtest"] = row.get("with_backtest") if row.get("with_backtest") is not None else payload.get("with_backtest")
        payload["summary"] = row.get("summary") or payload.get("summary")
        payload["backtest_summary"] = row.get("backtest_summary") or payload.get("backtest_summary")
        payload["as_of"] = row.get("as_of") or payload.get("as_of")
        payload["cache_id"] = row.get("id")
        payload["cache_source"] = "supabase"
        return payload
    except Exception as exc:
        logger.debug("DB load universe signals failed: %s", exc)
        return None


def load_universe_signals() -> dict | None:
    db = load_universe_signals_db()
    if db:
        return db
    local = load_universe_signals_local()
    if local:
        local["ok"] = True
        local["cached"] = True
        local["cache_source"] = "local"
        return local
    return None


def store_universe_signals(payload: dict) -> dict:
    payload = _json_safe(payload)
    local_path = store_universe_signals_local(payload)
    db_id = store_universe_signals_db(payload)
    return {"local_path": local_path, "db_id": db_id}
