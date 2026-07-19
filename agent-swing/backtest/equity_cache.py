"""Persist / load Strategies-vs-Nifty comparison runs."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from .config import SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "equity_comparison"


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


def _cache_path(ticker: str, mode: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}__{mode}.json"


def store_equity_comparison_local(payload: dict) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ticker = str(payload.get("ticker", "UNKNOWN")).upper()
    mode = str(payload.get("mode", "lite"))
    path = _cache_path(ticker, mode)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return str(path)


def load_equity_comparison_local(ticker: str, mode: str) -> dict | None:
    path = _cache_path(ticker.upper(), mode)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Local equity cache read failed: %s", exc)
        return None


def store_equity_comparison_db(payload: dict) -> str | None:
    """Insert into bt_equity_comparison_runs. Returns id or None if table missing."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from .db_store import get_supabase

        sb = get_supabase()
        row = {
            "ticker": str(payload.get("ticker", "")).upper(),
            "mode": str(payload.get("mode", "lite")),
            "ran_at": payload.get("ran_at") or datetime.now(timezone.utc).isoformat(),
            "duration_sec": payload.get("duration_sec"),
            "total_strategies": payload.get("total_strategies"),
            "total_experiments": payload.get("total_experiments"),
            "date_from": payload.get("date_from"),
            "date_to": payload.get("date_to"),
            "payload": _json_safe(payload),
        }
        resp = sb.table("bt_equity_comparison_runs").insert(row).execute()
        return resp.data[0]["id"] if resp.data else None
    except Exception as exc:
        msg = str(exc).lower()
        if "bt_equity_comparison_runs" in msg or "pgrst" in msg or "does not exist" in msg:
            logger.warning(
                "bt_equity_comparison_runs missing — apply migration 014. Using local cache only."
            )
            return None
        logger.warning("DB store equity comparison failed: %s", exc)
        return None


def load_equity_comparison_db(ticker: str, mode: str) -> dict | None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from .db_store import get_supabase

        sb = get_supabase()
        resp = (
            sb.table("bt_equity_comparison_runs")
            .select("payload, ran_at, duration_sec, id")
            .eq("ticker", ticker.upper())
            .eq("mode", mode)
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
        # Ensure meta echoes latest DB columns
        payload["ok"] = True
        payload["cached"] = True
        payload["ran_at"] = row.get("ran_at") or payload.get("ran_at")
        payload["duration_sec"] = row.get("duration_sec") or payload.get("duration_sec")
        payload["cache_id"] = row.get("id")
        payload["cache_source"] = "supabase"
        return payload
    except Exception as exc:
        logger.debug("DB load equity comparison failed: %s", exc)
        return None


def load_equity_comparison(ticker: str, mode: str) -> dict | None:
    """Prefer Supabase, fall back to local JSON cache."""
    db = load_equity_comparison_db(ticker, mode)
    if db:
        return db
    local = load_equity_comparison_local(ticker, mode)
    if local:
        local["ok"] = True
        local["cached"] = True
        local["cache_source"] = "local"
        return local
    return None


def store_equity_comparison(payload: dict) -> dict:
    """Write local + DB. Returns {local_path, db_id}."""
    payload = _json_safe(payload)
    local_path = store_equity_comparison_local(payload)
    db_id = store_equity_comparison_db(payload)
    return {"local_path": local_path, "db_id": db_id}
