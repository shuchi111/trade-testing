#!/usr/bin/env python3
"""Harvest closed-trade P&L into durable ``ai_trade_lessons`` for next AI runs.

Run after ``execute_ai_trades.py`` so today's exits become tomorrow's scar tissue.

Usage:
  python agent/write_reflection_memory.py
  python agent/write_reflection_memory.py --lookback-days 60
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
from trade_lessons import (
    ensure_lessons_table,
    harvest_lessons_from_closed_trades,
    load_lessons,
)

logger = logging.getLogger("write_reflection_memory")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(description="Persist lessons from closed paper trades")
    p.add_argument("--lookback-days", type=int, default=45)
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()

    db_url = resolve_psycopg2_url()
    if not db_url:
        print(json.dumps({"ok": False, "error": "Missing DIRECT_URL or DATABASE_URL"}))
        return 1

    conn = psycopg2.connect(db_url)
    try:
        ensure_lessons_table(conn)
        result = harvest_lessons_from_closed_trades(
            conn, lookback_days=args.lookback_days, limit=args.limit
        )
        recent = load_lessons(conn, limit=5, losses_only=True)
        result["sample_loss_lessons"] = [
            {
                "ticker": L["ticker"],
                "trade_date": str(L["trade_date"]),
                "pnl": L["realized_pnl"],
                "lesson": (L["lesson"] or "")[:180],
            }
            for L in recent
        ]
        print(json.dumps(result, default=str))
        logger.info(
            "Reflection harvest: closed=%s written=%s skipped=%s",
            result.get("closed_sells"),
            result.get("written"),
            result.get("skipped"),
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
