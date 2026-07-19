"""Score universe with LightGBM and persist to ml_signal_* + ml_run_reports.

Usage:
  python -m backtest.ml_score --universe --json
  python -m backtest.ml_score --ticker RELIANCE.NS --json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def run_ml_score(
    *,
    tickers: list[str] | None = None,
    universe: bool = False,
    start: str = "2022-01-01",
    source: str = "manual",
) -> dict:
    # Load .env via backtest.config before DB writes
    from . import config as _cfg  # noqa: F401
    from .ml_desk_store import insert_run_report, load_min_pred_return_pct, upsert_signal_rows
    from .ml_predict import predict_latest, predict_universe

    t0 = time.perf_counter()
    min_pred = load_min_pred_return_pct()

    if universe or (tickers and len(tickers) > 1):
        payload = predict_universe(
            tickers=tickers,
            start=start,
            stocks_only=True,
            with_backtest=False,
            store=False,
        )
        signals = payload.get("signals") or []
        summary = payload.get("summary") or {}
        as_of = payload.get("as_of")
        mode = "universe"
    else:
        sym = (tickers or ["RELIANCE.NS"])[0]
        one = predict_latest(sym, start=start)
        signals = [one]
        ok = bool(one.get("ok"))
        side = str(one.get("side") or "HOLD").upper() if ok else "HOLD"
        summary = {
            "total": 1,
            "ok": 1 if ok else 0,
            "failed": 0 if ok else 1,
            "buy": 1 if side == "BUY" else 0,
            "sell": 1 if side == "SELL" else 0,
            "hold": 1 if side == "HOLD" and ok else 0,
        }
        as_of = one.get("as_of") if ok else None
        mode = "single"

    store_counts = upsert_signal_rows(
        signals,
        min_pred_return_pct=min_pred,
        source=source,
    )
    duration = round(time.perf_counter() - t0, 1)
    report = {
        "ok": True,
        "mode": mode,
        "as_of": as_of,
        "summary": summary,
        "duration_sec": duration,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "min_pred_return_pct": min_pred,
        "store_counts": store_counts,
        "signals": signals,
    }
    report_id = insert_run_report(report)
    report["report_id"] = report_id
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="ML desk score → ml_signal_cache/history")
    p.add_argument("--ticker", help="Single ticker")
    p.add_argument("--universe", action="store_true", help="All NSE tickers in config")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--source", default="manual", help="manual | cron | ui")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if not args.ticker and not args.universe:
        p.error("Provide --ticker or --universe")

    tickers = [args.ticker.strip().upper()] if args.ticker else None
    result = run_ml_score(
        tickers=tickers,
        universe=args.universe,
        start=args.start,
        source=args.source,
    )
    if args.json:
        # Trim huge signal dump for cron logs unless single ticker
        out = dict(result)
        if args.universe and "signals" in out:
            out["signals"] = [
                {
                    "ticker": s.get("ticker"),
                    "ok": s.get("ok"),
                    "side": s.get("side"),
                    "predicted_return_pct": s.get("predicted_return_pct"),
                    "as_of": s.get("as_of"),
                }
                for s in (result.get("signals") or [])
            ]
        print(json.dumps(out))
    else:
        s = result.get("summary") or {}
        print(
            f"ML score done mode={result.get('mode')} "
            f"ok={s.get('ok')}/{s.get('total')} "
            f"buy={s.get('buy')} sell={s.get('sell')} hold={s.get('hold')} "
            f"stored={result.get('store_counts')} report={result.get('report_id')}"
        )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
