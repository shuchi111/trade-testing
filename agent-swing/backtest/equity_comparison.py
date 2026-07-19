"""
Equity comparison for the ML tab — Strategies vs Buy&Hold / Nifty.

Runs VectorBT strategies for one ticker and emits daily equity curves + ROI
summary so the UI can draw a multi-line chart with clickable strategy toggles.

CLI:
  python -m backtest.equity_comparison --ticker RELIANCE.NS --json
  python -m backtest.equity_comparison --ticker RELIANCE.NS --lite --json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback

import numpy as np
import pandas as pd

from .config import BACKTEST_END, BACKTEST_START, DEFAULT_FEES, DEFAULT_INIT_CASH, DEFAULT_SLIPPAGE
from .data_loader import load_price_data, load_volume_data
from .runner import build_strategies, extract_metrics

logger = logging.getLogger(__name__)

# Slow strategies skipped in --lite mode
_SLOW = frozenset({"ml_forecast", "prophet_forecast", "ai_agent"})


def _safe_float(v, d=0.0):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return d
        return f
    except (TypeError, ValueError):
        return d


def _downsample(series: pd.Series, max_points: int = 180) -> list[dict]:
    """Return [{date, value}] capped for chart payload size."""
    if series.empty:
        return []
    s = series.dropna()
    if len(s) > max_points:
        idx = np.linspace(0, len(s) - 1, max_points).astype(int)
        s = s.iloc[idx]
    out = []
    for ts, val in s.items():
        date = str(ts.date()) if hasattr(ts, "date") else str(ts)[:10]
        out.append({"date": date, "value": round(float(val), 2)})
    return out


def _count_buys_sells(entries: pd.Series, exits: pd.Series) -> tuple[int, int]:
    return int(entries.sum()), int(exits.sum())


def _load_nifty(start: str, end: str) -> pd.Series | None:
    """Load Nifty 50 close as a benchmark series (optional)."""
    try:
        import yfinance as yf

        raw = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            # Fallback: Nifty BeES ETF
            raw = yf.download("NIFTYBEES.NS", start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()
        if close.empty:
            return None
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.astype(float)
    except Exception as exc:
        logger.warning("Nifty load failed: %s", exc)
        return None


def _normalize_to_cash(close: pd.Series, init_cash: float, index: pd.DatetimeIndex) -> pd.Series:
    """Buy-and-hold portfolio value of `close` aligned to `index`, starting at init_cash."""
    # NSE equities often use UTC midnight timestamps; Nifty yields calendar dates.
    # Align both on calendar days so the benchmark overlays correctly.
    def _to_day(idx) -> pd.DatetimeIndex:
        ts = pd.to_datetime(idx)
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
        return ts.normalize()

    close_day = close.copy()
    close_day.index = _to_day(close_day.index)
    close_day = close_day[~close_day.index.duplicated(keep="last")]

    target = _to_day(index)
    aligned = close_day.reindex(target).ffill().bfill()
    aligned = pd.to_numeric(aligned, errors="coerce")
    if aligned.isna().all():
        return pd.Series(dtype=float)
    base = float(aligned.dropna().iloc[0])
    if not math.isfinite(base) or base <= 0:
        return pd.Series(dtype=float)
    # Return series keyed to original `index` (for chart alignment with strategies)
    values = (init_cash * (aligned.astype(float) / base)).replace([np.inf, -np.inf], np.nan)
    out = pd.Series(values.values, index=index)
    return out.ffill().bfill().dropna()


def run_equity_comparison(
    ticker: str,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    init_cash: float = DEFAULT_INIT_CASH,
    fees: float = DEFAULT_FEES,
    slippage: float = DEFAULT_SLIPPAGE,
    lite: bool = False,
    include_nifty: bool = True,
) -> dict:
    price = load_price_data(ticker, start, end)
    if price.empty:
        return {"ok": False, "error": f"No price data for {ticker}"}

    volume = load_volume_data(ticker, start, end)
    strategies = build_strategies(ticker)
    if lite:
        strategies = [s for s in strategies if s.name not in _SLOW]

    series_map: dict[str, list[dict]] = {}
    summaries: list[dict] = []
    colors_cycle = [
        "#3b82f6", "#22c55e", "#a855f7", "#f59e0b", "#ef4444",
        "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
        "#14b8a6", "#eab308",
    ]

    for i, strategy in enumerate(strategies):
        raw_name = strategy.name
        # Disambiguate duplicate names (e.g. two SMA variants)
        if raw_name == "sma_crossover":
            fast = getattr(strategy, "fast", 20)
            slow = getattr(strategy, "slow", 50)
            name = f"sma_{fast}_{slow}"
        else:
            name = raw_name
        try:
            if raw_name == "ml_forecast" and volume is not None:
                strategy._volume = volume.reindex(price.index)  # type: ignore[attr-defined]
            entries, exits = strategy.generate_signals(price)
            pf = strategy.run(price, init_cash=init_cash, fees=fees, slippage=slippage)
            equity = pf.value()
            metrics = extract_metrics(pf)
            buy_n, sell_n = _count_buys_sells(entries, exits)
            brokerage = round(init_cash * fees * (buy_n + sell_n), 2)
            drag_pct = round((brokerage / init_cash) * 100, 2) if init_cash else 0.0

            series_map[name] = _downsample(equity)
            summaries.append({
                "strategy": name,
                "label": _label(name),
                "color": colors_cycle[i % len(colors_cycle)],
                "roi_pct": round(_safe_float(metrics.get("total_return_pct")), 2),
                "sharpe": round(_safe_float(metrics.get("sharpe_ratio")), 2),
                "max_drawdown_pct": round(_safe_float(metrics.get("max_drawdown_pct")), 2),
                "brokerage": brokerage,
                "brokerage_drag_pct": drag_pct,
                "buy_count": buy_n,
                "sell_count": sell_n,
                "total_trades": int(metrics.get("total_trades") or 0),
                "final_value": round(_safe_float(metrics.get("final_value"), init_cash), 2),
                "win_rate_pct": round(_safe_float(metrics.get("win_rate_pct")), 1),
            })
        except Exception as exc:
            logger.error("Strategy %s failed: %s\n%s", name, exc, traceback.format_exc())
            summaries.append({
                "strategy": name,
                "label": _label(name),
                "color": colors_cycle[i % len(colors_cycle)],
                "error": str(exc),
                "roi_pct": None,
                "buy_count": 0,
                "sell_count": 0,
            })

    benchmarks: dict[str, list[dict]] = {}
    # Ticker buy & hold already in strategies; also emit as benchmark alias
    buy_hold_eq = None
    for s in summaries:
        if s["strategy"] == "buy_hold" and s.get("roi_pct") is not None:
            buy_hold_eq = series_map.get("buy_hold")
    if buy_hold_eq:
        benchmarks["buy_hold"] = buy_hold_eq

    if include_nifty:
        nifty = _load_nifty(start, end)
        if nifty is not None and not nifty.empty:
            nifty_port = _normalize_to_cash(nifty, init_cash, price.index)
            if not nifty_port.empty:
                benchmarks["nifty"] = _downsample(nifty_port)
                nifty_roi = _safe_float((float(nifty_port.iloc[-1]) / init_cash - 1) * 100)
                summaries.insert(0, {
                    "strategy": "nifty",
                    "label": "Nifty 50",
                    "color": "#94a3b8",
                    "roi_pct": round(nifty_roi, 2),
                    "sharpe": None,
                    "max_drawdown_pct": None,
                    "brokerage": 0,
                    "brokerage_drag_pct": 0,
                    "buy_count": 1,
                    "sell_count": 0,
                    "total_trades": 1,
                    "final_value": round(float(nifty_port.iloc[-1]), 2),
                    "is_benchmark": True,
                })
                series_map["nifty"] = benchmarks["nifty"]
            else:
                logger.warning("Nifty series empty after normalize")
        else:
            logger.warning("Nifty download empty")

    chart_rows = _build_chart(series_map)
    return {
        "ok": True,
        "ticker": ticker,
        "date_from": start,
        "date_to": end,
        "init_cash": init_cash,
        "lite": lite,
        "mode": "lite" if lite else "full",
        "summaries": summaries,
        "chart": chart_rows,
        "series_keys": list(series_map.keys()),
    }


def _build_chart(series_map: dict[str, list[dict]]) -> list[dict]:
    all_dates = sorted({p["date"] for series in series_map.values() for p in series})
    lookup = {name: {p["date"]: p["value"] for p in pts} for name, pts in series_map.items()}
    chart_rows: list[dict] = []
    for d in all_dates:
        row: dict = {"date": d}
        for name, by_date in lookup.items():
            if d in by_date:
                row[name] = by_date[d]
        chart_rows.append(row)
    for name in series_map:
        last = None
        for row in chart_rows:
            if name in row and row[name] is not None:
                last = row[name]
            elif last is not None:
                row[name] = last
    return chart_rows


def _add_nifty(
    summaries: list[dict],
    series_map: dict[str, list[dict]],
    price: pd.Series,
    init_cash: float,
    start: str,
    end: str,
) -> None:
    nifty = _load_nifty(start, end)
    if nifty is None or nifty.empty:
        return
    nifty_port = _normalize_to_cash(nifty, init_cash, price.index)
    if nifty_port.empty:
        return
    series_map["nifty"] = _downsample(nifty_port)
    nifty_roi = _safe_float((float(nifty_port.iloc[-1]) / init_cash - 1) * 100)
    summaries.insert(0, {
        "strategy": "nifty",
        "label": "Nifty 50",
        "color": "#94a3b8",
        "roi_pct": round(nifty_roi, 2),
        "sharpe": None,
        "max_drawdown_pct": None,
        "brokerage": 0,
        "brokerage_drag_pct": 0,
        "buy_count": 1,
        "sell_count": 0,
        "total_trades": 1,
        "final_value": round(float(nifty_port.iloc[-1]), 2),
        "is_benchmark": True,
    })


def run_mass_equity_comparison(
    ticker: str,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    init_cash: float = DEFAULT_INIT_CASH,
    fees: float = DEFAULT_FEES,
    slippage: float = DEFAULT_SLIPPAGE,
    include_nifty: bool = True,
    top_chart: int = 12,
    top_table: int = 80,
) -> dict:
    """Sweep 900+ parameter combos (incl. Prophet), chart top performers."""
    import os
    import time
    from datetime import datetime, timezone

    import vectorbt as vbt

    from .config import FREQ_LABEL
    from .grid_search import build_strategy
    from .param_grids import GRIDS, total_grid_count
    from .results_ranker import rank_score
    from .strategies.buy_hold import BuyHoldStrategy

    os.environ.setdefault("PROPHET_FORCE_FALLBACK", "1")  # speed for mass sweep
    t0 = time.perf_counter()
    price = load_price_data(ticker, start, end)
    if price.empty:
        return {"ok": False, "error": f"No price data for {ticker}"}

    total_experiments = total_grid_count()
    colors_cycle = [
        "#3b82f6", "#22c55e", "#a855f7", "#f59e0b", "#ef4444",
        "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
        "#14b8a6", "#eab308",
    ]

    # Fast families first; prophet last so a timeout still leaves a usable checkpoint.
    family_order = [
        "sma_crossover", "rsi", "bollinger", "keltner", "macd",
        "ensemble", "composite", "prophet_forecast",
    ]
    families = [f for f in family_order if f in GRIDS] + [f for f in GRIDS if f not in family_order]

    ranked: list[dict] = []
    tested = 0
    by_family_done: dict[str, int] = {}

    def _fast_row(pf: vbt.Portfolio, family: str, params: dict, buy_n: int, sell_n: int) -> dict:
        """Lighter metrics than full extract_metrics/stats() — critical for 900+ loops."""
        try:
            total_return_pct = float(pf.total_return() * 100)
        except Exception:
            total_return_pct = 0.0
        try:
            sharpe = float(pf.sharpe_ratio())
            if not math.isfinite(sharpe):
                sharpe = 0.0
        except Exception:
            sharpe = 0.0
        try:
            max_dd = float(abs(pf.max_drawdown()) * 100)
        except Exception:
            max_dd = 0.0
        try:
            total_trades = int(pf.trades.count())
        except Exception:
            total_trades = buy_n
        try:
            wr = float(pf.trades.win_rate() * 100) if total_trades else 0.0
            if not math.isfinite(wr):
                wr = 0.0
        except Exception:
            wr = 0.0
        try:
            final_value = float(pf.value().iloc[-1])
        except Exception:
            final_value = init_cash

        brokerage = round(init_cash * fees * (buy_n + sell_n), 2)
        metrics = {
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
            "total_trades": total_trades,
            "win_rate_pct": wr,
            "final_value": final_value,
            "profit_factor": None,
            "expectancy_pct": None,
        }
        return {
            **metrics,
            "family": family,
            "params": params,
            "buy_count": buy_n,
            "sell_count": sell_n,
            "brokerage": brokerage,
            "brokerage_drag_pct": round((brokerage / init_cash) * 100, 2) if init_cash else 0,
            "score": rank_score(metrics),
        }

    for family in families:
        combos = GRIDS[family]()
        fam_ok = 0
        for params in combos:
            tested += 1
            try:
                strategy = build_strategy(family, params)
                # One signal pass only (strategy.run would regenerate signals).
                entries, exits = strategy.generate_signals(price)
                pf = vbt.Portfolio.from_signals(
                    price,
                    entries,
                    exits,
                    init_cash=init_cash,
                    fees=fees,
                    slippage=slippage,
                    freq=FREQ_LABEL,
                )
                buy_n, sell_n = int(entries.sum()), int(exits.sum())
                ranked.append(_fast_row(pf, family, params, buy_n, sell_n))
                fam_ok += 1
            except Exception as exc:
                logger.debug("Mass combo failed %s %s: %s", family, params, exc)

            if tested % 50 == 0:
                elapsed = time.perf_counter() - t0
                logger.warning(
                    "Mass equity %d/%d (%.0fs) — last family %s",
                    tested, total_experiments, elapsed, family,
                )

        by_family_done[family] = fam_ok
        # Checkpoint after each family so a timeout still leaves usable results.
        if ranked:
            _write_mass_checkpoint(
                ticker=ticker,
                start=start,
                end=end,
                init_cash=init_cash,
                ranked=ranked,
                tested=tested,
                total_experiments=total_experiments,
                by_family_done=by_family_done,
                duration=round(time.perf_counter() - t0, 1),
                include_nifty=include_nifty,
                price=price,
                fees=fees,
                slippage=slippage,
                top_chart=top_chart,
                top_table=top_table,
                colors_cycle=colors_cycle,
            )

    ranked.sort(key=lambda x: x.get("score", 0) or 0, reverse=True)
    payload = _assemble_mass_payload(
        ticker=ticker,
        start=start,
        end=end,
        init_cash=init_cash,
        ranked=ranked,
        tested=tested,
        total_experiments=total_experiments,
        by_family_done=by_family_done,
        duration=round(time.perf_counter() - t0, 1),
        include_nifty=include_nifty,
        price=price,
        fees=fees,
        slippage=slippage,
        top_chart=top_chart,
        top_table=top_table,
        colors_cycle=colors_cycle,
        partial=False,
    )
    payload["ran_at"] = datetime.now(timezone.utc).isoformat()
    return payload


def _write_mass_checkpoint(
    *,
    ticker: str,
    start: str,
    end: str,
    init_cash: float,
    ranked: list[dict],
    tested: int,
    total_experiments: int,
    by_family_done: dict[str, int],
    duration: float,
    **_ignored,
) -> None:
    """Save slim mid-run progress so a timeout still leaves loadable results."""
    try:
        from datetime import datetime, timezone

        from .equity_cache import store_equity_comparison_local

        ordered = sorted(ranked, key=lambda x: x.get("score", 0) or 0, reverse=True)[:40]
        summaries = []
        for i, row in enumerate(ordered):
            summaries.append({
                "strategy": f"{row['family']}_{i+1}",
                "label": f"{_label(row['family'])} #{i+1}",
                "color": "#3b82f6",
                "roi_pct": round(_safe_float(row.get("total_return_pct")), 2),
                "sharpe": round(_safe_float(row.get("sharpe_ratio")), 2),
                "max_drawdown_pct": round(_safe_float(row.get("max_drawdown_pct")), 2),
                "brokerage": row.get("brokerage", 0),
                "brokerage_drag_pct": row.get("brokerage_drag_pct", 0),
                "buy_count": row.get("buy_count", 0),
                "sell_count": row.get("sell_count", 0),
                "total_trades": int(row.get("total_trades") or 0),
                "family": row["family"],
                "params": row.get("params"),
                "score": round(_safe_float(row.get("score")), 4),
            })
        payload = {
            "ok": True,
            "ticker": ticker,
            "mode": "mass",
            "date_from": start,
            "date_to": end,
            "init_cash": init_cash,
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "duration_sec": duration,
            "total_experiments": total_experiments,
            "total_strategies": len(ranked),
            "tested": tested,
            "completed_by_family": by_family_done,
            "summaries": summaries,
            "chart": [],
            "series_keys": [],
            "cached": False,
            "partial": True,
            "checkpoint": True,
        }
        store_equity_comparison_local(payload)
    except Exception as exc:
        logger.debug("Checkpoint write failed: %s", exc)


def _assemble_mass_payload(
    *,
    ticker: str,
    start: str,
    end: str,
    init_cash: float,
    ranked: list[dict],
    tested: int,
    total_experiments: int,
    by_family_done: dict[str, int],
    duration: float,
    include_nifty: bool,
    price: pd.Series,
    fees: float,
    slippage: float,
    top_chart: int,
    top_table: int,
    colors_cycle: list[str],
    partial: bool,
) -> dict:
    from .grid_search import build_strategy
    from .param_grids import GRIDS
    from .strategies.buy_hold import BuyHoldStrategy
    import vectorbt as vbt
    from .config import FREQ_LABEL

    series_map: dict[str, list[dict]] = {}
    summaries: list[dict] = []

    try:
        bh = BuyHoldStrategy()
        bh_pf = bh.run(price, init_cash=init_cash)
        bh_m = extract_metrics(bh_pf)
        series_map["buy_hold"] = _downsample(bh_pf.value())
        summaries.append({
            "strategy": "buy_hold",
            "label": "Buy & Hold",
            "color": "#64748b",
            "roi_pct": round(_safe_float(bh_m.get("total_return_pct")), 2),
            "sharpe": round(_safe_float(bh_m.get("sharpe_ratio")), 2),
            "max_drawdown_pct": round(_safe_float(bh_m.get("max_drawdown_pct")), 2),
            "brokerage": 0,
            "brokerage_drag_pct": 0,
            "buy_count": 1,
            "sell_count": 0,
            "total_trades": 1,
            "final_value": round(_safe_float(bh_m.get("final_value"), init_cash), 2),
            "is_benchmark": True,
            "family": "buy_hold",
        })
    except Exception as exc:
        logger.warning("Buy&Hold failed: %s", exc)

    ordered = sorted(ranked, key=lambda x: x.get("score", 0) or 0, reverse=True)
    table_rows = ordered[:top_table]
    chart_candidates = ordered[:top_chart]

    for i, row in enumerate(table_rows):
        family = row["family"]
        params = row["params"]
        key = f"{family}_{i+1}"
        label = f"{_label(family)} #{i+1}"
        summaries.append({
            "strategy": key,
            "label": label,
            "color": colors_cycle[i % len(colors_cycle)],
            "roi_pct": round(_safe_float(row.get("total_return_pct")), 2),
            "sharpe": round(_safe_float(row.get("sharpe_ratio")), 2),
            "max_drawdown_pct": round(_safe_float(row.get("max_drawdown_pct")), 2),
            "brokerage": row.get("brokerage", 0),
            "brokerage_drag_pct": row.get("brokerage_drag_pct", 0),
            "buy_count": row.get("buy_count", 0),
            "sell_count": row.get("sell_count", 0),
            "total_trades": int(row.get("total_trades") or 0),
            "final_value": round(_safe_float(row.get("final_value"), init_cash), 2),
            "win_rate_pct": round(_safe_float(row.get("win_rate_pct")), 1),
            "family": family,
            "params": params,
            "score": round(_safe_float(row.get("score")), 4),
            "on_chart": False,
        })

    for i, row in enumerate(chart_candidates):
        family = row["family"]
        params = row["params"]
        key = f"{family}_{i+1}"
        try:
            strategy = build_strategy(family, params)
            entries, exits = strategy.generate_signals(price)
            pf = vbt.Portfolio.from_signals(
                price, entries, exits,
                init_cash=init_cash, fees=fees, slippage=slippage, freq=FREQ_LABEL,
            )
            series_map[key] = _downsample(pf.value())
            for s in summaries:
                if s["strategy"] == key:
                    s["on_chart"] = True
                    break
        except Exception as exc:
            logger.debug("Chart equity failed %s: %s", key, exc)

    if include_nifty:
        _add_nifty(summaries, series_map, price, init_cash, start, end)

    chart_rows = _build_chart(series_map)
    return {
        "ok": True,
        "ticker": ticker,
        "mode": "mass",
        "lite": False,
        "date_from": start,
        "date_to": end,
        "init_cash": init_cash,
        "duration_sec": duration,
        "total_experiments": total_experiments,
        "total_strategies": len(ranked),
        "tested": tested,
        "tested_by_family": {k: len(v()) for k, v in GRIDS.items()},
        "completed_by_family": by_family_done,
        "top_chart": top_chart,
        "summaries": summaries,
        "chart": chart_rows,
        "series_keys": list(series_map.keys()),
        "cached": False,
        "partial": partial,
    }


def finalize_and_store(payload: dict, mode: str) -> dict:
    """Stamp metadata, persist to local + Supabase, return payload."""
    import time
    from datetime import datetime, timezone

    from .equity_cache import store_equity_comparison

    if "ran_at" not in payload:
        payload["ran_at"] = datetime.now(timezone.utc).isoformat()
    payload["mode"] = mode
    payload["cached"] = False
    store_info = store_equity_comparison(payload)
    payload["store"] = store_info
    return payload



def _label(name: str) -> str:
    labels = {
        "nifty": "Nifty 50",
        "buy_hold": "Buy & Hold",
        "ai_agent": "AI Agent",
        "sma_crossover": "SMA Crossover",
        "sma_20_50": "SMA 20/50",
        "sma_10_30": "SMA 10/30",
        "rsi": "RSI",
        "bollinger": "Bollinger",
        "macd": "MACD",
        "composite": "Composite",
        "ensemble": "Ensemble",
        "ml_forecast": "ML Forecast",
        "prophet_forecast": "Prophet",
        "keltner": "Keltner",
    }
    if name in labels:
        return labels[name]
    if name.startswith("sma_"):
        parts = name.split("_")
        if len(parts) == 3:
            return f"SMA {parts[1]}/{parts[2]}"
    return name.replace("_", " ").title()


def _json_safe(obj):
    """Replace NaN/Inf so JSON is valid."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description="Multi-strategy equity comparison")
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", default=BACKTEST_START)
    p.add_argument("--end", default=BACKTEST_END)
    p.add_argument("--mode", choices=["lite", "full", "mass"], default=None)
    p.add_argument("--lite", action="store_true", help="Skip ML / Prophet / AI (fast)")
    p.add_argument("--mass", action="store_true", help="Sweep 900+ grid combos incl. Prophet")
    p.add_argument("--load-cache", action="store_true", help="Return stored run only (no compute)")
    p.add_argument("--store", action="store_true", default=True, help="Persist result (default on)")
    p.add_argument("--no-store", action="store_true")
    p.add_argument("--no-nifty", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    mode = args.mode
    if mode is None:
        if args.mass:
            mode = "mass"
        elif args.lite:
            mode = "lite"
        else:
            mode = "full"

    store = not args.no_store

    if args.load_cache:
        from .equity_cache import load_equity_comparison

        result = load_equity_comparison(args.ticker, mode) or {
            "ok": False,
            "error": f"No cached {mode} run for {args.ticker}",
            "cached": True,
        }
    elif mode == "mass":
        result = run_mass_equity_comparison(
            args.ticker,
            start=args.start,
            end=args.end,
            include_nifty=not args.no_nifty,
        )
        if store and result.get("ok"):
            result = finalize_and_store(result, "mass")
    else:
        import time
        from datetime import datetime, timezone

        t0 = time.perf_counter()
        result = run_equity_comparison(
            args.ticker,
            start=args.start,
            end=args.end,
            lite=(mode == "lite"),
            include_nifty=not args.no_nifty,
        )
        if result.get("ok"):
            result["mode"] = mode
            result["ran_at"] = datetime.now(timezone.utc).isoformat()
            result["duration_sec"] = round(time.perf_counter() - t0, 1)
            result["total_strategies"] = len([s for s in result.get("summaries", []) if not s.get("error")])
            if store:
                result = finalize_and_store(result, mode)

    if args.json:
        print(json.dumps(_json_safe(result)))
    else:
        if not result.get("ok"):
            print(result.get("error"), file=sys.stderr)
            sys.exit(1)
        print(f"mode={result.get('mode')} ran_at={result.get('ran_at')} duration={result.get('duration_sec')}s")
        for s in result.get("summaries", [])[:20]:
            if s.get("error"):
                print(f"  {s['label']}: ERROR {s['error']}")
            else:
                roi = s.get("roi_pct")
                roi_s = f"{roi:+7.2f}%" if roi is not None else "    N/A"
                print(
                    f"  {s['label']:28s} ROI {roi_s}  "
                    f"BUY {s.get('buy_count', 0):3d}  SELL {s.get('sell_count', 0):3d}"
                )

