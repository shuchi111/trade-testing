"""Rank grid-search results by composite score."""

from __future__ import annotations


def _norm(val: float | None, lo: float, hi: float) -> float:
    if val is None:
        return 0.0
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def rank_score(metrics: dict) -> float:
    sharpe = metrics.get("sharpe_ratio") or 0
    expectancy = metrics.get("expectancy_pct") or 0
    max_dd = metrics.get("max_drawdown_pct") or 100
    win_rate = metrics.get("win_rate_pct") or 0
    pf = metrics.get("profit_factor") or 0

    return (
        0.35 * _norm(sharpe, -1, 3)
        + 0.25 * _norm(expectancy, -5, 5)
        + 0.20 * (1 - _norm(max_dd, 0, 50))
        + 0.10 * _norm(win_rate, 0, 100)
        + 0.10 * _norm(pf, 0, 3)
    )


def passes_filters(metrics: dict) -> bool:
    """Survival filters aligned with ML tab validation funnel (lib/ml/validation-funnel.ts)."""
    trades = metrics.get("total_trades") or 0
    max_dd = metrics.get("max_drawdown_pct") or 0
    expectancy = metrics.get("expectancy_pct") or 0
    sharpe = metrics.get("sharpe_ratio") or 0
    pf = metrics.get("profit_factor") or 0
    win_rate = metrics.get("win_rate_pct")

    if not (
        trades >= 10
        and max_dd <= 35
        and expectancy > 0
        and sharpe >= 0.5
        and pf >= 1.0
    ):
        return False

    # Overfit proxy: very high win rate without supporting Sharpe
    if win_rate is not None and win_rate > 75 and sharpe < 0.35:
        return False

    # IC quality when present (skip if column/null — same as TS funnel)
    ic = metrics.get("ic")
    if ic is not None:
        try:
            if abs(float(ic)) < 0.02:
                return False
        except (TypeError, ValueError):
            pass

    return True
