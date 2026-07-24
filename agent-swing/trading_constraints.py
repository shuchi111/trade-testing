"""Env-tunable swing trading constraints (shared by executor + prompts)."""
from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def max_position_inr() -> float:
    return _env_float("MAX_POSITION_INR", 25_000.0)


def min_wallet_cash_reserve_inr() -> float:
    return _env_float("MIN_WALLET_CASH_RESERVE_INR", 5_000.0)


def swing_exit_window_days() -> int:
    return _env_int("MIN_HOLD_DAYS", _env_int("SWING_EXIT_WINDOW_DAYS", 90))


def thesis_break_loss_pct() -> float:
    return _env_float("THESIS_BREAK_LOSS_PCT", 10.0)


def trailing_stop_loss_pct() -> float:
    return _env_float("TRAILING_STOP_LOSS_PCT", 5.0)


def sell_transaction_charge_inr() -> float:
    return _env_float("SELL_TRANSACTION_CHARGE_INR", 150.0)


def buy_transaction_charge_inr() -> float:
    return _env_float("BUY_TRANSACTION_CHARGE_INR", 0.0)


def min_risk_reward() -> float:
    return _env_float("MIN_RISK_REWARD", 1.5)


def min_profit_sell_pct() -> float:
    return _env_float("MIN_PROFIT_SELL_PCT", 3.0)


def max_open_positions() -> int:
    return _env_int("MAX_OPEN_POSITIONS", 5)


def min_ai_confidence_pct() -> float:
    """BUY skipped when confidence is present and below this (0 = disabled)."""
    return min(100.0, max(0.0, _env_float("MIN_AI_CONFIDENCE_PCT", 80.0)))


def confidence_missing_scale() -> float:
    """Missing confidence → SKIP by default (no guessing)."""
    return min(1.0, max(0.0, _env_float("CONFIDENCE_MISSING_SCALE", 0.0)))


def confidence_at_min_scale() -> float:
    """Room fraction at the min bar (80% → 0.50 → ₹12,500 of ₹25k)."""
    return min(1.0, max(0.0, _env_float("CONFIDENCE_AT_MIN_SCALE", 0.50)))


def confidence_buy_scale(confidence_pct: float | None) -> float:
    """Map AI confidence (0–100) → fraction of room under the ₹25k cap.

    Remap [min_bar .. 100] → [at_min_scale .. 1.0]:
    80% → ₹12,500, 100% → ₹25,000. Missing / below bar → SKIP.
    No equity / stop-distance risk sizing — confidence only.
    """
    if confidence_pct is None:
        return confidence_missing_scale()
    try:
        conf = float(confidence_pct)
    except (TypeError, ValueError):
        return confidence_missing_scale()
    if not (conf == conf):  # NaN
        return confidence_missing_scale()
    conf = min(100.0, max(0.0, conf))
    floor = min_ai_confidence_pct()
    if floor > 0 and conf + 1e-9 < floor:
        return 0.0
    at_min = confidence_at_min_scale()
    if conf + 1e-9 >= 100.0 or floor >= 100.0:
        return 1.0
    t = (conf - floor) / (100.0 - floor)
    return min(1.0, max(0.0, at_min + t * (1.0 - at_min)))


def sized_buy_budget_inr(
    *,
    cash_available: float,
    room_to_cap: float,
    confidence_pct: float | None,
) -> float:
    """Confidence-scaled buy budget; cash and room_to_cap remain hard ceilings."""
    scale = confidence_buy_scale(confidence_pct)
    if scale <= 0:
        return 0.0
    ceiling = min(max(0.0, cash_available), max(0.0, room_to_cap))
    return ceiling * scale


def can_sell_under_min_hold(
    *,
    days_held: int | None,
    price: float,
    avg_entry: float,
    min_hold_days: int | None = None,
    thesis_break_pct: float | None = None,
) -> tuple[bool, str | None]:
    hold_days = swing_exit_window_days() if min_hold_days is None else min_hold_days
    break_pct = thesis_break_loss_pct() if thesis_break_pct is None else thesis_break_pct

    if days_held is None:
        return True, None
    if days_held >= hold_days:
        return True, None
    if avg_entry <= 0 or price <= 0:
        return True, None

    loss_pct = (price - avg_entry) / avg_entry * 100.0
    if loss_pct <= -abs(break_pct):
        return True, None
    return False, "min_hold_period"


def can_sell_for_profit(
    *,
    price: float,
    avg_entry: float,
    min_pct: float | None = None,
) -> tuple[bool, str | None]:
    floor = min_profit_sell_pct() if min_pct is None else min_pct
    if avg_entry <= 0 or price <= 0:
        return True, None
    gain_pct = (price - avg_entry) / avg_entry * 100.0
    if gain_pct <= 0:
        return True, None
    if gain_pct + 1e-9 < floor:
        return False, "min_profit_sell"
    return True, None
