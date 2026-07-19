"""Shared trading constraint defaults for executor and AI context."""
from __future__ import annotations

import os

DEFAULT_MAX_POSITION_INR = 25_000.0
DEFAULT_MIN_WALLET_CASH_RESERVE_INR = 5_000.0
DEFAULT_SWING_EXIT_WINDOW_DAYS = 90
DEFAULT_THESIS_BREAK_LOSS_PCT = 10.0
DEFAULT_TRAILING_STOP_LOSS_PCT = 5.0

# Paper transaction charges (INR flat per leg).
# Indian delivery reality (see plan in README / docs): STT 0.1% on BOTH buy and sell,
# plus stamp duty on buy, DP charge on sell (~₹15–80 all-in per leg on small trades).
# Default model: ₹150 flat SELL penalty only (exit costs); BUY charge optional via env.
DEFAULT_BUY_TRANSACTION_CHARGE_INR = 0.0
DEFAULT_SELL_TRANSACTION_CHARGE_INR = 150.0

# Post-loss cool-off / portfolio quality gates (used by trade_lessons BUY guards).
DEFAULT_RECENT_LOSS_COOLDOWN_DAYS = 10
DEFAULT_MIN_LOSS_INR_FOR_COOLDOWN = 100.0
# Fractional loss of exit notional that counts as meaningful (e.g. 0.05 = 5%).
DEFAULT_MIN_LOSS_PCT_FOR_COOLDOWN = 0.05
# Exclude sells newer than this many calendar days from lesson harvest (settlement lag).
DEFAULT_LESSON_SETTLEMENT_LAG_DAYS = 1
DEFAULT_QUALITY_MIN_CLOSED_TRADES = 5
DEFAULT_QUALITY_WIN_RATE_MAX_PCT = 35.0
# Block new risk when expectancy is worse than this % of max position size.
DEFAULT_QUALITY_EXPECTANCY_PCT_OF_CAP = -0.8


def buy_transaction_charge_inr() -> float:
    raw = os.getenv(
        "BUY_TRANSACTION_CHARGE_INR", str(DEFAULT_BUY_TRANSACTION_CHARGE_INR)
    ).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_BUY_TRANSACTION_CHARGE_INR


def sell_transaction_charge_inr() -> float:
    raw = os.getenv(
        "SELL_TRANSACTION_CHARGE_INR", str(DEFAULT_SELL_TRANSACTION_CHARGE_INR)
    ).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_SELL_TRANSACTION_CHARGE_INR


def transaction_charge_for_action(action: str) -> float:
    """Flat INR charge for this trade leg (0 if disabled for that side)."""
    if action == "BUY":
        return buy_transaction_charge_inr()
    if action == "SELL":
        return sell_transaction_charge_inr()
    return 0.0


def round_trip_charge_inr() -> float:
    """BUY charge + SELL charge (full round trip)."""
    return buy_transaction_charge_inr() + sell_transaction_charge_inr()


def max_position_inr() -> float:
    raw = os.getenv("MAX_POSITION_INR", str(DEFAULT_MAX_POSITION_INR)).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_MAX_POSITION_INR


def min_wallet_cash_reserve_inr() -> float:
    raw = os.getenv(
        "MIN_WALLET_CASH_RESERVE_INR", str(DEFAULT_MIN_WALLET_CASH_RESERVE_INR)
    ).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_MIN_WALLET_CASH_RESERVE_INR


def swing_exit_window_days() -> int:
    raw = os.getenv("SWING_EXIT_WINDOW_DAYS", str(DEFAULT_SWING_EXIT_WINDOW_DAYS)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_SWING_EXIT_WINDOW_DAYS


def thesis_break_loss_pct() -> float:
    raw = os.getenv("THESIS_BREAK_LOSS_PCT", str(DEFAULT_THESIS_BREAK_LOSS_PCT)).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_THESIS_BREAK_LOSS_PCT


def trailing_stop_loss_pct() -> float:
    raw = os.getenv(
        "TRAILING_STOP_LOSS_PCT", str(DEFAULT_TRAILING_STOP_LOSS_PCT)
    ).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_TRAILING_STOP_LOSS_PCT


def recent_loss_cooldown_days() -> int:
    raw = os.getenv(
        "RECENT_LOSS_COOLDOWN_DAYS", str(DEFAULT_RECENT_LOSS_COOLDOWN_DAYS)
    ).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_RECENT_LOSS_COOLDOWN_DAYS


def min_loss_inr_for_cooldown() -> float:
    raw = os.getenv(
        "MIN_LOSS_INR_FOR_COOLDOWN", str(DEFAULT_MIN_LOSS_INR_FOR_COOLDOWN)
    ).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_MIN_LOSS_INR_FOR_COOLDOWN


def min_loss_pct_for_cooldown() -> float:
    """Minimum fractional loss of exit notional (0.05 = 5%) for cool-off."""
    raw = os.getenv(
        "MIN_LOSS_PCT_FOR_COOLDOWN", str(DEFAULT_MIN_LOSS_PCT_FOR_COOLDOWN)
    ).strip()
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return DEFAULT_MIN_LOSS_PCT_FOR_COOLDOWN


def lesson_settlement_lag_days() -> int:
    raw = os.getenv(
        "LESSON_SETTLEMENT_LAG_DAYS", str(DEFAULT_LESSON_SETTLEMENT_LAG_DAYS)
    ).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_LESSON_SETTLEMENT_LAG_DAYS


def quality_min_closed_trades() -> int:
    raw = os.getenv(
        "QUALITY_MIN_CLOSED_TRADES", str(DEFAULT_QUALITY_MIN_CLOSED_TRADES)
    ).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_QUALITY_MIN_CLOSED_TRADES


def quality_win_rate_max_pct() -> float:
    raw = os.getenv(
        "QUALITY_WIN_RATE_MAX_PCT", str(DEFAULT_QUALITY_WIN_RATE_MAX_PCT)
    ).strip()
    try:
        return min(100.0, max(0.0, float(raw)))
    except ValueError:
        return DEFAULT_QUALITY_WIN_RATE_MAX_PCT


def quality_expectancy_pct_of_cap() -> float:
    """Expectancy threshold as % of max position (negative = loss per trade)."""
    raw = os.getenv(
        "QUALITY_EXPECTANCY_PCT_OF_CAP", str(DEFAULT_QUALITY_EXPECTANCY_PCT_OF_CAP)
    ).strip()
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_QUALITY_EXPECTANCY_PCT_OF_CAP
