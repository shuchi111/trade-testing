"""Shared trading constraint defaults for executor and AI context."""
from __future__ import annotations

import os

DEFAULT_MAX_POSITION_INR = 25_000.0
DEFAULT_MIN_HOLD_DAYS = 90
DEFAULT_THESIS_BREAK_LOSS_PCT = 10.0

# Paper transaction charges (INR flat per leg).
# Indian delivery reality (see plan in README / docs): STT 0.1% on BOTH buy and sell,
# plus stamp duty on buy, DP charge on sell (~₹15–80 all-in per leg on small trades).
# Default model: ₹150 flat SELL penalty only (exit costs); BUY charge optional via env.
DEFAULT_BUY_TRANSACTION_CHARGE_INR = 0.0
DEFAULT_SELL_TRANSACTION_CHARGE_INR = 150.0


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


def min_hold_days() -> int:
    raw = os.getenv("MIN_HOLD_DAYS", str(DEFAULT_MIN_HOLD_DAYS)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MIN_HOLD_DAYS


def thesis_break_loss_pct() -> float:
    raw = os.getenv("THESIS_BREAK_LOSS_PCT", str(DEFAULT_THESIS_BREAK_LOSS_PCT)).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_THESIS_BREAK_LOSS_PCT
