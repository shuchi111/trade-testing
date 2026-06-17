"""Shared trading constraint defaults for executor and AI context."""
from __future__ import annotations

import os

DEFAULT_MAX_POSITION_INR = 25_000.0
DEFAULT_MIN_HOLD_DAYS = 90
DEFAULT_THESIS_BREAK_LOSS_PCT = 10.0


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
