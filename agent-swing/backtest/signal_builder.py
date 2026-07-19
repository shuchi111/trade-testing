"""Convert AI recommendation decisions into VectorBT entry/exit boolean arrays."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from recommendation_bucket import recommendation_bucket
except ImportError:
    try:
        from agent.recommendation_bucket import recommendation_bucket  # type: ignore
    except ImportError:
        # Fallback if import path differs during backtest package runs
        def recommendation_bucket(decision: str | None) -> str:  # type: ignore
            if not decision:
                return "unknown"
            u = str(decision).strip().upper()
            if u in {"BUY", "OVERWEIGHT"} or "BUY" in u or "OVERWEIGHT" in u:
                return "buy"
            if u in {"SELL", "UNDERWEIGHT"} or "SELL" in u or "UNDERWEIGHT" in u:
                return "sell"
            if "HOLD" in u:
                return "hold"
            return "unknown"


def _normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize a DatetimeIndex to naive (no tz) for safe comparison."""
    return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx


def _normalize_timestamp(ts) -> pd.Timestamp:
    """Normalize a single timestamp to naive (no tz) for safe comparison."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        return t.tz_localize(None)
    return t


def _row_bucket(rec: pd.Series) -> str:
    """Resolve buy/sell/hold from bucket and/or decision text."""
    raw_bucket = str(rec.get("bucket") or "").strip().lower()
    if raw_bucket in {"buy", "sell", "hold"}:
        return raw_bucket
    decision = (
        rec.get("decision")
        or rec.get("final_trade_decision")
        or rec.get("action_taken")
        or ""
    )
    return recommendation_bucket(str(decision))


def normalize_reco_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure trade_date + normalized lowercase bucket column exist."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["ticker", "trade_date", "decision", "bucket", "reference_price"]
        )
    out = df.copy()
    if "trade_date" not in out.columns:
        raise ValueError("recommendation frame missing trade_date")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"])
    out["bucket"] = out.apply(_row_bucket, axis=1)
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    if "decision" not in out.columns:
        if "action_taken" in out.columns:
            out["decision"] = out["action_taken"]
        else:
            out["decision"] = out["bucket"]
    return out


def ai_recommendations_to_signals(
    recommendations: pd.DataFrame,
    price: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Convert AI recommendation / execution history into VectorBT entry/exit arrays.

    Logic:
      - bucket buy  (BUY / OVERWEIGHT)  → entries[date] = True
      - bucket sell (SELL / UNDERWEIGHT) → exits[date]   = True
      - hold / unknown → ignored

    Non-trading-day dates shift forward to the next available price bar.
    """
    price_idx = _normalize_index(price.index)
    entries = pd.Series(False, index=price.index)
    exits = pd.Series(False, index=price.index)

    recs = normalize_reco_frame(recommendations)
    if recs.empty or price.empty:
        return entries, exits

    for _, rec in recs.iterrows():
        bucket = str(rec["bucket"]).lower()
        if bucket not in {"buy", "sell"}:
            continue
        rec_date = _normalize_timestamp(rec["trade_date"])
        mask = price_idx >= rec_date
        if not mask.any():
            continue
        pos = int(mask.argmax())
        trade_date = price.index[pos]
        if bucket == "buy":
            entries[trade_date] = True
        else:
            exits[trade_date] = True

    return entries, exits


def compute_signal_accuracy(
    recommendations: pd.DataFrame,
    price: pd.Series,
    hold_days: int = 20,
) -> dict:
    """
    For each AI recommendation, check if price moved in the predicted direction
    within `hold_days` trading days (not calendar days).
    """
    if price.empty or recommendations.empty:
        return {
            "buy_precision": None,
            "sell_precision": None,
            "directional_acc": None,
            "total_buys": 0,
            "total_sells": 0,
            "correct_buys": 0,
            "correct_sells": 0,
        }

    correct_buys = 0
    total_buys = 0
    correct_sells = 0
    total_sells = 0

    price_idx = _normalize_index(price.index)
    recs = normalize_reco_frame(recommendations)

    for _, rec in recs.iterrows():
        bucket = str(rec["bucket"]).lower()
        if bucket not in ("buy", "sell"):
            continue

        rec_date = _normalize_timestamp(rec["trade_date"])
        mask = price_idx >= rec_date
        if not mask.any():
            continue
        entry_pos = int(mask.argmax())
        entry_price = price.iloc[entry_pos]

        exit_pos = min(entry_pos + hold_days, len(price) - 1)
        if exit_pos == entry_pos:
            continue
        exit_price = price.iloc[exit_pos]
        actual_return = (exit_price - entry_price) / entry_price

        if bucket == "buy":
            total_buys += 1
            if actual_return > 0:
                correct_buys += 1
        elif bucket == "sell":
            total_sells += 1
            if actual_return < 0:
                correct_sells += 1

    buy_precision = correct_buys / total_buys if total_buys > 0 else None
    sell_precision = correct_sells / total_sells if total_sells > 0 else None
    total = total_buys + total_sells
    correct_total = correct_buys + correct_sells
    directional_acc = correct_total / total if total > 0 else None

    return {
        "buy_precision": buy_precision,
        "sell_precision": sell_precision,
        "directional_acc": directional_acc,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "correct_buys": correct_buys,
        "correct_sells": correct_sells,
    }


def _pearson(x: pd.Series, y: pd.Series) -> float | None:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n < 3:
        return None
    sx = df["x"]
    sy = df["y"]
    if sx.std(ddof=0) == 0 or sy.std(ddof=0) == 0:
        return None
    return float(sx.corr(sy))


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return None
    return _pearson(df["x"].rank(), df["y"].rank())


def compute_ic(
    recommendations: pd.DataFrame,
    price: pd.Series,
    hold_days: int = 20,
) -> dict:
    empty = {"ic": None, "rank_ic": None, "ic_sample": 0}
    if price.empty or recommendations.empty:
        return empty

    price_idx = _normalize_index(price.index)
    strengths: list[float] = []
    forward_returns: list[float] = []
    recs = normalize_reco_frame(recommendations)

    for _, rec in recs.iterrows():
        bucket = str(rec["bucket"]).lower()
        if bucket not in ("buy", "sell"):
            continue
        rec_date = _normalize_timestamp(rec["trade_date"])

        mask = price_idx >= rec_date
        if not mask.any():
            continue
        entry_pos = int(mask.argmax())
        entry_price = price.iloc[entry_pos]

        exit_pos = min(entry_pos + hold_days, len(price) - 1)
        if exit_pos == entry_pos:
            continue
        exit_price = price.iloc[exit_pos]
        if entry_price == 0:
            continue

        strengths.append(1.0 if bucket == "buy" else -1.0)
        forward_returns.append(float((exit_price - entry_price) / entry_price))

    if len(strengths) < 3:
        return empty

    s = pd.Series(strengths, dtype=float)
    r = pd.Series(forward_returns, dtype=float)
    return {
        "ic": _pearson(s, r),
        "rank_ic": _spearman(s, r),
        "ic_sample": len(strengths),
    }


def compute_ic_from_predictions(
    predictions: pd.Series,
    price: pd.Series,
    hold_days: int = 5,
) -> dict:
    empty = {"ic": None, "rank_ic": None, "ic_sample": 0}
    if predictions.empty or price.empty:
        return empty

    strengths: list[float] = []
    forward_returns: list[float] = []

    for dt, pred in predictions.items():
        if pred is None or (isinstance(pred, float) and not np.isfinite(pred)):
            continue
        if pd.isna(pred):
            continue
        try:
            pos = int(price.index.get_loc(dt))
        except KeyError:
            continue
        if isinstance(pos, slice):
            continue
        exit_pos = min(pos + hold_days, len(price) - 1)
        if exit_pos == pos:
            continue
        entry_price = price.iloc[pos]
        exit_price = price.iloc[exit_pos]
        if entry_price == 0:
            continue
        strengths.append(float(pred))
        forward_returns.append(float((exit_price - entry_price) / entry_price))

    if len(strengths) < 3:
        return empty

    s = pd.Series(strengths, dtype=float)
    r = pd.Series(forward_returns, dtype=float)
    return {
        "ic": _pearson(s, r),
        "rank_ic": _spearman(s, r),
        "ic_sample": len(strengths),
    }
