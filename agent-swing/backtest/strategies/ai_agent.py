"""Strategy: use stored AI recommendations / paper trades as entry/exit signals."""
from __future__ import annotations

import pandas as pd

from .base import Strategy
from ..data_loader import (
    load_ai_cache,
    load_ai_recommendations,
    load_ai_trade_executions,
)
from ..signal_builder import (
    _normalize_index,
    _normalize_timestamp,
    ai_recommendations_to_signals,
    normalize_reco_frame,
)


def _ticker_mask(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df
    want = ticker.strip().upper()
    return df[df["ticker"].astype(str).str.strip().str.upper() == want].copy()


def _merge_reco_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = [normalize_reco_frame(f) for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # Prefer actionable rows; drop exact date+bucket dupes (keep last).
    out = out.sort_values("trade_date")
    out = out.drop_duplicates(subset=["trade_date", "bucket"], keep="last")
    return out.reset_index(drop=True)


class AiAgentStrategy(Strategy):
    """
    Backtest your AI system from **database history**, not a fresh LLM run.

    Signal source priority:
      1. ``ai_trade_executions`` — actual paper BUY/SELL fills (best ground truth)
      2. Else ``ai_recommendation_history`` + ``ai_recommendation_cache``
         (BUY/OVERWEIGHT → entry, SELL/UNDERWEIGHT → exit)

    If both recommendation tables only contain HOLD/unknown, trades stay at 0 —
    re-run cron or pick a ticker that has stored BUY/SELL rows.
    """

    def __init__(self, ticker: str):
        self._ticker = ticker.strip().upper()
        self._source = "none"

        executions = _ticker_mask(load_ai_trade_executions(self._ticker), self._ticker)
        if not executions.empty and executions["bucket"].isin(["buy", "sell"]).any():
            self._recommendations = normalize_reco_frame(executions)
            self._source = "ai_trade_executions"
        else:
            history = _ticker_mask(load_ai_recommendations(self._ticker), self._ticker)
            cache = _ticker_mask(load_ai_cache(self._ticker), self._ticker)
            merged = _merge_reco_frames(history, cache)
            self._recommendations = merged
            if not history.empty and not cache.empty:
                self._source = "ai_recommendation_history+cache"
            elif not history.empty:
                self._source = "ai_recommendation_history"
            elif not cache.empty:
                self._source = "ai_recommendation_cache"
            else:
                self._source = "none"

    @property
    def name(self) -> str:
        return "ai_agent"

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        return ai_recommendations_to_signals(self._recommendations, price)

    def build_trade_reasons(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        recs = self._recommendations
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)
        entry_values = pd.Series(None, index=price.index, dtype=float)
        exit_values = pd.Series(None, index=price.index, dtype=float)

        if recs.empty:
            return entry_reasons, exit_reasons, entry_values, exit_values

        entries, exits = self.generate_signals(price)
        price_idx = _normalize_index(price.index)

        for _, rec in recs.iterrows():
            bucket = str(rec.get("bucket", "")).lower()
            rec_date = _normalize_timestamp(rec["trade_date"])
            mask = price_idx >= rec_date
            if not mask.any():
                continue
            pos = int(mask.argmax())
            trade_date = price.index[pos]
            dec = rec.get("decision", bucket)
            ref = rec.get("reference_price", "")
            src = rec.get("signal_source", self._source)

            if bucket == "buy" and bool(entries.get(trade_date, False)):
                entry_reasons[trade_date] = f"AI {src}: {dec} (price={ref})"
                try:
                    entry_values[trade_date] = float(ref) if ref not in ("", None) else float(price[trade_date])
                except (TypeError, ValueError):
                    entry_values[trade_date] = float(price[trade_date])

            elif bucket == "sell" and bool(exits.get(trade_date, False)):
                exit_reasons[trade_date] = f"AI {src}: {dec} (price={ref})"
                try:
                    exit_values[trade_date] = float(ref) if ref not in ("", None) else float(price[trade_date])
                except (TypeError, ValueError):
                    exit_values[trade_date] = float(price[trade_date])

        return entry_reasons, exit_reasons, entry_values, exit_values

    @property
    def config(self) -> dict:
        n = 0 if self._recommendations.empty else len(self._recommendations)
        buys = 0 if self._recommendations.empty else int((self._recommendations["bucket"] == "buy").sum())
        sells = 0 if self._recommendations.empty else int((self._recommendations["bucket"] == "sell").sum())
        return {
            "source": self._source,
            "ticker": self._ticker,
            "rows": n,
            "buy_signals": buys,
            "sell_signals": sells,
        }
