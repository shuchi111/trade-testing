"""Strategy: Facebook Prophet forward-return forecast.

Fits a Prophet model on expanding (or periodic) history and goes long when the
predicted N-day return exceeds a buy threshold. Falls back to a lightweight
seasonal-trend model if `prophet` is not installed.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .base import Strategy

logger = logging.getLogger(__name__)


def _try_import_prophet():
    import os

    if os.environ.get("PROPHET_FORCE_FALLBACK", "").strip() in ("1", "true", "yes"):
        return None
    try:
        from prophet import Prophet  # type: ignore

        return Prophet
    except Exception:
        return None


def _prophet_predict_return(history: pd.Series, horizon: int) -> float | None:
    """Fit Prophet on `history` and return predicted % return over `horizon` days."""
    Prophet = _try_import_prophet()
    if Prophet is None or len(history) < 60:
        return None

    df = pd.DataFrame({
        "ds": pd.to_datetime(history.index).tz_localize(None),
        "y": history.values.astype(float),
    }).dropna()
    if len(df) < 60:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
            )
            model.fit(df)
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)
        last_close = float(df["y"].iloc[-1])
        if last_close <= 0:
            return None
        pred_close = float(forecast["yhat"].iloc[-1])
        return (pred_close / last_close) - 1.0
    except Exception as exc:
        logger.debug("Prophet fit failed: %s", exc)
        return None


def _fallback_predict_return(history: pd.Series, horizon: int) -> float | None:
    """Lightweight trend + weekly seasonality when Prophet is unavailable."""
    if len(history) < 60:
        return None
    y = np.log(history.astype(float).clip(lower=1e-9).values)
    n = len(y)
    t = np.arange(n, dtype=float)
    # Weekly dummies from weekday of index
    weekdays = pd.to_datetime(history.index).dayofweek.values
    X = np.column_stack([
        np.ones(n),
        t,
        np.sin(2 * np.pi * weekdays / 7),
        np.cos(2 * np.pi * weekdays / 7),
    ])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return None
    t_f = n - 1 + horizon
    wd_f = int((weekdays[-1] + horizon) % 7)
    y_hat = (
        coef[0]
        + coef[1] * t_f
        + coef[2] * np.sin(2 * np.pi * wd_f / 7)
        + coef[3] * np.cos(2 * np.pi * wd_f / 7)
    )
    last = float(history.iloc[-1])
    if last <= 0:
        return None
    return float(np.exp(y_hat) / last - 1.0)


class ProphetForecastStrategy(Strategy):
    """BUY when Prophet-style N-day forecast return > buy_threshold."""

    def __init__(
        self,
        horizon: int = 5,
        buy_threshold: float = 0.005,
        sell_threshold: float = 0.005,
        retrain_step: int = 21,
        min_train: int = 90,
    ):
        self.horizon = horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.retrain_step = max(1, retrain_step)
        self.min_train = min_train
        self._backend = "prophet" if _try_import_prophet() is not None else "fallback"
        self.config = {
            "horizon": horizon,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "retrain_step": retrain_step,
            "backend": self._backend,
        }

    @property
    def name(self) -> str:
        return "prophet_forecast"

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=price.index)
        exits = pd.Series(False, index=price.index)
        if len(price) < self.min_train + self.horizon:
            return entries, exits

        last_pred: float | None = None
        next_retrain = self.min_train

        for i in range(self.min_train, len(price)):
            if i >= next_retrain or last_pred is None:
                hist = price.iloc[:i]
                if self._backend == "prophet":
                    last_pred = _prophet_predict_return(hist, self.horizon)
                    if last_pred is None:
                        last_pred = _fallback_predict_return(hist, self.horizon)
                else:
                    last_pred = _fallback_predict_return(hist, self.horizon)
                next_retrain = i + self.retrain_step

            if last_pred is None:
                continue
            if last_pred >= self.buy_threshold:
                entries.iloc[i] = True
            elif last_pred <= -self.sell_threshold:
                exits.iloc[i] = True

        return entries, exits

    def build_trade_reasons(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)
        entry_values = pd.Series(None, index=price.index, dtype=float)
        exit_values = pd.Series(None, index=price.index, dtype=float)
        entries, exits = self.generate_signals(price)
        backend = self._backend
        for i in price.index[entries]:
            entry_reasons.loc[i] = f"Prophet({backend}) forecast ≥ +{self.buy_threshold*100:.1f}%"
            entry_values.loc[i] = float(price.loc[i])
        for i in price.index[exits]:
            exit_reasons.loc[i] = f"Prophet({backend}) forecast ≤ −{self.sell_threshold*100:.1f}%"
            exit_values.loc[i] = float(price.loc[i])
        return entry_reasons, exit_reasons, entry_values, exit_values
