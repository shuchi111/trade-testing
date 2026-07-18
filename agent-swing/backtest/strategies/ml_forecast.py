"""Strategy: LightGBM ML forecasting on Qlib Alpha158-style features.

Qlib's headline baseline (plan §1.2): a LightGBM model trained on the
``factors.compute_factors`` feature frame predicts the N-day forward return;
the strategy emits BUY when the predicted return crosses above a threshold and
SELL when it crosses below the negative threshold.

Walk-forward discipline (no look-ahead):
  - The signal at bar ``t`` is produced by a model that only saw bars strictly
    before ``t``. Concretely, an expanding window grows by one bar per step;
    features and the forward target are aligned so that predicting bar ``t``
    never observes any return at or after ``t``.
  - This is the lightweight equivalent of Qlib's rolling retrain (plan §1.3)
    done inline per backtest; a separate scheduled retrain can land later in
    trade-circleci-cron.

Convention-compatible with the backtest package: inherits ``Strategy``, returns
``(entries, exits)`` boolean Series from ``generate_signals``, exposes a
``config`` dict, and stores ML metadata on the instance (``last_train_rows``,
``last_feature_count``, ``last_retrain_date``, ``last_horizon``) for the runner
to persist via migration 009.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import Strategy
from ..factors import compute_factors

logger = logging.getLogger(__name__)


class MlForecastStrategy(Strategy):
    """LightGBM walk-forward forecasting strategy (Qlib §1.2 baseline)."""

    def __init__(
        self,
        horizon: int = 5,
        min_train_rows: int = 120,
        buy_threshold: float = 0.005,
        sell_threshold: float = 0.005,
        retrain_step: int = 1,
        lgb_params: dict | None = None,
    ):
        self.horizon = horizon
        self.min_train_rows = min_train_rows
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.retrain_step = max(1, int(retrain_step))
        self.lgb_params = lgb_params or {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "n_jobs": 1,
        }
        # Populated by generate_signals; read by the runner to persist ml_* cols.
        self.last_train_rows: int | None = None
        self.last_feature_count: int | None = None
        self.last_retrain_date: pd.Timestamp | None = None
        self.last_horizon: int | None = None
        # Last prediction series (for trade-reason display).
        self.last_predictions: pd.Series | None = None
        # Top feature importances from the most recent walk-forward fold.
        self.last_feature_importances: dict[str, float] | None = None
        # Optional volume aligned to price (set by runner / predict CLI).
        self._volume: pd.Series | None = None
        # Cached signals so build_trade_reasons can reuse them without retraining.
        self._cached_entries: pd.Series | None = None
        self._cached_exits: pd.Series | None = None

    @property
    def name(self) -> str:
        return "ml_forecast"

    def _build_model(self):
        """Lazy-import LightGBM so the module imports even before deps are pip-installed."""
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**self.lgb_params)

    def _prepare_feature_frame(self, price: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Index, list[str]] | None:
        """Build factor features + forward-return target aligned to ``price``."""
        features = compute_factors(price, volume=self._volume)
        target = price.pct_change(self.horizon).shift(-self.horizon)

        df = features.copy()
        df["__target__"] = target
        df = df.replace([np.inf, -np.inf], np.nan)
        feature_cols = list(features.columns)

        usable_idx = df.dropna(subset=feature_cols).index
        if len(usable_idx) < self.min_train_rows + self.horizon:
            return None

        X_all = df.loc[usable_idx, feature_cols]
        usable_targets = df.loc[usable_idx, "__target__"]
        return X_all, usable_targets, usable_idx, feature_cols

    def _store_importances(self, model, feature_cols: list[str]) -> None:
        imps = getattr(model, "feature_importances_", None)
        if imps is not None and len(imps) == len(feature_cols):
            self.last_feature_importances = {
                feature_cols[j]: float(imps[j]) for j in range(len(feature_cols))
            }

    def predict_live(self, price: pd.Series) -> float | None:
        """Train once on realized labels and predict the latest bar only.

        Much faster than the full walk-forward loop in ``generate_signals`` —
        intended for ``/api/ml/predict`` and the ML tab live signal.
        """
        prepared = self._prepare_feature_frame(price)
        if prepared is None:
            return None

        X_all, usable_targets, usable_idx, feature_cols = prepared
        n = len(usable_idx)
        i = n - 1
        train_end = i - self.horizon
        if train_end <= 0:
            return None

        train_targets = usable_targets.iloc[:train_end]
        train_mask = train_targets.notna()
        if int(train_mask.sum()) < self.min_train_rows:
            return None

        X_train = X_all.iloc[:train_end].loc[train_mask]
        y_train = train_targets.loc[train_mask]

        model = self._build_model()
        try:
            # Keep DataFrame column names for fit + predict (avoids sklearn warnings).
            model.fit(X_train, y_train)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("ml_forecast predict_live fit failed: %s", e)
            return None

        self.last_train_rows = len(X_train)
        self.last_feature_count = len(feature_cols)
        self.last_retrain_date = usable_idx[i]
        self.last_horizon = self.horizon
        self._store_importances(model, feature_cols)

        try:
            pred = float(model.predict(X_all.iloc[[i]])[0])
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("ml_forecast predict_live predict failed: %s", e)
            return None

        predictions = pd.Series(np.nan, index=price.index)
        entries = pd.Series(False, index=price.index)
        exits = pd.Series(False, index=price.index)

        full_pos = price.index.get_indexer([usable_idx[i]])[0]
        predictions.iloc[full_pos] = pred
        if pred > self.buy_threshold:
            entries.iloc[full_pos] = True
        elif pred < -self.sell_threshold:
            exits.iloc[full_pos] = True

        self.last_predictions = predictions
        self._cached_entries = entries
        self._cached_exits = exits
        return pred

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Walk-forward LightGBM on Alpha158 features.

        Returns ``(entries, exits)`` aligned with ``price``. Bars before the
        first trainable fold (insufficient history) produce no signal.
        """
        entries = pd.Series(False, index=price.index)
        exits = pd.Series(False, index=price.index)
        if price.empty:
            self._cached_entries = entries
            self._cached_exits = exits
            return entries, exits

        # ── Features + target ──────────────────────────────────────────
        prepared = self._prepare_feature_frame(price)
        if prepared is None:
            logger.warning(
                "ml_forecast: only %d usable rows (need >= %d) — no signals emitted.",
                0, self.min_train_rows + self.horizon,
            )
            self.last_predictions = pd.Series(np.nan, index=price.index)
            self._cached_entries = entries
            self._cached_exits = exits
            return entries, exits

        X_all, usable_targets, usable_idx, feature_cols = prepared
        usable_pos = price.index.get_indexer(usable_idx)

        predictions = pd.Series(np.nan, index=price.index)
        model = self._build_model()
        train_rows_seen = 0
        last_fit_at = None
        n = len(usable_idx)
        model_fitted = False

        for i in range(n):
            train_end = i - self.horizon
            if train_end <= 0:
                continue
            train_targets = usable_targets.iloc[:train_end]
            train_mask = train_targets.notna()
            if train_mask.sum() < self.min_train_rows:
                continue
            X_train = X_all.iloc[:train_end].loc[train_mask]
            y_train = train_targets.loc[train_mask]

            should_fit = (not model_fitted) or (i % self.retrain_step == 0)
            if should_fit:
                try:
                    model.fit(X_train, y_train)
                    model_fitted = True
                    train_rows_seen = len(X_train)
                    last_fit_at = usable_idx[i]
                    self._store_importances(model, feature_cols)
                except Exception as e:  # pragma: no cover — defensive
                    logger.warning("ml_forecast: fit failed at %s: %s", usable_idx[i], e)
                    continue
            elif not model_fitted:
                continue

            try:
                pred = float(model.predict(X_all.iloc[[i]])[0])
            except Exception:  # pragma: no cover — defensive
                continue
            full_pos = usable_pos[i]
            predictions.iloc[full_pos] = pred

            if pred > self.buy_threshold:
                entries.iloc[full_pos] = True
            elif pred < -self.sell_threshold:
                exits.iloc[full_pos] = True

        self.last_train_rows = train_rows_seen or None
        self.last_feature_count = len(feature_cols)
        self.last_retrain_date = last_fit_at
        self.last_horizon = self.horizon
        self.last_predictions = predictions
        self._cached_entries = entries
        self._cached_exits = exits

        return entries, exits

    def build_trade_reasons(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Provide predicted-return values as the trade indicator.

        Reuses the signals cached by the preceding ``generate_signals`` /
        ``run`` call instead of retraining. Falls back to a fresh run only if
        no cache exists (e.g. called out of order).
        """
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)
        entry_values = pd.Series(np.nan, index=price.index, dtype=float)
        exit_values = pd.Series(np.nan, index=price.index, dtype=float)

        if self._cached_entries is not None and self._cached_exits is not None:
            entries = self._cached_entries
            exits = self._cached_exits
        else:
            entries, exits = self.generate_signals(price)
        preds = self.last_predictions if self.last_predictions is not None else pd.Series(
            np.nan, index=price.index
        )

        for i in price.index:
            if entries.get(i, False):
                p = preds.get(i, np.nan)
                entry_reasons[i] = f"LightGBM predicts +{self.horizon}d return above +{self.buy_threshold:.1%} threshold"
                if pd.notna(p):
                    entry_values[i] = float(p)
            if exits.get(i, False):
                p = preds.get(i, np.nan)
                exit_reasons[i] = f"LightGBM predicts +{self.horizon}d return below -{self.sell_threshold:.1%} threshold"
                if pd.notna(p):
                    exit_values[i] = float(p)

        return entry_reasons, exit_reasons, entry_values, exit_values

    @property
    def config(self) -> dict:
        return {
            "horizon": self.horizon,
            "min_train_rows": self.min_train_rows,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "retrain_step": self.retrain_step,
        }
