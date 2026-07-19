"""Strategy: Keltner Channel mean reversion (video winner family).

Band width uses rolling realized volatility (std of log returns) because the
backtest strategy API passes close prices only — not full OHLC — so classic
Wilder ATR cannot be computed without changing the runner contract.
"""
import numpy as np
import pandas as pd

from ..factors import realized_volatility
from .base import Strategy


class KeltnerStrategy(Strategy):
    def __init__(self, window: int = 20, atr_mult: float = 2.0):
        self.window = window
        self.atr_mult = atr_mult

    @property
    def name(self) -> str:
        return "keltner"

    def _channels(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        mid = price.rolling(self.window, min_periods=max(2, self.window // 2)).mean()
        rv = realized_volatility(price, self.window)
        # Log-return std → approximate price band: mid ± mult * mid * σ
        band = mid * rv * self.atr_mult
        upper = mid + band
        lower = mid - band
        return lower, mid, upper

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        lower, _mid, upper = self._channels(price)
        entries = price < lower
        raw = (price > upper).shift(1)
        exits = pd.Series(
            np.where(raw.isna(), False, raw.to_numpy(dtype=bool)),
            index=raw.index,
            dtype=bool,
        )
        return entries, exits

    def build_trade_reasons(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        lower, mid, upper = self._channels(price)
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)
        entries, exits = self.generate_signals(price)
        for i in price.index:
            if entries.get(i, False):
                entry_reasons[i] = (
                    f"Price below Keltner lower (window={self.window}, "
                    f"σ×{self.atr_mult}) — mean reversion buy"
                )
            if exits.get(i, False):
                exit_reasons[i] = (
                    f"Price above Keltner upper (window={self.window}, "
                    f"σ×{self.atr_mult}) — mean reversion sell"
                )
        return entry_reasons, exit_reasons, lower, upper

    @property
    def config(self) -> dict:
        return {"window": self.window, "atr_mult": self.atr_mult}
