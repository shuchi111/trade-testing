"""Strategy: RSI oversold/overbought."""
import numpy as np
import pandas as pd
import vectorbt as vbt

from .base import Strategy


class RsiStrategy(Strategy):
    def __init__(self, window: int = 14, oversold: int = 30, overbought: int = 70):
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return "rsi"

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        rsi = vbt.RSI.run(price, self.window)
        entries = rsi.rsi_crossed_below(self.oversold)
        raw = rsi.rsi_crossed_above(self.overbought).shift(1)
        exits = pd.Series(np.where(raw.isna(), False, raw.to_numpy(dtype=bool)),
                          index=raw.index, dtype=bool)
        return entries, exits

    def build_trade_reasons(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        rsi = vbt.RSI.run(price, self.window).rsi
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)

        entries, exits = self.generate_signals(price)

        for i in price.index:
            if entries.get(i, False):
                val = rsi.get(i, None)
                entry_reasons[i] = f"RSI={val:.1f} crossed below {self.oversold} (oversold → expect bounce)"
            if exits.get(i, False):
                val = rsi.get(i, None)
                exit_reasons[i] = f"RSI={val:.1f} crossed above {self.overbought} (overbought → expect pullback)"

        return entry_reasons, exit_reasons, rsi, rsi

    @property
    def config(self) -> dict:
        return {"window": self.window, "oversold": self.oversold, "overbought": self.overbought}
