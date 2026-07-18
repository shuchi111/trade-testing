"""Strategy: Bollinger Bands mean reversion."""
import numpy as np
import pandas as pd
import vectorbt as vbt

from .base import Strategy


class BollingerStrategy(Strategy):
    def __init__(self, window: int = 20, std_dev: float = 2.0):
        self.window = window
        self.std_dev = std_dev

    @property
    def name(self) -> str:
        return "bollinger"

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        bb = vbt.BBANDS.run(price, self.window, self.std_dev)
        entries = price < bb.lower
        raw = (price > bb.upper).shift(1)
        exits = pd.Series(np.where(raw.isna(), False, raw.to_numpy(dtype=bool)),
                          index=raw.index, dtype=bool)
        return entries, exits

    def build_trade_reasons(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        bb = vbt.BBANDS.run(price, self.window, self.std_dev)
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)

        entries, exits = self.generate_signals(price)

        for i in price.index:
            if entries.get(i, False):
                p = price.get(i, None)
                lb = bb.lower.get(i, None)
                entry_reasons[i] = f"Price={p:.2f} touched lower band={lb:.2f} (unusually cheap → buy)"
            if exits.get(i, False):
                p = price.get(i, None)
                ub = bb.upper.get(i, None)
                exit_reasons[i] = f"Price={p:.2f} touched upper band={ub:.2f} (unusually expensive → sell)"

        return entry_reasons, exit_reasons, bb.lower, bb.upper

    @property
    def config(self) -> dict:
        return {"window": self.window, "std_dev": self.std_dev}
