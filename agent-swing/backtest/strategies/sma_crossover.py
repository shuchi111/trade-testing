"""Strategy: SMA (Simple Moving Average) crossover."""
import pandas as pd
import vectorbt as vbt

from .base import Strategy


class SmaCrossoverStrategy(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return "sma_crossover"

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        fast_ma = vbt.MA.run(price, self.fast)
        slow_ma = vbt.MA.run(price, self.slow)
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        return entries, exits

    def build_trade_reasons(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        fast_s = vbt.MA.run(price, self.fast).ma
        slow_s = vbt.MA.run(price, self.slow).ma
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)

        entries, exits = self.generate_signals(price)

        for i in price.index:
            if entries.get(i, False):
                f_val = fast_s.get(i, None)
                s_val = slow_s.get(i, None)
                entry_reasons[i] = f"SMA{self.fast}={f_val:.2f} crossed above SMA{self.slow}={s_val:.2f} (uptrend starting)"
            if exits.get(i, False):
                f_val = fast_s.get(i, None)
                s_val = slow_s.get(i, None)
                exit_reasons[i] = f"SMA{self.fast}={f_val:.2f} crossed below SMA{self.slow}={s_val:.2f} (uptrend ending)"

        return entry_reasons, exit_reasons, fast_s, slow_s

    @property
    def config(self) -> dict:
        return {"fast": self.fast, "slow": self.slow}
