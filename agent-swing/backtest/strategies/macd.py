"""Strategy: MACD crossover."""
import pandas as pd
import vectorbt as vbt

from .base import Strategy


class MacdStrategy(Strategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return "macd"

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        macd = vbt.MACD.run(price, self.fast, self.slow, self.signal)
        entries = macd.macd_crossed_above(macd.signal)
        raw = macd.macd_crossed_below(macd.signal).shift(1)
        exits = raw.where(raw.notna(), False).astype(bool)
        return entries, exits

    def build_trade_reasons(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        macd_obj = vbt.MACD.run(price, self.fast, self.slow, self.signal)
        macd_line = macd_obj.macd
        signal_line = macd_obj.signal
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)

        entries, exits = self.generate_signals(price)

        for i in price.index:
            if entries.get(i, False):
                m = macd_line.get(i, None)
                s = signal_line.get(i, None)
                entry_reasons[i] = f"MACD={m:.2f} crossed above Signal={s:.2f} (bullish momentum)"
            if exits.get(i, False):
                m = macd_line.get(i, None)
                s = signal_line.get(i, None)
                exit_reasons[i] = f"MACD={m:.2f} crossed below Signal={s:.2f} (bearish momentum)"

        return entry_reasons, exit_reasons, macd_line, signal_line

    @property
    def config(self) -> dict:
        return {"fast": self.fast, "slow": self.slow, "signal": self.signal}
