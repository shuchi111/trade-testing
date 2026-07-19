"""Benchmark strategy: buy on first day, hold till end."""
import pandas as pd
import vectorbt as vbt

from .base import Strategy
from ..config import FREQ_LABEL


class BuyHoldStrategy(Strategy):
    @property
    def name(self) -> str:
        return "buy_hold"

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=price.index)
        entries.iloc[0] = True
        exits = pd.Series(False, index=price.index)
        return entries, exits

    def build_trade_reasons(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)
        entry_values = pd.Series(None, index=price.index, dtype=float)
        exit_values = pd.Series(None, index=price.index, dtype=float)

        entry_reasons.iloc[0] = f"Buy & Hold — bought on day 1 at {price.iloc[0]:.2f}"
        entry_values.iloc[0] = price.iloc[0]

        return entry_reasons, exit_reasons, entry_values, exit_values

    def run(self, price: pd.Series, init_cash: float = 100_000, **kwargs) -> vbt.Portfolio:
        return vbt.Portfolio.from_holding(price, init_cash=init_cash, freq=FREQ_LABEL)
