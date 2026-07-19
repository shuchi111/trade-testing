"""Abstract base class for all backtest strategies."""
from abc import ABC, abstractmethod

import pandas as pd
import vectorbt as vbt

from ..config import FREQ_LABEL


class Strategy(ABC):
    """Base class for all backtest strategies."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Return (entries, exits) boolean Series aligned with price index."""
        ...

    def build_trade_reasons(self, price: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Return (entry_reasons, exit_reasons, entry_values, exit_values) as string/float Series.
        Override in subclasses to provide strategy-specific reasons.
        Defaults to generic reasons.
        """
        n = len(price)
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)
        entry_values = pd.Series(None, index=price.index, dtype=float)
        exit_values = pd.Series(None, index=price.index, dtype=float)
        return entry_reasons, exit_reasons, entry_values, exit_values

    def run(
        self,
        price: pd.Series,
        init_cash: float = 100_000,
        fees: float = 0.001,
        slippage: float = 0.002,
    ) -> vbt.Portfolio:
        entries, exits = self.generate_signals(price)
        return vbt.Portfolio.from_signals(
            price,
            entries,
            exits,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            freq=FREQ_LABEL,
        )
