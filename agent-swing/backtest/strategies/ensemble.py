"""Strategy: Ensemble vote across SMA, RSI, Bollinger, and MACD signals.

BUY when majority of component strategies are long; SELL when majority flips short.
Mirrors the "Ensemble" layer from multi-strategy research dashboards.
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy
from .bollinger import BollingerStrategy
from .macd import MacdStrategy
from .rsi import RsiStrategy
from .sma_crossover import SmaCrossoverStrategy


class EnsembleStrategy(Strategy):
    def __init__(self, min_votes: int = 2):
        self.min_votes = min_votes
        self.components = [
            SmaCrossoverStrategy(fast=20, slow=50),
            RsiStrategy(window=14, oversold=30, overbought=70),
            BollingerStrategy(window=20, std_dev=2.0),
            MacdStrategy(fast=12, slow=26, signal=9),
        ]
        self.config = {"min_votes": min_votes, "components": [c.name for c in self.components]}

    @property
    def name(self) -> str:
        return "ensemble"

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        long_votes = pd.Series(0, index=price.index, dtype=int)
        exit_votes = pd.Series(0, index=price.index, dtype=int)

        for strat in self.components:
            entries, exits = strat.generate_signals(price)
            # Track position state per component: +1 when entry, 0 after exit
            position = pd.Series(0, index=price.index, dtype=int)
            pos = 0
            for i in range(len(price)):
                if bool(entries.iloc[i]):
                    pos = 1
                elif bool(exits.iloc[i]):
                    pos = 0
                position.iloc[i] = pos
            long_votes += position
            exit_votes += exits.astype(int)

        # Enter when votes cross above min_votes; exit when votes drop below
        was_long = False
        entries = pd.Series(False, index=price.index)
        exits = pd.Series(False, index=price.index)
        for i in range(len(price)):
            votes = int(long_votes.iloc[i])
            if not was_long and votes >= self.min_votes:
                entries.iloc[i] = True
                was_long = True
            elif was_long and votes < self.min_votes:
                exits.iloc[i] = True
                was_long = False

        return entries, exits
