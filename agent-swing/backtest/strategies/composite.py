"""Composite strategy: weighted RSI + MACD + Bollinger + SMA votes."""
import numpy as np
import pandas as pd
import vectorbt as vbt

from .base import Strategy


class CompositeStrategy(Strategy):
    def __init__(
        self,
        w_rsi: float = 0.25,
        w_macd: float = 0.25,
        w_bb: float = 0.25,
        w_sma: float = 0.25,
        buy_threshold: float = 0.5,
        sell_threshold: float = -0.5,
        rsi_window: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        sma_fast: int = 20,
        sma_slow: int = 50,
        bb_window: int = 20,
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        self.w_rsi = w_rsi
        self.w_macd = w_macd
        self.w_bb = w_bb
        self.w_sma = w_sma
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    @property
    def name(self) -> str:
        return "composite"

    def _score_series(self, price: pd.Series) -> pd.Series:
        rsi = vbt.RSI.run(price, self.rsi_window).rsi
        macd = vbt.MACD.run(price, self.macd_fast, self.macd_slow, self.macd_signal)
        bb = vbt.BBANDS.run(price, self.bb_window, self.bb_std)
        fast = vbt.MA.run(price, self.sma_fast).ma
        slow = vbt.MA.run(price, self.sma_slow).ma

        rsi_vote = pd.Series(0.0, index=price.index)
        rsi_vote[rsi < self.rsi_oversold] = 1.0
        rsi_vote[rsi > self.rsi_overbought] = -1.0

        macd_vote = pd.Series(0.0, index=price.index)
        macd_vote[macd.macd > macd.signal] = 1.0
        macd_vote[macd.macd < macd.signal] = -1.0

        bb_vote = pd.Series(0.0, index=price.index)
        bb_vote[price <= bb.lower] = 1.0
        bb_vote[price >= bb.upper] = -1.0

        sma_vote = pd.Series(0.0, index=price.index)
        sma_vote[fast > slow] = 1.0
        sma_vote[fast < slow] = -1.0

        return (
            self.w_rsi * rsi_vote
            + self.w_macd * macd_vote
            + self.w_bb * bb_vote
            + self.w_sma * sma_vote
        )

    def generate_signals(self, price: pd.Series) -> tuple[pd.Series, pd.Series]:
        score = self._score_series(price)
        entries = (score >= self.buy_threshold) & (score.shift(1) < self.buy_threshold)
        raw = ((score <= self.sell_threshold) & (score.shift(1) > self.sell_threshold)).shift(1)
        exits = pd.Series(np.where(raw.isna(), False, raw.to_numpy(dtype=bool)),
                          index=raw.index, dtype=bool)
        return entries.fillna(False), exits

    def build_trade_reasons(self, price: pd.Series):
        score = self._score_series(price)
        entry_reasons = pd.Series("", index=price.index, dtype=object)
        exit_reasons = pd.Series("", index=price.index, dtype=object)
        entries, exits = self.generate_signals(price)
        for i in price.index:
            if entries.get(i, False):
                entry_reasons[i] = f"Composite score={score.get(i, 0):.2f} crossed buy threshold {self.buy_threshold}"
            if exits.get(i, False):
                exit_reasons[i] = f"Composite score={score.get(i, 0):.2f} crossed sell threshold {self.sell_threshold}"
        return entry_reasons, exit_reasons, score, score

    @property
    def config(self) -> dict:
        return {
            "w_rsi": self.w_rsi,
            "w_macd": self.w_macd,
            "w_bb": self.w_bb,
            "w_sma": self.w_sma,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
        }
