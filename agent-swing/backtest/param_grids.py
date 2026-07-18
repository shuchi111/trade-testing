"""Parameter grids for VectorBT grid search."""

from __future__ import annotations


def sma_grid() -> list[dict]:
    combos = []
    for fast in range(5, 51, 5):
        for slow in range(20, 201, 20):
            if fast < slow:
                combos.append({"fast": fast, "slow": slow})
    return combos


def rsi_grid() -> list[dict]:
    combos = []
    for window in range(7, 22, 2):
        for oversold in range(20, 36, 5):
            for overbought in range(65, 81, 5):
                if oversold < overbought:
                    combos.append({"window": window, "oversold": oversold, "overbought": overbought})
    return combos


def bollinger_grid() -> list[dict]:
    combos = []
    for window in range(10, 31, 5):
        for std in [1.5, 2.0, 2.5]:
            combos.append({"window": window, "std_dev": std})
    return combos


def macd_grid() -> list[dict]:
    combos = []
    for fast in range(8, 16, 2):
        for slow in range(20, 31, 2):
            for signal in range(7, 13, 2):
                if fast < slow:
                    combos.append({"fast": fast, "slow": slow, "signal": signal})
    return combos


def composite_weight_grid() -> list[dict]:
    """Weight combos for composite strategy (sum to 1.0)."""
    combos = []
    weights = [0.1, 0.2, 0.25, 0.3, 0.4]
    for w_rsi in weights:
        for w_macd in weights:
            for w_bb in weights:
                w_sma = round(1.0 - w_rsi - w_macd - w_bb, 2)
                if w_sma < 0.1:
                    continue
                for buy_th in [0.4, 0.5, 0.6]:
                    for sell_th in [-0.6, -0.5, -0.4]:
                        combos.append({
                            "w_rsi": w_rsi,
                            "w_macd": w_macd,
                            "w_bb": w_bb,
                            "w_sma": w_sma,
                            "buy_threshold": buy_th,
                            "sell_threshold": sell_th,
                        })
    return combos


def keltner_grid() -> list[dict]:
    combos = []
    for window in range(10, 31, 5):
        for atr_mult in [1.5, 2.0, 2.5, 3.0]:
            combos.append({"window": window, "atr_mult": atr_mult})
    return combos


def prophet_grid() -> list[dict]:
    """Prophet / seasonal-trend forecast parameter sweep (kept lean — each fit is expensive)."""
    combos = []
    for horizon in [5, 7, 10]:
        for buy_threshold in [0.005, 0.008, 0.01]:
            for sell_threshold in [0.005, 0.008]:
                for retrain_step in [42, 63]:
                    combos.append({
                        "horizon": horizon,
                        "buy_threshold": buy_threshold,
                        "sell_threshold": sell_threshold,
                        "retrain_step": retrain_step,
                    })
    return combos


def ensemble_grid() -> list[dict]:
    return [{"min_votes": v} for v in [2, 3]]


GRIDS = {
    "sma_crossover": sma_grid,
    "rsi": rsi_grid,
    "bollinger": bollinger_grid,
    "keltner": keltner_grid,
    "macd": macd_grid,
    "composite": composite_weight_grid,
    "prophet_forecast": prophet_grid,
    "ensemble": ensemble_grid,
}


def total_grid_count() -> int:
    return sum(len(fn()) for fn in GRIDS.values())
