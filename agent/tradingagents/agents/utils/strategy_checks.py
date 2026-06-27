from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradingagents.dataflows.symbol_utils import normalize_symbol


@dataclass(frozen=True)
class MinerviniEvidence:
    ticker: str
    latest_date: str
    latest_close: float
    passed_count: int
    total_count: int
    passed: bool
    lines: list[str]


def _fmt_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _float(value: Any) -> float | None:
    try:
        out = float(value)
        return out if pd.notna(out) else None
    except Exception:
        return None


def build_minervini_evidence(ticker: str, period: str = "1y") -> MinerviniEvidence:
    """Compute a deterministic Minervini-style trend-template evidence block."""
    import yfinance as yf

    symbol = normalize_symbol(ticker)
    hist = yf.Ticker(symbol).history(period=period)
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError(f"No real OHLCV history available for Minervini check: {symbol}")

    close = hist["Close"].dropna()
    if len(close) < 200:
        raise ValueError(f"Need at least 200 closes for Minervini check: {symbol}")

    latest_close = _float(close.iloc[-1])
    if latest_close is None or latest_close <= 0:
        raise ValueError(f"Invalid latest close for Minervini check: {symbol}")

    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    high_52w = _float(close.tail(252).max())
    low_52w = _float(close.tail(252).min())
    avg_vol50 = None
    latest_vol = None
    if "Volume" in hist.columns:
        volumes = hist["Volume"].dropna()
        if not volumes.empty:
            latest_vol = _float(volumes.iloc[-1])
            avg_vol50 = _float(volumes.tail(50).mean())

    latest_sma50 = _float(sma50.iloc[-1])
    latest_sma150 = _float(sma150.iloc[-1])
    latest_sma200 = _float(sma200.iloc[-1])
    sma200_20ago = _float(sma200.iloc[-21]) if len(sma200.dropna()) >= 21 else None

    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "Price above 50 DMA",
            latest_sma50 is not None and latest_close > latest_sma50,
            f"close={latest_close:.2f}, 50DMA={latest_sma50:.2f}" if latest_sma50 else "50DMA=n/a",
        )
    )
    checks.append(
        (
            "Price above 150 DMA",
            latest_sma150 is not None and latest_close > latest_sma150,
            f"close={latest_close:.2f}, 150DMA={latest_sma150:.2f}" if latest_sma150 else "150DMA=n/a",
        )
    )
    checks.append(
        (
            "Price above 200 DMA",
            latest_sma200 is not None and latest_close > latest_sma200,
            f"close={latest_close:.2f}, 200DMA={latest_sma200:.2f}" if latest_sma200 else "200DMA=n/a",
        )
    )
    checks.append(
        (
            "150 DMA above 200 DMA",
            latest_sma150 is not None and latest_sma200 is not None and latest_sma150 > latest_sma200,
            f"150DMA={latest_sma150:.2f}, 200DMA={latest_sma200:.2f}"
            if latest_sma150 and latest_sma200
            else "moving averages=n/a",
        )
    )
    checks.append(
        (
            "200 DMA trending up",
            latest_sma200 is not None and sma200_20ago is not None and latest_sma200 > sma200_20ago,
            f"200DMA now={latest_sma200:.2f}, 20 sessions ago={sma200_20ago:.2f}"
            if latest_sma200 and sma200_20ago
            else "200DMA trend=n/a",
        )
    )
    checks.append(
        (
            "Price near 52-week high",
            high_52w is not None and latest_close >= high_52w * 0.75,
            f"close={latest_close:.2f}, 52w high={high_52w:.2f}" if high_52w else "52w high=n/a",
        )
    )
    checks.append(
        (
            "Price above 52-week low",
            low_52w is not None and latest_close >= low_52w * 1.25,
            f"close={latest_close:.2f}, 52w low={low_52w:.2f}" if low_52w else "52w low=n/a",
        )
    )
    checks.append(
        (
            "Volume not dry",
            latest_vol is not None and avg_vol50 is not None and latest_vol >= avg_vol50 * 0.75,
            f"latest volume={latest_vol:.0f}, 50-day avg={avg_vol50:.0f}"
            if latest_vol and avg_vol50
            else "volume=n/a",
        )
    )

    passed_count = sum(1 for _, passed, _ in checks if passed)
    lines = [f"{name}: {_fmt_bool(passed)} ({detail})" for name, passed, detail in checks]
    return MinerviniEvidence(
        ticker=symbol,
        latest_date=str(close.index[-1].date()),
        latest_close=latest_close,
        passed_count=passed_count,
        total_count=len(checks),
        passed=passed_count >= 6,
        lines=lines,
    )


def format_minervini_evidence(ticker: str, period: str = "1y") -> str:
    evidence = build_minervini_evidence(ticker, period=period)
    header = (
        f"=== MINERVINI STRATEGY EVIDENCE ===\n"
        f"{evidence.ticker}: {evidence.passed_count}/{evidence.total_count} checks passed "
        f"on {evidence.latest_date}; overall {'PASS' if evidence.passed else 'FAIL'}."
    )
    return header + "\n" + "\n".join(f"- {line}" for line in evidence.lines)
