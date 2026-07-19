"""Claude Skills Pack — 5 screeners + TA / Nifty / India VIX + trade plan.

Runs during **Build context FIRST** (before any agent signal). Each screener
produces a scored result; ``connect_screener_results`` cross-links them into a
consensus + trade plan that decision agents must observe.

Screeners (Claude Skills Pack):
  1. VCP — Volatility Contraction Pattern
  2. PEAD — Post-Earnings Announcement Drift
  3. Relative Strength — vs Nifty 50
  4. Volume Breakout
  5. Momentum — rate-of-change + RSI

Plus: TA snapshot, Nifty regime, India VIX regime, connected trade plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd  # type: ignore[reportMissingImports]
import yfinance as yf  # type: ignore[reportMissingImports]

from tradingagents.dataflows.symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

NIFTY_SYMBOL = "^NSEI"
INDIA_VIX_SYMBOL = "^INDIAVIX"

# Soft weights used when connecting screener scores into one consensus.
_SCREENER_WEIGHTS = {
    "VCP": 0.22,
    "PEAD": 0.18,
    "Relative Strength": 0.20,
    "Volume Breakout": 0.20,
    "Momentum": 0.20,
}


@dataclass
class ScreenerResult:
    name: str
    score: float  # 0–100
    signal: str  # bullish | neutral | bearish | unavailable
    summary: str
    facts: list[str] = field(default_factory=list)
    levels: dict[str, float] = field(default_factory=dict)
    supports: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


def _f(value: Any) -> float | None:
    try:
        out = float(value)
        return out if pd.notna(out) else None
    except Exception:
        return None


def _signal_from_score(score: float) -> str:
    if score >= 65:
        return "bullish"
    if score <= 35:
        return "bearish"
    return "neutral"


def _resolve_yahoo(ticker: str) -> str:
    symbol = normalize_symbol(ticker)
    if not symbol:
        return ticker.strip().upper()
    # Prefer NSE suffix for bare Indian cash names used in some callers.
    if "." not in symbol and "-" not in symbol and "=" not in symbol and not symbol.startswith("^"):
        return f"{symbol}.NS"
    return symbol


def load_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch daily OHLCV; raises ValueError when history is unusable."""
    symbol = _resolve_yahoo(ticker)
    hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError(f"No OHLCV for {symbol}")
    out = hist.copy()
    out.columns = [str(c).title() for c in out.columns]
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(out.columns)):
        raise ValueError(f"Incomplete OHLCV columns for {symbol}")
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    return out.dropna(subset=["Close"])


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def screen_vcp(df: pd.DataFrame) -> ScreenerResult:
    """Simplified VCP: contracting ranges into a pivot near recent highs."""
    name = "VCP"
    if len(df) < 60:
        return ScreenerResult(name, 0, "unavailable", "Need 60+ sessions for VCP", ["insufficient data"])

    window = df.tail(60)
    close = float(window["Close"].iloc[-1])
    # Three successive 20-session ranges — expect contraction.
    ranges: list[float] = []
    for i in range(3):
        chunk = window.iloc[i * 20 : (i + 1) * 20]
        if len(chunk) < 15:
            continue
        hi = float(chunk["High"].max())
        lo = float(chunk["Low"].min())
        mid = (hi + lo) / 2 or 1.0
        ranges.append(((hi - lo) / mid) * 100)

    contractions = 0
    for i in range(1, len(ranges)):
        if ranges[i] < ranges[i - 1] * 0.85:
            contractions += 1

    high_60 = float(window["High"].max())
    pivot = high_60
    dist_to_pivot = ((pivot - close) / close) * 100 if close else 99.0
    near_pivot = dist_to_pivot <= 5.0
    atr = _f(_atr(df).iloc[-1]) or 0.0
    atr_pct = (atr / close) * 100 if close else 99.0

    score = 20.0
    score += min(40.0, contractions * 20.0)
    if near_pivot:
        score += 20.0
    if atr_pct < 3.0:
        score += 15.0
    elif atr_pct < 5.0:
        score += 8.0
    if close >= float(_sma(df["Close"], 50).iloc[-1] or 0):
        score += 5.0
    score = max(0.0, min(100.0, score))

    facts = [
        f"contractions={contractions}/2 possible across 20d windows",
        f"range depths={[round(r, 1) for r in ranges]}%",
        f"pivot≈{pivot:.2f}, close={close:.2f}, dist={dist_to_pivot:.1f}%",
        f"ATR%≈{atr_pct:.2f}",
    ]
    return ScreenerResult(
        name=name,
        score=score,
        signal=_signal_from_score(score),
        summary="Volatility contracting into a pivot" if contractions and near_pivot else "VCP incomplete or extended",
        facts=facts,
        levels={"pivot": pivot, "close": close, "atr": atr},
        supports=["Volume Breakout", "Momentum"] if score >= 60 else [],
        conflicts=["Momentum"] if score < 40 else [],
    )


def _daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV into ISO-week candles (Monday start)."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)
    # Normalize timezone-aware indexes so week grouping is stable.
    if getattr(work.index, "tz", None) is not None:
        work.index = work.index.tz_localize(None)
    weekly = work.resample("W-MON", label="left", closed="left").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna(subset=["Close"])
    return weekly


def _find_recent_earnings_date(ticker: str, lookback_days: int = 60) -> datetime | None:
    """Best-effort earnings date from yfinance (calendar or earnings history)."""
    symbol = _resolve_yahoo(ticker)
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    try:
        cal = yf.Ticker(symbol).calendar
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("earningsDate")
            if isinstance(raw, (list, tuple)) and raw:
                raw = raw[0]
            if raw is not None:
                dt = pd.Timestamp(raw).to_pydatetime().replace(tzinfo=None)
                if dt >= cutoff - timedelta(days=7):
                    return dt
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            for col in cal.columns:
                if "earn" in str(col).lower():
                    val = cal[col].iloc[0]
                    dt = pd.Timestamp(val).to_pydatetime().replace(tzinfo=None)
                    if dt >= cutoff - timedelta(days=7):
                        return dt
                    break
    except Exception as exc:
        logger.debug("earnings calendar unavailable for %s: %s", symbol, exc)

    try:
        dates = yf.Ticker(symbol).earnings_dates
        if dates is not None and not dates.empty:
            idx = pd.to_datetime(dates.index)
            for ts in sorted(idx, reverse=True):
                dt = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)
                if cutoff <= dt <= datetime.utcnow() + timedelta(days=1):
                    return dt
    except Exception as exc:
        logger.debug("earnings_dates unavailable for %s: %s", symbol, exc)
    return None


def _detect_gap_event(df: pd.DataFrame, *, min_gap_pct: float = 3.0, lookback: int = 45) -> dict[str, Any] | None:
    """Find the strongest recent gap-up day (proxy when earnings date is unknown)."""
    if len(df) < 5:
        return None
    window = df.tail(lookback)
    best: dict[str, Any] | None = None
    for i in range(1, len(window)):
        prev_close = float(window["Close"].iloc[i - 1])
        open_ = float(window["Open"].iloc[i])
        close = float(window["Close"].iloc[i])
        if prev_close <= 0:
            continue
        gap_pct = ((open_ / prev_close) - 1.0) * 100
        day_ret = ((close / prev_close) - 1.0) * 100
        if gap_pct < min_gap_pct and day_ret < min_gap_pct:
            continue
        strength = max(gap_pct, day_ret)
        candidate = {
            "date": window.index[i],
            "gap_pct": gap_pct,
            "day_ret": day_ret,
            "strength": strength,
            "close": close,
        }
        if best is None or strength > best["strength"]:
            best = candidate
    return best


def screen_pead(
    df: pd.DataFrame,
    *,
    ticker: str | None = None,
    earnings_date: datetime | None = None,
    watch_weeks: int = 6,
    min_gap_pct: float = 3.0,
) -> ScreenerResult:
    """PEAD — Post-Earnings Announcement Drift (gap-up → red weekly pullback → breakout)."""
    name = "PEAD"
    if len(df) < 30:
        return ScreenerResult(name, 0, "unavailable", "Need 30+ sessions for PEAD", ["insufficient data"])

    earn_dt = earnings_date
    if earn_dt is None and ticker:
        earn_dt = _find_recent_earnings_date(ticker, lookback_days=watch_weeks * 7 + 14)

    gap_event = None
    if earn_dt is not None:
        # Locate first session on/after earnings and measure gap vs prior close.
        idx = df.index
        if getattr(idx, "tz", None) is not None:
            earn_cmp = pd.Timestamp(earn_dt).tz_localize(idx.tz)
        else:
            earn_cmp = pd.Timestamp(earn_dt)
        post = df[df.index >= earn_cmp]
        if len(post) >= 1:
            loc = df.index.get_indexer([post.index[0]], method="pad")[0]
            if loc > 0:
                prev_close = float(df["Close"].iloc[loc - 1])
                open_ = float(df["Open"].iloc[loc])
                close = float(df["Close"].iloc[loc])
                gap_pct = ((open_ / prev_close) - 1.0) * 100 if prev_close else 0.0
                day_ret = ((close / prev_close) - 1.0) * 100 if prev_close else 0.0
                if gap_pct >= min_gap_pct or day_ret >= min_gap_pct:
                    gap_event = {
                        "date": df.index[loc],
                        "gap_pct": gap_pct,
                        "day_ret": day_ret,
                        "strength": max(gap_pct, day_ret),
                        "close": close,
                        "source": "earnings_date",
                    }
    if gap_event is None:
        gap_event = _detect_gap_event(df, min_gap_pct=min_gap_pct, lookback=watch_weeks * 7 + 10)
        if gap_event:
            gap_event["source"] = "price_gap_proxy"

    if gap_event is None:
        return ScreenerResult(
            name,
            15.0,
            "neutral",
            "No recent earnings gap-up in PEAD watch window",
            ["no qualifying gap-up ≥ {:.1f}%".format(min_gap_pct)],
            supports=[],
            conflicts=[],
        )

    weekly = _daily_to_weekly(df)
    gap_ts = pd.Timestamp(gap_event["date"])
    if getattr(weekly.index, "tz", None) is not None and gap_ts.tzinfo is None:
        gap_ts = gap_ts.tz_localize(weekly.index.tz)
    post_weeks = weekly[weekly.index >= (gap_ts - timedelta(days=7))].tail(watch_weeks + 1)
    latest_close = float(df["Close"].iloc[-1])

    stage = "MONITORING"
    red_high = red_low = None
    red_weeks = post_weeks[post_weeks["Close"] < post_weeks["Open"]] if not post_weeks.empty else post_weeks
    if not red_weeks.empty:
        red = red_weeks.iloc[-1]
        red_high = float(red["High"])
        red_low = float(red["Low"])
        after_red = post_weeks.loc[red.name:]
        broke = False
        if len(after_red) >= 2:
            nxt = after_red.iloc[1] if after_red.index[0] == red.name and len(after_red) > 1 else after_red.iloc[-1]
            broke = float(nxt["Close"]) >= red_high and float(nxt["Close"]) >= float(nxt["Open"])
        if latest_close >= red_high:
            broke = True
        stage = "BREAKOUT" if broke else "SIGNAL_READY"

    # Expire if too far past the gap without actionable structure.
    sessions_since = int((df.index[-1] - pd.Timestamp(gap_event["date"])).days) if len(df) else 99
    if sessions_since > watch_weeks * 7 and stage == "MONITORING":
        stage = "EXPIRED"

    score = 20.0
    gap_strength = float(gap_event.get("strength") or 0.0)
    score += min(25.0, gap_strength * 2.0)
    if stage == "BREAKOUT":
        score += 35.0
    elif stage == "SIGNAL_READY":
        score += 25.0
    elif stage == "MONITORING":
        score += 10.0
    else:
        score -= 10.0
    # Drift continuation: price still above gap-day close.
    if latest_close >= float(gap_event.get("close") or latest_close):
        score += 10.0
    score = max(0.0, min(100.0, score))

    facts = [
        f"stage={stage}",
        f"gap_source={gap_event.get('source')}, gap/day={gap_event.get('gap_pct', 0):+.1f}% / {gap_event.get('day_ret', 0):+.1f}%",
        f"event_date={pd.Timestamp(gap_event['date']).date()}",
        f"sessions_since_event≈{sessions_since}",
    ]
    if earn_dt is not None:
        facts.append(f"earnings_date={earn_dt.date()}")
    if red_high is not None:
        facts.append(f"red_week_high={red_high:.2f}, red_week_low={red_low:.2f}")

    summary = {
        "BREAKOUT": "PEAD breakout above post-earnings red weekly candle",
        "SIGNAL_READY": "PEAD red-week pullback ready — wait for breakout",
        "MONITORING": "Post-gap PEAD watch — no red weekly pullback yet",
        "EXPIRED": "PEAD window faded without clean signal",
    }.get(stage, "PEAD mixed")

    return ScreenerResult(
        name=name,
        score=score,
        signal=_signal_from_score(score),
        summary=summary,
        facts=facts,
        levels={
            "close": latest_close,
            "gap_close": float(gap_event.get("close") or 0.0),
            "red_high": float(red_high or 0.0),
            "red_low": float(red_low or 0.0),
        },
        supports=["Momentum", "Relative Strength"] if score >= 60 else [],
        conflicts=["Volume Breakout"] if stage == "SIGNAL_READY" else [],
    )


def screen_relative_strength(stock: pd.DataFrame, nifty: pd.DataFrame | None) -> ScreenerResult:
    """Relative Strength vs Nifty over 63 / 126 sessions."""
    name = "Relative Strength"
    if nifty is None or nifty.empty or len(stock) < 70:
        return ScreenerResult(name, 0, "unavailable", "Nifty or stock history missing", ["benchmark unavailable"])

    aligned = pd.concat(
        [stock["Close"].rename("stock"), nifty["Close"].rename("bench")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 70:
        return ScreenerResult(name, 0, "unavailable", "Insufficient aligned sessions vs Nifty", ["alignment short"])

    def _ret(series: pd.Series, n: int) -> float:
        if len(series) <= n:
            return 0.0
        a = float(series.iloc[-1])
        b = float(series.iloc[-1 - n])
        return ((a / b) - 1.0) * 100 if b else 0.0

    stock_63 = _ret(aligned["stock"], 63)
    nifty_63 = _ret(aligned["bench"], 63)
    stock_126 = _ret(aligned["stock"], 126) if len(aligned) > 126 else stock_63
    nifty_126 = _ret(aligned["bench"], 126) if len(aligned) > 126 else nifty_63
    rs_63 = stock_63 - nifty_63
    rs_126 = stock_126 - nifty_126
    blended = 0.6 * rs_63 + 0.4 * rs_126

    # Map excess return into 0–100 (roughly −20%..+20% → 0..100).
    score = max(0.0, min(100.0, 50.0 + blended * 2.5))

    facts = [
        f"stock_63d={stock_63:+.1f}% vs nifty_63d={nifty_63:+.1f}% → RS={rs_63:+.1f}%",
        f"stock_126d={stock_126:+.1f}% vs nifty_126d={nifty_126:+.1f}% → RS={rs_126:+.1f}%",
        f"blended_excess={blended:+.1f}%",
    ]
    return ScreenerResult(
        name=name,
        score=score,
        signal=_signal_from_score(score),
        summary="Outperforming Nifty" if blended > 0 else "Lagging Nifty",
        facts=facts,
        levels={"rs_63": rs_63, "rs_126": rs_126},
        supports=["Momentum", "VCP"] if score >= 60 else [],
        conflicts=["Momentum"] if score < 40 else [],
    )


def screen_volume_breakout(df: pd.DataFrame) -> ScreenerResult:
    """Volume breakout through recent resistance with expanding volume."""
    name = "Volume Breakout"
    if len(df) < 40:
        return ScreenerResult(name, 0, "unavailable", "Need 40+ sessions", ["insufficient data"])

    look = df.tail(40)
    close = float(look["Close"].iloc[-1])
    resistance = float(look["High"].iloc[:-1].tail(20).max())
    vol = look["Volume"]
    avg_vol = float(vol.iloc[:-1].tail(20).mean() or 1.0)
    last_vol = float(vol.iloc[-1] or 0.0)
    vol_ratio = last_vol / avg_vol if avg_vol else 0.0
    broke = close >= resistance * 0.995
    body_strength = (close - float(look["Open"].iloc[-1])) / close * 100 if close else 0.0

    score = 10.0
    if broke:
        score += 35.0
    if vol_ratio >= 1.5:
        score += 30.0
    elif vol_ratio >= 1.2:
        score += 18.0
    if body_strength > 0:
        score += min(15.0, body_strength * 3)
    if close > float(_sma(df["Close"], 50).iloc[-1] or 0):
        score += 10.0
    score = max(0.0, min(100.0, score))

    facts = [
        f"resistance≈{resistance:.2f}, close={close:.2f}, broke={broke}",
        f"volume_ratio={vol_ratio:.2f}x 20d avg",
        f"day_body={body_strength:+.2f}%",
    ]
    return ScreenerResult(
        name=name,
        score=score,
        signal=_signal_from_score(score),
        summary="Breakout with volume" if broke and vol_ratio >= 1.2 else "No confirmed volume breakout",
        levels={"resistance": resistance, "close": close, "volume_ratio": vol_ratio},
        facts=facts,
        supports=["VCP", "Momentum"] if score >= 60 else [],
        conflicts=["PEAD"] if score >= 70 else [],  # chasing breakout vs waiting PEAD red-week entry
    )


def screen_momentum(df: pd.DataFrame) -> ScreenerResult:
    """Momentum via 21d ROC and RSI(14)."""
    name = "Momentum"
    if len(df) < 40:
        return ScreenerResult(name, 0, "unavailable", "Need 40+ sessions", ["insufficient data"])

    close = df["Close"]
    roc21 = ((float(close.iloc[-1]) / float(close.iloc[-22])) - 1.0) * 100 if len(close) > 22 else 0.0
    rsi = _f(_rsi(close).iloc[-1]) or 50.0

    score = 50.0
    score += max(-25.0, min(25.0, roc21 * 2.0))
    if 55 <= rsi <= 70:
        score += 20.0
    elif 45 <= rsi < 55:
        score += 8.0
    elif rsi > 75:
        score -= 10.0  # extended
    elif rsi < 40:
        score -= 15.0
    score = max(0.0, min(100.0, score))

    facts = [f"ROC21={roc21:+.1f}%", f"RSI14={rsi:.1f}"]
    return ScreenerResult(
        name=name,
        score=score,
        signal=_signal_from_score(score),
        summary="Constructive momentum" if score >= 60 else "Momentum mixed or weak",
        facts=facts,
        levels={"roc21": roc21, "rsi14": rsi},
        supports=["Relative Strength", "Volume Breakout"] if score >= 60 else [],
        conflicts=["Relative Strength"] if score >= 65 else [],
    )


def analyze_nifty_regime(nifty: pd.DataFrame | None) -> dict[str, Any]:
    if nifty is None or len(nifty) < 50:
        return {"regime": "unknown", "score": 50.0, "facts": ["Nifty history unavailable"]}
    close = nifty["Close"]
    last = float(close.iloc[-1])
    sma50 = _f(_sma(close, 50).iloc[-1])
    sma200 = _f(_sma(close, 200).iloc[-1]) if len(close) >= 200 else None
    ret20 = ((last / float(close.iloc[-21])) - 1.0) * 100 if len(close) > 21 else 0.0
    score = 50.0
    regime = "neutral"
    if sma50 and last > sma50:
        score += 15
    if sma200 and last > sma200:
        score += 20
        regime = "risk-on"
    if sma50 and sma200 and sma50 < sma200 and last < sma50:
        score -= 25
        regime = "risk-off"
    if ret20 < -3:
        score -= 10
        if regime != "risk-off":
            regime = "cautious"
    elif ret20 > 3:
        score += 10
    score = max(0.0, min(100.0, score))
    return {
        "regime": regime,
        "score": score,
        "close": last,
        "ret20": ret20,
        "facts": [
            f"Nifty close={last:.2f}, 20d={ret20:+.1f}%",
            f"above_50DMA={bool(sma50 and last > sma50)}, above_200DMA={bool(sma200 and last > sma200)}",
            f"regime={regime}",
        ],
    }


def analyze_india_vix(vix: pd.DataFrame | None) -> dict[str, Any]:
    if vix is None or vix.empty:
        return {"regime": "unknown", "score": 50.0, "facts": ["India VIX unavailable"]}
    last = float(vix["Close"].iloc[-1])
    # Lower VIX → friendlier for swing entries.
    if last < 14:
        regime, score = "calm", 75.0
    elif last < 18:
        regime, score = "normal", 60.0
    elif last < 22:
        regime, score = "elevated", 40.0
    else:
        regime, score = "stress", 20.0
    return {
        "regime": regime,
        "score": score,
        "close": last,
        "facts": [f"India VIX={last:.2f} ({regime})"],
    }


def build_ta_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    close = df["Close"]
    last = float(close.iloc[-1])
    sma20 = _f(_sma(close, 20).iloc[-1])
    sma50 = _f(_sma(close, 50).iloc[-1])
    sma200 = _f(_sma(close, 200).iloc[-1]) if len(close) >= 200 else None
    rsi = _f(_rsi(close).iloc[-1])
    atr = _f(_atr(df).iloc[-1])
    return {
        "close": last,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi14": rsi,
        "atr14": atr,
        "facts": [
            f"TA close={last:.2f}",
            f"SMA20={sma20:.2f}" if sma20 else "SMA20=n/a",
            f"SMA50={sma50:.2f}" if sma50 else "SMA50=n/a",
            f"SMA200={sma200:.2f}" if sma200 else "SMA200=n/a",
            f"RSI14={rsi:.1f}" if rsi is not None else "RSI14=n/a",
            f"ATR14={atr:.2f}" if atr is not None else "ATR14=n/a",
        ],
    }


def connect_screener_results(
    screeners: list[ScreenerResult],
    *,
    ta: dict[str, Any],
    nifty: dict[str, Any],
    vix: dict[str, Any],
) -> dict[str, Any]:
    """Cross-link screener outputs into consensus, conflicts, and a trade plan."""
    usable = [s for s in screeners if s.signal != "unavailable"]
    if not usable:
        return {
            "consensus_score": 0.0,
            "consensus_signal": "unavailable",
            "agreeing": [],
            "conflicting_pairs": [],
            "gate": "blocked",
            "trade_plan": {
                "stance": "HOLD",
                "reason": "Skills pack screeners unavailable — no new risk.",
            },
            "links": [],
        }

    weight_sum = 0.0
    weighted = 0.0
    for s in usable:
        w = _SCREENER_WEIGHTS.get(s.name, 0.2)
        weighted += s.score * w
        weight_sum += w
    consensus = weighted / weight_sum if weight_sum else 0.0

    # Market regime gate softens bullish consensus.
    regime_adj = ((float(nifty.get("score", 50)) + float(vix.get("score", 50))) / 2.0 - 50.0) * 0.35
    consensus = max(0.0, min(100.0, consensus + regime_adj))

    agreeing = [s.name for s in usable if s.score >= 60]
    weak = [s.name for s in usable if s.score <= 40]

    conflicting_pairs: list[str] = []
    by_name = {s.name: s for s in usable}
    # Explicit cross-checks — this is the "connect each result with each other" layer.
    links: list[str] = []
    if "VCP" in by_name and "Volume Breakout" in by_name:
        vcp, vb = by_name["VCP"], by_name["Volume Breakout"]
        if vcp.score >= 60 and vb.score >= 60:
            links.append("VCP pivot + Volume Breakout agree → breakout-entry path.")
        elif vcp.score >= 60 and vb.score < 45:
            conflicting_pairs.append("VCP vs Volume Breakout")
            links.append("VCP forming but volume breakout absent → wait for confirmation.")
    if "PEAD" in by_name and "Volume Breakout" in by_name:
        pead, vb = by_name["PEAD"], by_name["Volume Breakout"]
        if "SIGNAL_READY" in (pead.summary or "") and vb.score >= 70:
            conflicting_pairs.append("PEAD vs Volume Breakout")
            links.append("PEAD wants red-week breakout entry while Volume Breakout already fired → prefer PEAD trigger, not chase.")
        elif pead.score >= 60 and vb.score >= 60:
            links.append("PEAD drift + Volume Breakout aligned → earnings continuation with volume.")
        elif pead.score >= 60 and vb.score < 50:
            links.append("PEAD active but volume quiet → wait for confirmation volume.")
    if "PEAD" in by_name and "Momentum" in by_name:
        pead, mom = by_name["PEAD"], by_name["Momentum"]
        if pead.score >= 60 and mom.score >= 60:
            links.append("PEAD + Momentum agree on post-earnings continuation.")
        elif pead.score >= 65 and mom.score <= 40:
            conflicting_pairs.append("PEAD vs Momentum")
            links.append("PEAD setup present but Momentum weak — treat as caution.")
    if "Relative Strength" in by_name and "Momentum" in by_name:
        rs, mom = by_name["Relative Strength"], by_name["Momentum"]
        if rs.score >= 60 and mom.score >= 60:
            links.append("Relative Strength + Momentum aligned vs Nifty.")
        elif abs(rs.score - mom.score) >= 30:
            conflicting_pairs.append("Relative Strength vs Momentum")
            links.append("RS/Momentum diverge — treat as caution unless TA overrides.")
    if "Momentum" in by_name and "VCP" in by_name:
        if by_name["Momentum"].score >= 60 and by_name["VCP"].score >= 55:
            links.append("Momentum supports VCP continuation thesis.")

    gate = "allow"
    if nifty.get("regime") == "risk-off" or vix.get("regime") == "stress":
        gate = "restrict"
        links.append("Nifty risk-off or India VIX stress → restrict new BUY size / prefer HOLD.")
    if len(agreeing) < 2:
        gate = "caution" if gate == "allow" else gate
        links.append("Fewer than 2 bullish screeners → caution.")

    close = float(ta.get("close") or 0.0)
    atr = float(ta.get("atr14") or (close * 0.02 if close else 0.0))
    pivot = by_name.get("VCP").levels.get("pivot") if "VCP" in by_name else None
    resistance = by_name.get("Volume Breakout").levels.get("resistance") if "Volume Breakout" in by_name else None
    entry = float(pivot or resistance or close)
    stop = close * 0.95 if close else entry * 0.95  # mandatory 5% trail distance reference
    risk = max(close - stop, atr, 1e-6) if close else 1.0
    target = close + 1.5 * risk if close else entry * 1.08
    rr = (target - close) / risk if close else 1.5

    if consensus >= 65 and gate == "allow" and len(agreeing) >= 2:
        stance = "BUY bias (skills)"
    elif consensus >= 55 and gate != "restrict":
        stance = "WATCH / staged entry"
    elif consensus <= 35 or gate == "restrict":
        stance = "HOLD / reduce risk"
    else:
        stance = "HOLD"

    trade_plan = {
        "stance": stance,
        "entry_zone": round(entry, 2),
        "stop_ref": round(stop, 2),
        "target_ref": round(target, 2),
        "risk_reward": round(rr, 2),
        "agreeing_screeners": agreeing,
        "weak_screeners": weak,
        "reason": (
            f"Consensus {consensus:.0f}/100; gate={gate}; "
            f"agreeing={agreeing or ['none']}; conflicts={conflicting_pairs or ['none']}."
        ),
    }

    return {
        "consensus_score": round(consensus, 1),
        "consensus_signal": _signal_from_score(consensus),
        "agreeing": agreeing,
        "weak": weak,
        "conflicting_pairs": conflicting_pairs,
        "gate": gate,
        "trade_plan": trade_plan,
        "links": links,
    }


def run_claude_skills_pack(ticker: str, *, period: str = "1y") -> dict[str, Any]:
    """Execute the full pack for one ticker. Network errors become unavailable blocks."""
    errors: list[str] = []
    stock = nifty = vix = None
    try:
        stock = load_ohlcv(ticker, period=period)
    except Exception as exc:
        errors.append(f"stock OHLCV: {exc}")
        logger.warning("Skills pack stock history failed for %s: %s", ticker, exc)
    try:
        nifty = load_ohlcv(NIFTY_SYMBOL, period=period)
    except Exception as exc:
        errors.append(f"Nifty: {exc}")
        logger.warning("Skills pack Nifty failed: %s", exc)
    try:
        vix = load_ohlcv(INDIA_VIX_SYMBOL, period="6mo")
    except Exception as exc:
        errors.append(f"India VIX: {exc}")
        logger.warning("Skills pack India VIX failed: %s", exc)

    if stock is None or stock.empty:
        return {
            "ticker": ticker,
            "ok": False,
            "errors": errors or ["no stock data"],
            "screeners": [],
            "connected": connect_screener_results([], ta={}, nifty={}, vix={}),
            "ta": {},
            "nifty": {"regime": "unknown", "facts": ["unavailable"]},
            "vix": {"regime": "unknown", "facts": ["unavailable"]},
        }

    screeners = [
        screen_vcp(stock),
        screen_pead(stock, ticker=ticker),
        screen_relative_strength(stock, nifty),
        screen_volume_breakout(stock),
        screen_momentum(stock),
    ]
    ta = build_ta_snapshot(stock)
    nifty_info = analyze_nifty_regime(nifty)
    vix_info = analyze_india_vix(vix)
    connected = connect_screener_results(screeners, ta=ta, nifty=nifty_info, vix=vix_info)
    return {
        "ticker": ticker,
        "ok": True,
        "errors": errors,
        "screeners": screeners,
        "connected": connected,
        "ta": ta,
        "nifty": nifty_info,
        "vix": vix_info,
    }


def format_claude_skills_pack_block(pack: dict[str, Any]) -> str:
    """Markdown-ish plain-text block for portfolio_context / agent observe."""
    lines: list[str] = [
        "=== CLAUDE SKILLS PACK (observe BEFORE any Buy/Sell/Hold signal) ===",
        "MANDATORY: read every screener, the connected consensus, Nifty/India VIX, and trade plan "
        "before arguing a stance. Do not ignore conflicts between screeners.",
    ]
    if not pack.get("ok"):
        lines.append("Skills pack unavailable this run — default to HOLD unless other evidence is overwhelming.")
        for err in pack.get("errors") or []:
            lines.append(f"- error: {err}")
        lines.append("=== END CLAUDE SKILLS PACK ===")
        return "\n".join(lines)

    lines.append(f"Ticker under screen: {pack.get('ticker')}")
    lines.append("")
    lines.append("--- 5 SCREENERS ---")
    for s in pack.get("screeners") or []:
        if isinstance(s, ScreenerResult):
            lines.append(f"[{s.name}] score={s.score:.0f}/100 signal={s.signal} — {s.summary}")
            for fact in s.facts:
                lines.append(f"  • {fact}")
            if s.supports:
                lines.append(f"  • supports: {', '.join(s.supports)}")
            if s.conflicts:
                lines.append(f"  • watch conflicts with: {', '.join(s.conflicts)}")
        else:
            lines.append(str(s))

    connected = pack.get("connected") or {}
    lines.append("")
    lines.append("--- CONNECTED CONSENSUS (screeners linked to each other) ---")
    lines.append(
        f"Consensus score={connected.get('consensus_score')}/100 "
        f"signal={connected.get('consensus_signal')} gate={connected.get('gate')}"
    )
    lines.append(f"Agreeing: {', '.join(connected.get('agreeing') or []) or 'none'}")
    lines.append(f"Weak: {', '.join(connected.get('weak') or []) or 'none'}")
    lines.append(f"Conflicts: {', '.join(connected.get('conflicting_pairs') or []) or 'none'}")
    for link in connected.get("links") or []:
        lines.append(f"  ↔ {link}")

    plan = connected.get("trade_plan") or {}
    lines.append("")
    lines.append("--- TA + NIFTY + INDIA VIX + TRADE PLAN ---")
    for fact in (pack.get("ta") or {}).get("facts") or []:
        lines.append(f"  • {fact}")
    for fact in (pack.get("nifty") or {}).get("facts") or []:
        lines.append(f"  • {fact}")
    for fact in (pack.get("vix") or {}).get("facts") or []:
        lines.append(f"  • {fact}")
    lines.append(
        f"Trade plan stance: {plan.get('stance')} | entry≈{plan.get('entry_zone')} "
        f"| stop_ref≈{plan.get('stop_ref')} | target_ref≈{plan.get('target_ref')} "
        f"| R:R≈{plan.get('risk_reward')}"
    )
    if plan.get("reason"):
        lines.append(f"  • {plan['reason']}")
    if pack.get("errors"):
        lines.append("Partial data warnings:")
        for err in pack["errors"]:
            lines.append(f"  • {err}")
    lines.append("=== END CLAUDE SKILLS PACK ===")
    return "\n".join(lines)


def build_claude_skills_context(ticker: str) -> str:
    """Convenience: run pack + format for injection into analysis context."""
    try:
        pack = run_claude_skills_pack(ticker)
        return format_claude_skills_pack_block(pack)
    except Exception as exc:
        logger.warning("build_claude_skills_context failed for %s: %s", ticker, exc)
        return (
            "=== CLAUDE SKILLS PACK (observe BEFORE any Buy/Sell/Hold signal) ===\n"
            f"Skills pack failed: {exc}\n"
            "Default to HOLD unless other evidence is overwhelming.\n"
            "=== END CLAUDE SKILLS PACK ==="
        )


def skills_observe_excerpt(portfolio_context: str, limit: int = 3500) -> str:
    """Extract the Claude Skills Pack section for analyst prompts."""
    ctx = (portfolio_context or "").strip()
    if not ctx:
        return (
            "No Claude Skills Pack context was supplied. "
            "Avoid strong directional claims without confirmation."
        )
    start = ctx.find("=== CLAUDE SKILLS PACK")
    if start < 0:
        return "Claude Skills Pack section missing from context — note that gap."
    end = ctx.find("=== END CLAUDE SKILLS PACK ===", start)
    block = ctx[start : end + len("=== END CLAUDE SKILLS PACK ===")] if end >= 0 else ctx[start:]
    if len(block) > limit:
        return block[:limit] + "\n... (skills pack truncated for prompt size)"
    return block


_SKILLS_OBSERVE_PREAMBLE = (
    "\nMANDATORY: OBSERVE the Claude Skills Pack excerpt below BEFORE any Buy/Sell lean. "
    "Cross-check against VCP / PEAD / Relative Strength / Volume Breakout / Momentum "
    "and the connected consensus / Nifty / India VIX gate.\n\n"
    "=== SKILLS PACK EXCERPT (observe first) ===\n"
)
