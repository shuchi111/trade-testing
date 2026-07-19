"""Load price data from Yahoo Finance and AI recommendations from Supabase."""
import logging

import numpy as np
import vectorbt as vbt
import pandas as pd
from supabase import create_client

from .config import SUPABASE_URL, SUPABASE_KEY, BACKTEST_START, BACKTEST_END, TIMEFRAME

logger = logging.getLogger(__name__)


def _sb():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _filter_ticker(df: pd.DataFrame, ticker: str | None) -> pd.DataFrame:
    if df.empty or not ticker or "ticker" not in df.columns:
        return df
    want = ticker.strip().upper()
    return df[df["ticker"].astype(str).str.strip().str.upper() == want].copy()


def load_ai_recommendations(ticker: str | None = None) -> pd.DataFrame:
    """
    Read ALL rows from ai_recommendation_history (append-only table).
    Returns DataFrame with columns:
      ticker, trade_date, decision, bucket, reference_price, computed_at
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials missing — AI recommendation history empty")
        return pd.DataFrame()

    sb = _sb()
    try:
        query = sb.table("ai_recommendation_history").select(
            "ticker,trade_date,decision,final_trade_decision,bucket,reference_price,computed_at"
        )
        if ticker:
            query = query.ilike("ticker", ticker.strip())
        resp = query.order("trade_date", desc=False).execute()
    except Exception as exc:
        logger.warning("ai_recommendation_history query failed (%s); retry eq", exc)
        query = sb.table("ai_recommendation_history").select("*")
        if ticker:
            query = query.eq("ticker", ticker.strip().upper())
        resp = query.order("trade_date", desc=False).execute()

    rows = resp.data or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return _filter_ticker(df, ticker)


def load_ai_cache(ticker: str | None = None) -> pd.DataFrame:
    """
    Read recommendation rows from ai_recommendation_cache (all dates for ticker).
    Cron writes here every run — primary fallback when history is thin/empty.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials missing — AI recommendation cache empty")
        return pd.DataFrame()

    sb = _sb()
    # Do not select bucket — older schemas only have decision.
    cols = "ticker,trade_date,decision,reference_price,computed_at,id"
    try:
        query = sb.table("ai_recommendation_cache").select(cols)
        if ticker:
            query = query.ilike("ticker", ticker.strip())
        resp = query.order("trade_date", desc=False).execute()
    except Exception as exc:
        logger.warning("ai_recommendation_cache query failed (%s); retry select *", exc)
        query = sb.table("ai_recommendation_cache").select("*")
        if ticker:
            query = query.eq("ticker", ticker.strip().upper())
        resp = query.order("trade_date", desc=False).execute()

    rows = resp.data or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return _filter_ticker(df, ticker)


def load_ai_trade_executions(
    ticker: str | None = None,
    *,
    include_dry_run: bool = False,
) -> pd.DataFrame:
    """
    Paper fills from ai_trade_executions → recommendation-shaped frame.

    Only real BUY/SELL actions become signals (SKIP ignored).
    Prefer live runs (dry_run=false) unless include_dry_run=True.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials missing — AI trade executions empty")
        return pd.DataFrame()

    sb = _sb()
    try:
        query = sb.table("ai_trade_executions").select(
            "ticker,trade_date,decision,action_taken,quantity,price,pnl,dry_run,skip_reason"
        )
        if ticker:
            query = query.ilike("ticker", ticker.strip())
        if not include_dry_run:
            query = query.eq("dry_run", False)
        resp = query.order("trade_date", desc=False).execute()
    except Exception as exc:
        logger.warning("ai_trade_executions query failed: %s", exc)
        return pd.DataFrame()

    rows = resp.data or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = _filter_ticker(df, ticker)
    if df.empty:
        return df

    action = df["action_taken"].astype(str).str.strip().str.upper()
    df = df[action.isin(["BUY", "SELL"])].copy()
    if df.empty:
        return df

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["bucket"] = action.map({"BUY": "buy", "SELL": "sell"})
    df["decision"] = df["decision"].fillna(df["action_taken"])
    df["reference_price"] = df["price"]
    df["signal_source"] = "ai_trade_executions"
    return df.reset_index(drop=True)


def _download_yf_fields(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    fields: tuple[str, ...],
) -> pd.DataFrame:
    """Download selected Yahoo Finance fields for one ticker."""
    try:
        raw = vbt.YFData.download(
            ticker, start=start, end=end, interval=interval, missing_index="drop"
        )
    except Exception as e:
        logger.error("Yahoo Finance download failed for %s: %s", ticker, e)
        return pd.DataFrame()

    cols: dict[str, pd.Series] = {}
    for name in fields:
        try:
            series = raw.get(name)
        except Exception:
            continue
        if series is None:
            continue
        if isinstance(series, pd.DataFrame):
            series = series.squeeze()
        if isinstance(series, pd.Series) and not series.empty:
            cols[name] = series.astype(float)

    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df


def load_ohlcv(
    ticker: str,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    interval: str = TIMEFRAME,
) -> pd.DataFrame:
    """
    Load OHLCV for factor / ML / backtests.

    Prefer ``market_daily_bars`` (10–15y Yahoo cache). Sync from yfinance when
    the cache is missing or thin. Daily interval only for the DB path.
    """
    if interval == TIMEFRAME:
        try:
            from .price_history import ensure_history

            cached = ensure_history(ticker, start=start, end=end)
            if not cached.empty:
                logger.info(
                    "%s: loaded %d daily bars from market_daily_bars / yfinance (%s → %s)",
                    ticker,
                    len(cached),
                    str(cached.index.min().date()),
                    str(cached.index.max().date()),
                )
                return cached
        except Exception as exc:
            logger.warning("price_history path failed for %s: %s — falling back to YFData", ticker, exc)

    return _download_yf_fields(
        ticker, start, end, interval, ("Open", "High", "Low", "Close", "Volume")
    )


def load_volume_data(
    ticker: str,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    interval: str = TIMEFRAME,
) -> pd.Series | None:
    """Volume series aligned to close prices, or None when unavailable."""
    ohlcv = load_ohlcv(ticker, start, end, interval)
    if ohlcv.empty or "Volume" not in ohlcv.columns:
        return None
    vol = ohlcv["Volume"]
    return vol if vol.notna().any() else None


def load_price_data(
    ticker: str,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    interval: str = TIMEFRAME,
    min_periods: int = 50,
) -> pd.Series:
    """
    Historical close prices for backtests (DB cache → yfinance → YFData).

    Validates data quality:
      - Sufficient length for strategy parameters
      - No NaN/Inf values (forward/backward filled if found)
      - Logs warnings for data issues
    """
    ohlcv = load_ohlcv(ticker, start, end, interval)
    if ohlcv.empty or "Close" not in ohlcv.columns:
        logger.warning("No price data returned for %s", ticker)
        return pd.Series(dtype=float)

    price = ohlcv["Close"].astype(float)
    price = price.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Check sufficient data length
    if len(price) < min_periods:
        logger.warning(
            "%s: only %d data points (need %d for most strategies)",
            ticker, len(price), min_periods,
        )

    return price


def load_multi_price(
    tickers: list[str],
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    interval: str = TIMEFRAME,
) -> pd.DataFrame:
    """Download prices for multiple tickers at once."""
    price_df = (
        vbt.YFData.download(tickers, start=start, end=end, interval=interval, missing_index="drop")
        .get("Close")
    )
    return price_df
