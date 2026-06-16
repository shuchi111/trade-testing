import time
import logging
from datetime import timedelta

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config

logger = logging.getLogger(__name__)


def _cache_date_range(years: int = 15) -> tuple[str, str]:
    """Return a stable (start_date, end_date) pair for the historical data cache.

    Parameters
    ----------
    years : int
        How many years of history to cover (default 15).

    Returns
    -------
    tuple[str, str]
        ``(start_date_str, end_date_str)`` both in ``YYYY-MM-DD`` format.

    Notes
    -----
    End-date is rounded to the **Monday of the current ISO week** so the cache
    filename stays identical for the whole week.  Without this, a filename that
    embeds today's date changes daily, causing a fresh 15-year download every
    single day instead of at most once per week.
    """
    today = pd.Timestamp.today().normalize()
    # Snap end to the Monday of the current week (day=0 is Monday)
    week_monday = today - timedelta(days=today.dayofweek)
    end_date = week_monday
    start_date = end_date - pd.DateOffset(years=years)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats."""
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[str, "quantitative indicator based off of the stock data"],
        curr_date: Annotated[str, "curr date for retrieving stock price data, YYYY-mm-dd"],
    ):
        config = get_config()

        curr_date_dt = pd.to_datetime(curr_date)
        # Use week-stable date range so the cache file is reused the whole week
        start_date_str, end_date_str = _cache_date_range(years=15)

        os.makedirs(config["data_cache_dir"], exist_ok=True)

        data_file = os.path.join(
            config["data_cache_dir"],
            f"{symbol}-YFin-data-{start_date_str}-{end_date_str}.csv",
        )

        if os.path.exists(data_file):
            data = pd.read_csv(data_file, on_bad_lines="skip")
        else:
            data = yf_retry(lambda: yf.download(
                symbol,
                start=start_date_str,
                end=end_date_str,
                multi_level_index=False,
                progress=False,
                auto_adjust=True,
            ))
            data = data.reset_index()
            data.to_csv(data_file, index=False)

        data = _clean_dataframe(data)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = curr_date_dt.strftime("%Y-%m-%d")

        df[indicator]
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            return matching_rows[indicator].values[0]
        return "N/A: Not a trading day (weekend or holiday)"
