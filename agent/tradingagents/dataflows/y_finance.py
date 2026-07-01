import os
from datetime import datetime
from typing import Annotated

from dateutil.relativedelta import relativedelta
import pandas as pd  # type: ignore[reportMissingImports]
from stockstats import wrap  # type: ignore[reportMissingImports]
import yfinance as yf  # type: ignore[reportMissingImports]
from .config import get_config
from .stockstats_utils import StockstatsUtils, _clean_dataframe, yf_retry, _cache_date_range


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    ticker = yf.Ticker(symbol.upper())
    data = yf_retry(lambda: ticker.history(start=start_date, end=end_date))

    if data.empty:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    csv_string = data.to_csv()
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    best_ind_params = {
        "close_50_sma": "50 SMA: Medium-term trend indicator.",
        "close_200_sma": "200 SMA: Long-term trend benchmark.",
        "close_10_ema": "10 EMA: Responsive short-term average.",
        "macd": "MACD: Momentum via differences of EMAs.",
        "macds": "MACD Signal: EMA smoothing of the MACD line.",
        "macdh": "MACD Histogram: Gap between MACD and its signal.",
        "rsi": "RSI: Momentum to flag overbought/oversold conditions.",
        "boll": "Bollinger Middle: 20 SMA basis for Bollinger Bands.",
        "boll_ub": "Bollinger Upper Band: 2 standard deviations above the middle.",
        "boll_lb": "Bollinger Lower Band: 2 standard deviations below the middle.",
        "atr": "ATR: Averages true range to measure volatility.",
        "vwma": "VWMA: Moving average weighted by volume.",
        "mfi": "MFI: Money Flow Index using price and volume.",
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)
        current_dt = curr_date_dt
        date_values = []

        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')
            indicator_value = indicator_data.get(date_str, "N/A: Not a trading day (weekend or holiday)")
            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)

        ind_string = "".join(f"{d}: {v}\n" for d, v in date_values)

    except Exception as e:
        print(f"Error getting bulk stockstats data: {e}")
        ind_string = ""
        curr_date_dt2 = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt2 >= before:
            indicator_value = get_stockstats_indicator(symbol, indicator, curr_date_dt2.strftime("%Y-%m-%d"))
            ind_string += f"{curr_date_dt2.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt2 = curr_date_dt2 - relativedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )


def _get_stock_stats_bulk(symbol, indicator, curr_date) -> dict:
    config = get_config()
    online = config["data_vendors"]["technical_indicators"] != "local"

    if not online:
        try:
            data = pd.read_csv(
                os.path.join(config.get("data_cache_dir", "data"), f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv"),
                on_bad_lines="skip",
            )
        except FileNotFoundError:
            raise Exception("Stockstats fail: Yahoo Finance data not fetched yet!")
    else:
        # Use week-stable date range: same cache file is reused all week,
        # preventing a fresh 15-year download on every daily request.
        start_date_str, end_date_str = _cache_date_range(years=15)

        os.makedirs(config["data_cache_dir"], exist_ok=True)

        data_file = os.path.join(config["data_cache_dir"], f"{symbol}-YFin-data-{start_date_str}-{end_date_str}.csv")

        if os.path.exists(data_file):
            data = pd.read_csv(data_file, on_bad_lines="skip")
        else:
            data = yf_retry(lambda: yf.download(symbol, start=start_date_str, end=end_date_str, multi_level_index=False, progress=False, auto_adjust=True))
            data = data.reset_index()
            data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df[indicator]

    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]
        result_dict[date_str] = "N/A" if pd.isna(indicator_value) else str(indicator_value)

    return result_dict


def get_stockstats_indicator(symbol, indicator, curr_date) -> str:
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")
    try:
        indicator_value = StockstatsUtils.get_stock_stats(symbol, indicator, curr_date)
    except Exception as e:
        print(f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}")
        return ""
    return str(indicator_value)


def get_fundamentals(ticker, curr_date=None):
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = yf_retry(lambda: ticker_obj.info)
        if not info:
            return f"No fundamentals data found for symbol '{ticker}'"

        fields = [
            ("Name", info.get("longName")), ("Sector", info.get("sector")), ("Industry", info.get("industry")),
            ("Market Cap", info.get("marketCap")), ("PE Ratio (TTM)", info.get("trailingPE")),
            ("Forward PE", info.get("forwardPE")), ("PEG Ratio", info.get("pegRatio")),
            ("Price to Book", info.get("priceToBook")), ("EPS (TTM)", info.get("trailingEps")),
            ("Forward EPS", info.get("forwardEps")), ("Dividend Yield", info.get("dividendYield")),
            ("Beta", info.get("beta")), ("52 Week High", info.get("fiftyTwoWeekHigh")),
            ("52 Week Low", info.get("fiftyTwoWeekLow")), ("50 Day Average", info.get("fiftyDayAverage")),
            ("200 Day Average", info.get("twoHundredDayAverage")), ("Revenue (TTM)", info.get("totalRevenue")),
            ("Gross Profit", info.get("grossProfits")), ("EBITDA", info.get("ebitda")),
            ("Net Income", info.get("netIncomeToCommon")), ("Profit Margin", info.get("profitMargins")),
            ("Operating Margin", info.get("operatingMargins")), ("Return on Equity", info.get("returnOnEquity")),
            ("Return on Assets", info.get("returnOnAssets")), ("Debt to Equity", info.get("debtToEquity")),
            ("Current Ratio", info.get("currentRatio")), ("Book Value", info.get("bookValue")),
            ("Free Cash Flow", info.get("freeCashflow")),
        ]

        lines = [f"{label}: {value}" for label, value in fields if value is not None]
        header = f"# Company Fundamentals for {ticker.upper()}\n# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + "\n".join(lines)
    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_balance_sheet(ticker, freq="quarterly", curr_date=None):
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: ticker_obj.quarterly_balance_sheet if freq.lower() == "quarterly" else ticker_obj.balance_sheet)
        if data.empty:
            return f"No balance sheet data found for symbol '{ticker}'"
        header = f"# Balance Sheet data for {ticker.upper()} ({freq})\n# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(ticker, freq="quarterly", curr_date=None):
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: ticker_obj.quarterly_cashflow if freq.lower() == "quarterly" else ticker_obj.cashflow)
        if data.empty:
            return f"No cash flow data found for symbol '{ticker}'"
        header = f"# Cash Flow data for {ticker.upper()} ({freq})\n# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(ticker, freq="quarterly", curr_date=None):
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: ticker_obj.quarterly_income_stmt if freq.lower() == "quarterly" else ticker_obj.income_stmt)
        if data.empty:
            return f"No income statement data found for symbol '{ticker}'"
        header = f"# Income Statement data for {ticker.upper()} ({freq})\n# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(ticker):
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = yf_retry(lambda: ticker_obj.insider_transactions)
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"
        header = f"# Insider Transactions data for {ticker.upper()}\n# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + data.to_csv()
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"
