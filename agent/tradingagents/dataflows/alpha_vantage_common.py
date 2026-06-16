import os
import requests
import pandas as pd
import json
from datetime import datetime
from io import StringIO

API_BASE_URL = "https://www.alphavantage.co/query"


def get_api_key() -> str:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set.")
    return api_key


def format_datetime_for_api(date_input) -> str:
    if isinstance(date_input, str):
        if len(date_input) == 8 and "T" not in date_input:
            return date_input + "T0000"
        if len(date_input) == 10:
            return date_input.replace("-", "") + "T0000"
        return date_input
    if hasattr(date_input, "strftime"):
        return date_input.strftime("%Y%m%dT%H%M")
    return str(date_input)


def _make_api_request(function: str, params: dict) -> str:
    api_key = get_api_key()
    request_params = {"function": function, "apikey": api_key, **params}
    response = requests.get(API_BASE_URL, params=request_params)
    response.raise_for_status()
    return response.text


def _filter_csv_by_date_range(csv_data: str, start_date: str, end_date: str) -> str:
    try:
        df = pd.read_csv(StringIO(csv_data))
        if df.empty:
            return csv_data
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
        filtered = df[mask]
        return filtered.to_csv(index=False)
    except Exception:
        return csv_data


class AlphaVantageRateLimitError(Exception):
    pass
