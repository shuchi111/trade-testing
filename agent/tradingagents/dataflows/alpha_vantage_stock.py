from datetime import datetime
from .alpha_vantage_common import _make_api_request, _filter_csv_by_date_range


def get_stock(symbol: str, start_date: str, end_date: str) -> str:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    days_from_today = (datetime.now() - start_dt).days
    outputsize = "compact" if days_from_today < 100 else "full"

    params = {"symbol": symbol, "outputsize": outputsize, "datatype": "csv"}
    response = _make_api_request("TIME_SERIES_DAILY_ADJUSTED", params)
    return _filter_csv_by_date_range(response, start_date, end_date)
