from .alpha_vantage_common import _make_api_request, format_datetime_for_api


def get_news(ticker, start_date, end_date):
    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
    }
    return _make_api_request("NEWS_SENTIMENT", params)


def get_global_news(curr_date, look_back_days: int = 7, limit: int = 50):
    from datetime import datetime, timedelta
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (curr_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date),
        "limit": str(limit),
    }
    return _make_api_request("NEWS_SENTIMENT", params)


def get_insider_transactions(symbol: str):
    return _make_api_request("INSIDER_TRANSACTIONS", {"symbol": symbol})
