from .alpha_vantage_common import _make_api_request


def get_indicator(symbol: str, indicator: str, curr_date: str, look_back_days: int,
                  interval: str = "daily", time_period: int = 14, series_type: str = "close") -> str:
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    indicator_map = {
        "close_50_sma": ("SMA", "50"),
        "close_200_sma": ("SMA", "200"),
        "close_10_ema": ("EMA", "10"),
        "macd": ("MACD", None),
        "macds": ("MACD", None),
        "macdh": ("MACD", None),
        "rsi": ("RSI", str(time_period)),
        "boll": ("BBANDS", "20"),
        "boll_ub": ("BBANDS", "20"),
        "boll_lb": ("BBANDS", "20"),
        "atr": ("ATR", str(time_period)),
        "vwma": (None, None),
    }

    if indicator not in indicator_map:
        raise ValueError(f"Indicator {indicator} is not supported.")

    av_function, period = indicator_map[indicator]
    if av_function is None:
        return f"## VWMA for {symbol}:\n\nVWMA is not directly available from Alpha Vantage API."

    params = {"symbol": symbol, "interval": interval, "series_type": series_type, "datatype": "csv"}
    if period:
        params["time_period"] = period

    try:
        data = _make_api_request(av_function, params)
        return f"## {indicator.upper()} for {symbol}:\n\n{data}"
    except Exception as e:
        return f"Error retrieving {indicator} data: {str(e)}"
