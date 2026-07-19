from .alpha_vantage_common import _make_api_request


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    return _make_api_request("OVERVIEW", {"symbol": ticker})


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _make_api_request("BALANCE_SHEET", {"symbol": ticker})


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _make_api_request("CASH_FLOW", {"symbol": ticker})


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
