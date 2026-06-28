from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .alpha_vantage_common import AlphaVantageRateLimitError
from .config import get_config
from .errors import VendorConfigurationError, VendorDataError

VENDOR_METHODS = {
    "get_stock_data": {"alpha_vantage": get_alpha_vantage_stock, "yfinance": get_YFin_data_online},
    "get_indicators": {"alpha_vantage": get_alpha_vantage_indicator, "yfinance": get_stock_stats_indicators_window},
    "get_fundamentals": {"alpha_vantage": get_alpha_vantage_fundamentals, "yfinance": get_yfinance_fundamentals},
    "get_balance_sheet": {"alpha_vantage": get_alpha_vantage_balance_sheet, "yfinance": get_yfinance_balance_sheet},
    "get_cashflow": {"alpha_vantage": get_alpha_vantage_cashflow, "yfinance": get_yfinance_cashflow},
    "get_income_statement": {"alpha_vantage": get_alpha_vantage_income_statement, "yfinance": get_yfinance_income_statement},
    "get_news": {"alpha_vantage": get_alpha_vantage_news, "yfinance": get_news_yfinance},
    "get_global_news": {"yfinance": get_global_news_yfinance, "alpha_vantage": get_alpha_vantage_global_news},
    "get_insider_transactions": {"alpha_vantage": get_alpha_vantage_insider_transactions, "yfinance": get_yfinance_insider_transactions},
}

TOOLS_CATEGORIES = {
    "core_stock_apis": {"tools": ["get_stock_data"]},
    "technical_indicators": {"tools": ["get_indicators"]},
    "fundamental_data": {"tools": ["get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"]},
    "news_data": {"tools": ["get_news", "get_global_news", "get_insider_transactions"]},
}


def get_category_for_method(method: str) -> str:
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


def get_vendor(category: str, method: str = None) -> str:
    config = get_config()
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]
    return config.get("data_vendors", {}).get(category, "yfinance")


def route_to_vendor(method: str, *args, **kwargs):
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    configured_vendors = [v.strip() for v in vendor_config.split(',') if v.strip()]

    if method not in VENDOR_METHODS:
        raise VendorConfigurationError(f"Method '{method}' not supported")
    if not configured_vendors:
        raise VendorConfigurationError(f"No vendors configured for '{method}'")

    errors = []
    for vendor in configured_vendors:
        if vendor not in VENDOR_METHODS[method]:
            raise VendorConfigurationError(
                f"Vendor '{vendor}' not supported for '{method}'. "
                f"Supported: {', '.join(VENDOR_METHODS[method].keys())}"
            )
        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl
        try:
            return impl_func(*args, **kwargs)
        except AlphaVantageRateLimitError as exc:
            errors.append(f"{vendor}: rate limited ({exc})")
            continue
        except Exception as exc:
            errors.append(f"{vendor}: {exc}")
            continue

    raise VendorDataError(
        f"No configured vendor returned data for '{method}'. "
        f"Tried {', '.join(configured_vendors)}. Errors: {'; '.join(errors) or 'none'}"
    )
