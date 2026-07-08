import logging

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
from .fred import get_macro_data as get_fred_macro_data
from .polymarket import get_prediction_markets as get_polymarket_prediction_markets
from .alpha_vantage_common import AlphaVantageRateLimitError
from .config import get_config
from .errors import (
    NoMarketDataError,
    VendorConfigurationError,
    VendorDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)

logger = logging.getLogger(__name__)

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
    "get_macro_indicators": {"fred": get_fred_macro_data},
    "get_prediction_markets": {"polymarket": get_polymarket_prediction_markets},
}

TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "Daily OHLCV price history for a ticker.",
        "tools": ["get_stock_data"],
    },
    "technical_indicators": {
        "description": "Technical indicators (RSI, MACD, Bollinger, EMA/SMA, ATR, ...).",
        "tools": ["get_indicators"],
    },
    "fundamental_data": {
        "description": "Fundamentals: profile, balance sheet, cash flow, income statement.",
        "tools": ["get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"],
    },
    "news_data": {
        "description": "Company news, macro/global news, and insider transactions.",
        "tools": ["get_news", "get_global_news", "get_insider_transactions"],
    },
    "macro_data": {
        "description": "Macroeconomic indicators (rates, inflation, labor, growth).",
        "tools": ["get_macro_indicators"],
    },
    "prediction_markets": {
        "description": "Market-implied probabilities for forward-looking events.",
        "tools": ["get_prediction_markets"],
    },
}

OPTIONAL_CATEGORIES = {"macro_data", "prediction_markets"}

for _category, _info in TOOLS_CATEGORIES.items():
    _vendors = set()
    for _tool in _info["tools"]:
        _vendors.update(VENDOR_METHODS.get(_tool, {}).keys())
    _info["VENDOR_LIST"] = sorted(_vendors)
del _category, _info, _vendors, _tool


def available_vendors(category: str = None) -> dict:
    if category is not None:
        if category not in TOOLS_CATEGORIES:
            raise VendorConfigurationError(f"Unknown category '{category}'")
        return list(TOOLS_CATEGORIES[category]["VENDOR_LIST"])
    return {cat: list(info["VENDOR_LIST"]) for cat, info in TOOLS_CATEGORIES.items()}


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
    configured_vendors = [v.strip() for v in vendor_config.split(",") if v.strip()]

    if method not in VENDOR_METHODS:
        raise VendorConfigurationError(f"Method '{method}' not supported")
    if not configured_vendors:
        raise VendorConfigurationError(f"No vendors configured for '{method}'")

    last_no_data: NoMarketDataError | None = None
    first_error: Exception | None = None
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
        except VendorRateLimitError as exc:
            errors.append(f"{vendor}: rate limited ({exc})")
            continue
        except VendorNotConfiguredError as exc:
            logger.warning("Vendor %r not configured for %s: %s", vendor, method, exc)
            errors.append(f"{vendor}: not configured ({exc})")
            if first_error is None:
                first_error = exc
            continue
        except NoMarketDataError as exc:
            last_no_data = exc
            errors.append(f"{vendor}: no data ({exc})")
            continue
        except Exception as exc:
            logger.warning("Vendor %r failed for %s: %s", vendor, method, exc)
            errors.append(f"{vendor}: {exc}")
            if first_error is None:
                first_error = exc
            continue

    if last_no_data is not None:
        sym = last_no_data.symbol
        canonical = last_no_data.canonical
        resolved = "" if canonical == sym else f" (resolved to '{canonical}')"
        reason = f" ({last_no_data.detail})" if last_no_data.detail else ""
        return (
            f"NO_DATA_AVAILABLE: No usable market data for '{sym}'{resolved} from "
            f"any configured vendor{reason}. The symbol may be invalid, delisted, "
            f"not covered, or the vendor returned stale data. Do not estimate or "
            f"fabricate values — report that data is unavailable for this symbol."
        )

    if first_error is not None:
        if category in OPTIONAL_CATEGORIES:
            logger.warning("Optional %s unavailable for %s: %s", category, method, first_error)
            return (
                f"DATA_UNAVAILABLE: optional {category} could not be retrieved "
                f"({first_error}). Proceed without it; do not fabricate values."
            )
        raise VendorDataError(
            f"No configured vendor returned data for '{method}'. "
            f"Tried {', '.join(configured_vendors)}. Errors: {'; '.join(errors) or 'none'}"
        )

    raise VendorDataError(
        f"No configured vendor returned data for '{method}'. "
        f"Tried {', '.join(configured_vendors)}. Errors: {'; '.join(errors) or 'none'}"
    )
