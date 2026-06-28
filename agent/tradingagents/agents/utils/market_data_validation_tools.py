from typing import Annotated

from tradingagents.dataflows.market_data_validator import (
    format_market_snapshot,
    verified_market_snapshot,
)


def get_verified_market_snapshot(
    ticker: Annotated[str, "Ticker symbol, preserving exchange suffix"],
    trade_date: Annotated[str, "Analysis date in YYYY-MM-DD format"] = "",
) -> str:
    """Return a deterministic real-data snapshot for the market analyst."""
    snapshot = verified_market_snapshot(ticker, trade_date or None)
    return format_market_snapshot(snapshot)
