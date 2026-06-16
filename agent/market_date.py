"""IST market trade date (Mon–Fri; Sat/Sun → Friday). No NSE holiday calendar."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def ist_today() -> date:
    return datetime.now(IST).date()


def adjust_to_last_trading_day(d: date) -> date:
    offset = {5: 1, 6: 2}.get(d.weekday(), 0)
    if offset:
        return d - timedelta(days=offset)
    return d


def market_trade_date() -> str:
    return adjust_to_last_trading_day(ist_today()).strftime("%Y-%m-%d")
