"""Short-lived in-process cache for optional social/macro fetchers.

Cron batch runs analyze many tickers in one process. Caching identical
Reddit/StockTwits requests within a few minutes avoids tripping public
per-IP rate limits during smoke tests and back-to-back ticker runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_CACHE: dict[tuple, tuple[float, T]] = {}
DEFAULT_TTL_SEC = 300.0


def cached_fetch(
    key: tuple,
    fetcher: Callable[[], T],
    ttl_sec: float = DEFAULT_TTL_SEC,
) -> T:
    now = time.time()
    hit = _CACHE.get(key)
    if hit is not None and now - hit[0] < ttl_sec:
        return hit[1]
    value = fetcher()
    _CACHE[key] = (now, value)
    return value
