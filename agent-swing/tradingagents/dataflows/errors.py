class VendorError(RuntimeError):
    """Base error for configured data-vendor failures."""


class VendorConfigurationError(VendorError):
    """Raised when a tool is routed to an unknown or unsupported vendor."""


class VendorDataError(VendorError):
    """Raised when a configured vendor cannot return usable real data."""


class StaleMarketDataError(VendorDataError):
    """Raised when market data is real but too old for a live recommendation."""


class NoMarketDataError(VendorError):
    """A vendor returned no usable rows for a symbol (empty result or stale data)."""

    def __init__(self, symbol: str, canonical: str | None = None, detail: str = ""):
        self.symbol = symbol
        self.canonical = canonical or symbol
        self.detail = detail
        msg = f"No market data for {symbol!r}"
        if canonical and canonical != symbol:
            msg += f" (queried as {canonical!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class VendorRateLimitError(VendorError):
    """A vendor throttled the request; the router skips to the next vendor."""


class VendorNotConfiguredError(VendorError, ValueError):
    """A vendor was selected but its API key/configuration is missing."""
