class VendorError(RuntimeError):
    """Base error for configured data-vendor failures."""


class VendorConfigurationError(VendorError):
    """Raised when a tool is routed to an unknown or unsupported vendor."""


class VendorDataError(VendorError):
    """Raised when a configured vendor cannot return usable real data."""


class StaleMarketDataError(VendorDataError):
    """Raised when market data is real but too old for a live recommendation."""
