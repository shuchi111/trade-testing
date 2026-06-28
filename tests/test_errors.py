from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tradingagents.dataflows.errors import (
    StaleMarketDataError,
    VendorConfigurationError,
    VendorDataError,
    VendorError,
)


class ErrorHierarchyTests(unittest.TestCase):
    def test_vendor_error_hierarchy(self):
        self.assertTrue(issubclass(VendorConfigurationError, VendorError))
        self.assertTrue(issubclass(VendorDataError, VendorError))
        self.assertTrue(issubclass(StaleMarketDataError, VendorDataError))


if __name__ == "__main__":
    unittest.main()
