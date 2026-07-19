import unittest
from unittest.mock import MagicMock

from tradingagents.graph.confidence_extraction import (
    extract_confidence_pct,
    parse_explicit_confidence_pct,
)


class TestParseExplicitConfidence(unittest.TestCase):
    def test_ai_confidence_line(self):
        self.assertEqual(parse_explicit_confidence_pct("AI confidence: 74%"), 74.0)

    def test_confidence_without_ai_prefix(self):
        self.assertEqual(parse_explicit_confidence_pct("Confidence: 92%"), 92.0)

    def test_reversed_percent_order(self):
        self.assertEqual(parse_explicit_confidence_pct("72% confidence"), 72.0)

    def test_out_of_range_returns_none(self):
        self.assertIsNone(parse_explicit_confidence_pct("Confidence: 150%"))

    def test_narrative_confidence_ignored(self):
        self.assertIsNone(parse_explicit_confidence_pct("institutional confidence returning"))


class TestExtractConfidencePct(unittest.TestCase):
    def test_prefers_explicit_over_llm(self):
        llm = MagicMock()
        value = extract_confidence_pct(
            "Portfolio Manager\nAI confidence: 81%",
            decision="BUY",
            llm=llm,
        )
        self.assertEqual(value, 81.0)
        llm.invoke.assert_not_called()

    def test_llm_synthesis_when_missing_explicit(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="67")
        value = extract_confidence_pct(
            "# Trading Analysis Report\nBull case strong, bear case weak.\n**Rating**: Buy",
            decision="BUY",
            llm=llm,
        )
        self.assertEqual(value, 67.0)
        llm.invoke.assert_called_once()

    def test_no_llm_returns_none_without_explicit(self):
        self.assertIsNone(
            extract_confidence_pct(
                "No structured confidence in this report.",
                decision="BUY",
                llm=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
