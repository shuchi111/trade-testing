"""Derive AI confidence (0–100) from a complete trading analysis report."""

from __future__ import annotations

import os
import re
import time
from typing import Any

_CONFIDENCE_PATTERNS = [
    r"\b(?:ai\s+)?(?:confidence|probability|conviction)\s*[:=@-]?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%",
    r"\b([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(?:ai\s+)?(?:confidence|probability|conviction)\b",
]

_PCT_ONLY_RE = re.compile(r"^\s*(\d{1,3})(?:\.\d+)?\s*$")


def parse_explicit_confidence_pct(text: str) -> float | None:
    """Parse an explicit ``Confidence: 72%`` style value from report text."""
    if not text or not text.strip():
        return None
    for pattern in _CONFIDENCE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if 0 <= value <= 100:
            return value
    return None


def _clamp_pct(value: float) -> float | None:
    if not (0 <= value <= 100):
        return None
    return round(value, 2)


def _parse_llm_confidence_output(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    explicit = parse_explicit_confidence_pct(text)
    if explicit is not None:
        return explicit
    first_token = text.split()[0] if text.split() else ""
    match = _PCT_ONLY_RE.match(first_token)
    if match:
        return _clamp_pct(float(match.group(1)))
    number_match = re.search(r"\b(\d{1,3})(?:\.\d+)?\b", text)
    if number_match:
        return _clamp_pct(float(number_match.group(1)))
    return None


def extract_confidence_pct(
    full_report: str,
    *,
    decision: str = "",
    final_trade_decision: str = "",
    llm: Any | None = None,
) -> float | None:
    """Return 0–100 confidence from explicit fields or LLM synthesis of the full report."""
    combined = f"{decision or ''}\n{final_trade_decision or ''}\n{full_report or ''}"
    explicit = parse_explicit_confidence_pct(combined)
    if explicit is not None:
        return explicit

    report = (full_report or "").strip()
    if not report or llm is None:
        return None

    messages = [
        (
            "system",
            "You assess conviction in a trading recommendation from a multi-analyst report. "
            "Read the full report: analyst views, research debate, risk debate, and the final "
            "Portfolio Manager decision. "
            "Output ONLY one integer from 0 to 100 representing how confident you are that the "
            "final recommendation is well-supported by the evidence. "
            "0 = no conviction / contradictory evidence; 100 = very strong unanimous support. "
            "No words, no percent sign — just the number.",
        ),
        (
            "human",
            f"Final decision context:\n{decision or 'See report'}\n\n---\n\n{report[:120_000]}",
        ),
    ]
    attempts = int(os.getenv("CONFIDENCE_EXTRACT_MAX_ATTEMPTS", "3"))
    delay = float(os.getenv("CONFIDENCE_EXTRACT_RETRY_DELAY_SEC", "1.25"))

    for attempt in range(attempts):
        try:
            raw = llm.invoke(messages).content
            text = raw if isinstance(raw, str) else str(raw)
            parsed = _parse_llm_confidence_output(text)
            if parsed is not None:
                return parsed
            if attempt < attempts - 1:
                time.sleep(delay * (attempt + 1))
        except Exception:
            if attempt < attempts - 1:
                time.sleep(delay * (attempt + 1))
                continue
            break
    return None


def build_quick_llm_from_config(config: dict) -> Any:
    from tradingagents.llm_clients.factory import create_llm_client

    client = create_llm_client(
        config["llm_provider"],
        config["quick_think_llm"],
        base_url=config.get("backend_url"),
        api_key=config.get("api_key"),
    )
    return client.get_llm()
