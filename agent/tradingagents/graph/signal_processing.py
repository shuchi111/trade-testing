import os
import re
import time

from langchain_openai import ChatOpenAI

_ALLOWED = frozenset({"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"})
_TOKEN_RE = re.compile(r"\b(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)\b", re.IGNORECASE)


def _looks_like_gateway_error(text: str) -> bool:
    if not text:
        return True
    u = text.upper()
    return any(
        s in u
        for s in (
            "CONNECTION ERROR",
            "TIMEOUT",
            "NETWORK",
            "ECONNRESET",
            "502",
            "503",
            "529",
            "RATE LIMIT",
        )
    )


def _extract_token_from_text(text: str) -> str | None:
    """First valid rating token in LLM output."""
    if not text or _looks_like_gateway_error(text):
        return None
    m = _TOKEN_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


def _fallback_from_report(full_signal: str) -> str:
    """If the extractor LLM fails, derive a rating from the portfolio-manager report text."""
    if not full_signal or not full_signal.strip():
        return "HOLD"
    m = _TOKEN_RE.search(full_signal)
    if m:
        return m.group(1).upper()
    return "HOLD"


def coerce_decision_token(token: str | None, report_text: str) -> str:
    """
    Never return gateway error strings as the decision — fall back to regex on the full report.
    Used at the end of ``run_propagate`` as a safety net.
    """
    if token:
        t = str(token).strip()
        if t and not _looks_like_gateway_error(t):
            m = _TOKEN_RE.search(t)
            if m:
                return m.group(1).upper()
    return _fallback_from_report(report_text or "")


def is_transient_propagate_error(exc: BaseException) -> bool:
    """True if retrying the full LangGraph run might succeed (network / gateway blips)."""
    msg = str(exc).lower()
    needles = (
        "connection error",
        "connection reset",
        "timeout",
        "timed out",
        "network",
        "econnreset",
        "remote protocol",
        "502",
        "503",
        "504",
        "429",
        "529",
        "rate limit",
        "temporarily unavailable",
        "overloaded",
        "try again",
        "ssl",
        "broken pipe",
        "reset by peer",
    )
    return any(n in msg for n in needles)


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        messages = [
            (
                "system",
                "You are an efficient assistant that extracts the trading decision from analyst reports. "
                "Extract the rating as exactly one of: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL. "
                "Output only the single rating word, nothing else.",
            ),
            ("human", full_signal),
        ]
        attempts = int(os.getenv("SIGNAL_EXTRACT_MAX_ATTEMPTS", "4"))
        delay = float(os.getenv("SIGNAL_EXTRACT_RETRY_DELAY_SEC", "1.25"))

        for attempt in range(attempts):
            try:
                raw = self.quick_thinking_llm.invoke(messages).content
                text = raw if isinstance(raw, str) else str(raw)
                token = _extract_token_from_text(text.strip())
                if token and token in _ALLOWED:
                    return token
                # Non-retryable garbage / error-shaped body → try regex on report
                if attempt < attempts - 1 and (
                    not text.strip() or _looks_like_gateway_error(text)
                ):
                    time.sleep(delay * (attempt + 1))
                    continue
                break
            except Exception:
                if attempt < attempts - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
                break

        return _fallback_from_report(full_signal)
