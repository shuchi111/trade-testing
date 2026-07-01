import os
import re
import time

from langchain_openai import ChatOpenAI

from tradingagents.agents.utils.swing_policy import SWING_SIGNAL_EXTRACT_SYSTEM

_ALLOWED = frozenset({"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"})
_TOKEN_RE = re.compile(r"\b(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)\b", re.IGNORECASE)
_RATING_LINE_RE = re.compile(
    r"Rating\s*:\s*(Buy|Overweight|Hold|Underweight|Sell)\b",
    re.IGNORECASE,
)


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


def _normalize_rating_word(word: str) -> str:
    return word.strip().upper()


def _rating_from_report(text: str) -> str | None:
    """Portfolio manager explicit ``Rating:`` line — authoritative for execution."""
    m = _RATING_LINE_RE.search(text or "")
    if not m:
        return None
    token = _normalize_rating_word(m.group(1))
    return token if token in _ALLOWED else None


def _fallback_from_report(full_signal: str) -> str:
    """Derive rating from portfolio-manager report (Rating line first, then first token)."""
    rated = _rating_from_report(full_signal)
    if rated:
        return rated
    if not full_signal or not full_signal.strip():
        return "HOLD"
    m = _TOKEN_RE.search(full_signal)
    if m:
        return m.group(1).upper()
    return "HOLD"


def coerce_decision_token(token: str | None, report_text: str) -> str:
    """
    Canonical rating for DB + executor. Prefers ``Rating:`` in the full report
    over a short extractor token (avoids BUY in prose overriding Hold).
    """
    rated = _rating_from_report(report_text or "")
    if rated:
        return rated
    if token:
        t = str(token).strip()
        if t and not _looks_like_gateway_error(t):
            m = _TOKEN_RE.search(t)
            if m:
                return m.group(1).upper()
    return _fallback_from_report(report_text or "")


def _extract_token_from_text(text: str) -> str | None:
    """First valid rating token in LLM output."""
    if not text or _looks_like_gateway_error(text):
        return None
    m = _TOKEN_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


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
        "1305",
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
        rated = _rating_from_report(full_signal)
        if rated:
            return rated

        messages = [
            (
                "system",
                SWING_SIGNAL_EXTRACT_SYSTEM,
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
                    return coerce_decision_token(token, full_signal)
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
