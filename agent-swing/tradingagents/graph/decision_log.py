"""Persistent, append-only markdown decision log.

This is an additive port of upstream's ``TradingMemoryLog``: a durable,
human-readable record of every propagation's final decision + reflection, written
to a single markdown file per ticker so memory survives process restarts.

Activation is **opt-in**: the log is only created when ``config["decision_log_dir"]``
is set (env ``TRADINGAGENTS_DECISION_LOG_DIR``). When unset (the default), every
method is a no-op, so existing behaviour is unchanged.

The JSON per-run logs under ``eval_results/<ticker>/...`` produced by
``TradingAgentsGraph._log_state`` are unaffected; this module adds a *separate*,
consolidated, markdown view.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _safe_component(value: Any, max_len: int = 1600) -> str:
    """Coerce a state field to a trimmed plain-text block."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + " …(truncated)"
    return text


class TradingMemoryLog:
    """Append-only markdown decision log keyed by ticker.

    Parameters
    ----------
    log_dir : str or Path
        Directory where ``trading_memory.md`` is written. Created if missing.
    """

    FILENAME = "trading_memory.md"

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Read-only filesystem / no permission → degrade to no-op rather than crash.
            self.log_dir = None

    @property
    def path(self) -> Optional[Path]:
        return self.log_dir / self.FILENAME if self.log_dir else None

    def log_decision(
        self,
        ticker: str,
        trade_date: str,
        decision: str,
        *,
        rating: Optional[str] = None,
        market_report: str = "",
        sentiment_report: str = "",
        news_report: str = "",
        fundamentals_report: str = "",
        investment_plan: str = "",
        final_trade_decision: str = "",
        reflection: Optional[str] = None,
        asset_type: str = "equity",
    ) -> None:
        """Append one decision record to ``trading_memory.md``.

        All inputs are coerced and length-capped; a malformed field never aborts the
        run. Failures to write are swallowed (the log is best-effort).
        """
        if self.path is None:
            return

        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
        header = f"# {ticker} — {trade_date}"
        rating_line = f"**Rating:** {rating}" if rating else ""

        block = f"""
{header}
- Logged (UTC): {ts}
- Asset type: {asset_type}
{rating_line}

## Final decision
{_safe_component(final_trade_decision or decision)}

## Investment plan
{_safe_component(investment_plan)}

### Market
{_safe_component(market_report)}

### Sentiment / social
{_safe_component(sentiment_report)}

### News
{_safe_component(news_report)}

### Fundamentals
{_safe_component(fundamentals_report)}
"""
        if reflection:
            block += f"""
## Reflection (returns/losses feedback)
{_safe_component(reflection)}
"""
        block += "\n---\n"

        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(block)
        except OSError:
            pass
