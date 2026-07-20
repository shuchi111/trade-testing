"""Holdings-aware decision helpers and full-report context blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from tradingagents.agents.utils.claude_skills_pack import skills_observe_excerpt
from tradingagents.graph.reporting import agent_report_sections

_SELL_TOKENS = frozenset({"SELL", "UNDERWEIGHT"})
_BUY_TOKENS = frozenset({"BUY", "OVERWEIGHT"})
_HOLD_QTY_RE = re.compile(
    r"Hold:\s*([0-9]+(?:\.[0-9]+)?)\s+([A-Z0-9.\-^=]+)",
    re.IGNORECASE,
)
_NO_POSITION_RE = re.compile(r"No open position in\s+([A-Z0-9.\-^=]+)", re.IGNORECASE)


@dataclass(frozen=True)
class TickerHoldingStatus:
    ticker: str
    is_holding: bool
    quantity: float
    summary: str


def parse_ticker_holding(portfolio_context: str, ticker: str) -> TickerHoldingStatus:
    """Parse holding status for *ticker* from portfolio context text."""
    ctx = (portfolio_context or "").strip()
    sym = (ticker or "").strip().upper()
    if not ctx:
        return TickerHoldingStatus(
            ticker=sym,
            is_holding=False,
            quantity=0.0,
            summary=f"No portfolio context supplied — assume flat in {sym or 'this ticker'}.",
        )

    focus_start = ctx.find("=== CURRENT TICKER FOCUS ===")
    focus_block = ctx[focus_start:] if focus_start >= 0 else ctx
    focus_end = focus_block.find("\n=== ", 1)
    if focus_end >= 0:
        focus_block = focus_block[:focus_end]

    if _NO_POSITION_RE.search(focus_block):
        return TickerHoldingStatus(
            ticker=sym,
            is_holding=False,
            quantity=0.0,
            summary=f"Flat — no open position in {sym}. SELL and UNDERWEIGHT are invalid.",
        )

    m = _HOLD_QTY_RE.search(focus_block)
    if m:
        qty = float(m.group(1))
        found = m.group(2).strip().upper()
        if sym and found and sym not in found and found not in sym:
            pass
        if qty > 0:
            line = next(
                (ln.strip() for ln in focus_block.splitlines() if ln.strip().startswith("Hold:")),
                f"Hold: {qty} {sym}",
            )
            return TickerHoldingStatus(
                ticker=sym,
                is_holding=True,
                quantity=qty,
                summary=line,
            )

    return TickerHoldingStatus(
        ticker=sym,
        is_holding=False,
        quantity=0.0,
        summary=f"Could not verify a position in {sym} — treat as flat; do not SELL.",
    )


def build_prior_agents_digest(final_state: Mapping[str, Any]) -> str:
    """Compact digest of every agent output that precedes the Portfolio Manager."""
    parts: list[str] = [
        "=== PRIOR AGENT OUTPUTS (read ALL before Rating) ===",
        "",
    ]
    for section_id, title, body in agent_report_sections(final_state):
        if section_id == "portfolio_manager":
            continue
        text = (body or "").strip()
        parts.append(f"--- {title} ---")
        parts.append(text if text else "(No report generated.)")
        parts.append("")
    parts.append("=== END PRIOR AGENT OUTPUTS ===")
    return "\n".join(parts).strip()


def build_holdings_rules_block(holding: TickerHoldingStatus) -> str:
    """Hard constraints the Portfolio Manager must obey."""
    lines = [
        "=== HOLDINGS CONSTRAINTS (binding — override invalid agent lean) ===",
        holding.summary,
    ]
    if not holding.is_holding:
        lines.extend(
            [
                f"Quantity held in {holding.ticker}: 0",
                "INVALID ratings when flat: Sell, Underweight (nothing to exit or trim).",
                "Valid ratings when flat: Hold (wait), Buy, Overweight (new entry only if cap/cash allow).",
                "If bear case is strong but flat, use Hold — not Sell.",
            ]
        )
    else:
        lines.extend(
            [
                f"Quantity held in {holding.ticker}: {holding.quantity:g}",
                "INVALID rating when already holding without cap room: Buy (use Overweight only if adding is justified).",
                "Valid exit ratings: Sell, Underweight, Hold.",
            ]
        )
    lines.append("=== END HOLDINGS CONSTRAINTS ===")
    return "\n".join(lines)


def enforce_holdings_decision(
    decision: str,
    portfolio_context: str,
    ticker: str,
) -> tuple[str, str | None]:
    """Clamp impossible ratings based on verified holdings."""
    token = (decision or "").strip().upper()
    holding = parse_ticker_holding(portfolio_context, ticker)

    if not holding.is_holding and token in _SELL_TOKENS:
        return (
            "HOLD",
            (
                f"Decision guard: PM rated {token} but {holding.ticker} is flat "
                f"({holding.summary}). Clamped to HOLD."
            ),
        )

    if holding.is_holding and token == "BUY":
        return (
            "HOLD",
            (
                f"Decision guard: PM rated BUY but {holding.ticker} is already held "
                f"(qty {holding.quantity:g}). Use OVERWEIGHT to add; clamped to HOLD."
            ),
        )

    return token if token else "HOLD", None


def build_ground_truth_report_section(
    portfolio_context: str,
    ticker: str,
    *,
    guard_note: str | None = None,
) -> str:
    """Markdown block for full_report section 0."""
    ctx = (portfolio_context or "").strip()
    holding = parse_ticker_holding(ctx, ticker)
    skills = skills_observe_excerpt(ctx) if ctx else "No Claude Skills Pack context."

    parts: list[str] = [
        "## 0. Ground Truth Context (Database, Holdings, Screeners)",
        "",
        "### Holdings Status",
        "",
        holding.summary,
        f"- **Is holding:** {'Yes' if holding.is_holding else 'No'}",
        f"- **Quantity:** {holding.quantity:g}",
        "",
        "### Claude Skills Pack & Screeners",
        "",
        skills,
        "",
        "### Full Portfolio & History Context",
        "",
        ctx if ctx else "_(No portfolio context was supplied this run.)_",
    ]
    if guard_note:
        parts.extend(["", "### Decision Guard", "", guard_note])
    return "\n".join(parts).strip()
