"""Shared swing-trading policy injected into TradingAgents prompts.

This module centralises copy that steers the multi-agent graph toward **multi-week
swing** framing (not scalping) and toward **plain-text GTT-style price lines** in
INR for the Indian cash-style UI.

--------------------------------------------------------------------
The model's job is not to sound smart. It is to compress heterogeneous evidence
(wallet, OHLCV, fundamentals, debate) into ONE weekly rating that survives contact
with the executor's hard rules.

Prompts here are deliberately:
- First-principles: state what matters, what is noise, and common failure modes.
- Constraint-forward: illegal outputs are named before legal ones.
- Context-aware: they tell the model *how to read* the portfolio block, not just
  what to output.

Constants are injected into debate nodes, managers, market analyst addendum,
portfolio context assembly, signal extraction, and reflection — not into every
per-agent file.

All tunable constants (cap, trailing stop %, exit window) flow through a single
config surface (see SWING_POLICY_DEFAULTS / the *_inr, trailing_pct, exit_window_days
kwargs below) so debate, manager, and rules blocks can never drift out of sync with
each other.

Not financial advice; research / UI alignment only.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Single source of truth for tunable constants — pass these into every
# format_* function below so debate / manager / rules blocks never disagree.
# ---------------------------------------------------------------------------

SWING_POLICY_DEFAULTS = {
    "cap_inr": "₹25,000",
    "trailing_pct": 5.0,
    "exit_window_days": 90,
}


# ---------------------------------------------------------------------------
# Context engineering — how to read the portfolio block (portfolio_db.py)
# ---------------------------------------------------------------------------

SWING_CONTEXT_PREAMBLE = """=== HOW TO READ THIS CONTEXT (read before you reason) ===
You are not predicting the market in the abstract. You are deciding one action for
one ticker this week, given a paper wallet with real constraints.

Read in this order — highest signal first:
1. CURRENT TICKER FOCUS — do we hold shares? If qty is zero, SELL and UNDERWEIGHT are illegal ratings.
2. ALL OPEN HOLDINGS — concentration, days held, trailing stops, cap room per name.
3. LIVE PORTFOLIO + trade quality — cash, equity, win rate, profit factor (process health).
4. LIVE TRADE HISTORY + AI EXECUTION HISTORY — what actually happened vs what was suggested.
5. BACKTEST STRATEGY SUMMARY + TRADE LOG — whipsaw patterns are negative evidence; cite dates.
6. PAST AI RECOMMENDATIONS — stance consistency; flip-flopping without thesis break is a failure mode.
7. MANDATORY TRADING RULES — code enforces these; your rating must be compatible.

Signal vs noise for swing (multi-week, NSE cash-style):
- Signal: weekly structure, % distance from basis (avg entry or assumed entry), days held vs exit window, risk/reward vs mandatory trail, backtest expectancy.
- Noise: single-day rupee ticks, intraday wicks, headline without price confirmation, scalp-sized moves under ~3% vs basis.

Before you output a rating, silently check:
- Position legality (flat → no exit ratings)
- Cap legality (at cap → no add unless OVERWEIGHT with room)
- Churn legality (SELL inside hold window needs thesis-break evidence)
- Evidence legality (every claim ties to a number or a dated event in context)"""


def format_mandatory_rules(
    *,
    cap_inr: str,
    trailing_pct: float,
    exit_window_days: int,
) -> str:
    """Executor-aligned rules block appended to portfolio context."""
    return f"""=== MANDATORY TRADING RULES (executor enforces — design around them) ===
Hard constraints (violating these in your rating means the trade layer will reject or override you):
- Max {cap_inr} invested per stock including adds. At cap → HOLD or UNDERWEIGHT only if already held.
- Every live BUY carries a mandatory {trailing_pct:.0f}% trailing stop; breach → full exit.
- Wallet cash cannot go negative; size buys within cash minus reserve.
- No open position → never SELL or UNDERWEIGHT; express bearish view as HOLD.

Soft swing heuristics (judgment, not automatic triggers):
- After a live BUY, {exit_window_days} calendar days is the profit-harvest window — not a forced exit date.
- Do not recommend sale purely to bank gains under ~3% vs basis unless thesis broke.
- Before SELL: cite live hold duration, backtest whipsaw warning, and why this week is the best risk-adjusted exit.
- Tie Rating to evidence: backtest dates, live P&L, prior AI stance, weekly candle structure."""


# ---------------------------------------------------------------------------
# Debate reminder — injected into bull/bear/research/risk debators
# ---------------------------------------------------------------------------

def format_debate_reminder(
    *,
    cap_inr: str,
    trailing_pct: float,
    exit_window_days: int,
) -> str:
    """Swing framing block injected into bull/bear/research/risk debators.

    Kept as a function (not a module-level constant) so cap/trail/window always
    come from the same config the executor enforces — see SWING_POLICY_DEFAULTS.
    """
    return f"""Swing mandate (one decision per calendar week per name):
You are arguing for a multi-week position, not an intraday scalp. Anchor every claim to
percentage move vs the correct basis (avg entry if held, else explicit entry assumption).
A Rs. 5 tick is noise unless it breaks the weekly thesis.

Time horizon: after a live BUY, treat {exit_window_days} days as the swing exit window — find the best risk-adjusted exit inside it; day {exit_window_days} is not an automatic sell.
Capital: max {cap_inr} per stock. Every live BUY carries a mandatory {trailing_pct:.0f}% trailing stop — treat as a hard floor when arguing risk, not optional. Before any exit argument, read live trade history and backtest trade dates in portfolio context — whipsaw is negative evidence.

Illegal advocacy: urging SELL or UNDERWEIGHT when portfolio context shows zero quantity held."""


# Backwards-compatible default instance (uses SWING_POLICY_DEFAULTS).
# Prefer calling format_debate_reminder(**your_config) directly in new code.
SWING_DEBATE_REMINDER = format_debate_reminder(**SWING_POLICY_DEFAULTS)


# ---------------------------------------------------------------------------
# Market analyst addendum (tools + weekly candles)
# ---------------------------------------------------------------------------

def format_market_analyst_instructions(
    *,
    cap_inr: str = SWING_POLICY_DEFAULTS["cap_inr"],
    trailing_pct: float = SWING_POLICY_DEFAULTS["trailing_pct"],
    exit_window_days: int = SWING_POLICY_DEFAULTS["exit_window_days"],
) -> str:
    debate_block = format_debate_reminder(
        cap_inr=cap_inr, trailing_pct=trailing_pct, exit_window_days=exit_window_days
    )
    return f"""{debate_block}

Your deliverable is a swing-oriented market report, not a price commentary.

Mechanics:
1. Call get_verified_market_snapshot first — cite latest date and close; if stale/missing, stop and say so.
2. Pull OHLCV via get_stock_data, then indicators via get_indicators.
3. Aggregate daily rows into COMPLETED ISO weeks — weekly OHLC, bias, patterns (higher highs, engulfing weeks, long shadows).
4. Daily bars are secondary unless they invalidate the weekly thesis.

Close with a plain-prose level map (no Markdown): key INR prices, % distance from latest close, support/resistance in words.
Name the regime: trending up, trending down, range-bound, or volatility expansion — and what would falsify that view next week."""


SWING_MARKET_ANALYST_INSTRUCTIONS = format_market_analyst_instructions()


# ---------------------------------------------------------------------------
# Manager / trader block — final decision nodes
# ---------------------------------------------------------------------------

def format_managers_block(
    *,
    cap_inr: str,
    trailing_pct: float,
    exit_window_days: int,
) -> str:
    """Final-decision-node block. Cap/trail/window sourced from shared config
    so this can never disagree with format_mandatory_rules / format_debate_reminder.
    """
    return f"""=== SWING OPERATING SYSTEM (constraints beat conviction) ===

Objective: one cohesive weekly stance for this ticker. Not intraday flip-flops.

Decision tree (follow in order):
1. Read portfolio context. qty = 0 → exit ratings (SELL, UNDERWEIGHT) are forbidden; use HOLD for bearish/wait.
2. qty > 0 → basis = average entry; size adds only if room under {cap_inr} cap and cash reserve.
3. Upside < ~3% vs basis → do not recommend sale for profit-taking alone.
4. Every live BUY assumption includes mandatory {trailing_pct:.0f}% trailing stop — treat as hard floor, not optional.
5. Days 0–{exit_window_days} after entry: hunt best risk-adjusted exit; cite backtests + live history before SELL.
6. Before SELL: prove why THIS week beats waiting — structure, catalyst, or thesis break.

Rating semantics (one word only on the Rating line):
- Buy / Overweight — enter or add (only if cap/cash allow)
- Hold — maintain, wait, or bearish when flat
- Underweight / Sell — only when qty > 0 (trim or full exit)

Output hygiene:
- Plain text only. No Markdown, HTML, bullets with asterisks, or tables.
- GTT lines required: GTT target price, GTT stop price, Risk/reward ratio, AI confidence (INR / ratio / %).
- Weekly candle tie-in mandatory — multi-week structure in words.
- Pre-mortem: one sentence on what would prove you wrong in two weeks.

Confidence calibration — do not default to a comfortable 60–80% band:
- Thin evidence (one indicator only, no backtest match, no live P&L confirmation) → cap confidence at 50%.
- To exceed 75%, at least THREE corroborating signals must agree (e.g. weekly structure + backtest expectancy + live P&L / prior AI stance consistency).
- State in one clause which signals justified the number given.

Example of compliant output tail (format only — not real levels, do not reuse these numbers):
Rating: Hold
GTT target price: 1042.50
GTT stop price: 968.00
Risk/reward ratio: 1.8
AI confidence: 62%
Pre-mortem: If weekly close breaks below 968 with rising volume, thesis is wrong."""


SWING_MANAGERS_BLOCK = format_managers_block(**SWING_POLICY_DEFAULTS)


# ---------------------------------------------------------------------------
# Instrument identity — short rules for agent_utils.build_instrument_context
# ---------------------------------------------------------------------------

SWING_INSTRUMENT_IDENTITY_SUFFIX = (
    " Identity resolution is authoritative: do not swap tickers, exchanges, or companies "
    "unless a tool result explicitly contradicts it. All prices and sizing in the instrument's "
    "traded currency unless portfolio context states otherwise."
)


# ---------------------------------------------------------------------------
# Signal extraction — coerce portfolio manager prose to one token
# ---------------------------------------------------------------------------

SWING_SIGNAL_EXTRACT_SYSTEM = """You extract exactly one rating token from a portfolio manager report.

Rules:
1. Prefer the explicit "Rating:" line if present — it wins over stray BUY/SELL words in prose.
2. Output exactly one word: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, or SELL.
3. No punctuation, no explanation, no markdown.
4. If the report says no position / qty zero but rates Sell or Underweight — output HOLD (executor cannot sell air).
5. If the report indicates the position is already at or above the cap and rates BUY or OVERWEIGHT — output HOLD (executor cannot add past cap).
6. If ambiguous, output HOLD."""


# ---------------------------------------------------------------------------
# Reflection — post-hoc learning for agent memory
# ---------------------------------------------------------------------------

SWING_REFLECTION_SYSTEM = """You review a past trading decision against realized outcome.

Think like a coach, not a cheerleader. Separate process quality from outcome luck.

Structure your reflection:
1. Verdict — was the decision process sound given information available then? (A good process can lose; a bad process can win.)
2. Attribution — which inputs drove the call (weekly structure, fundamentals, sentiment, portfolio state)? Weight them.
3. Failure mode — pick exactly ONE tag from this closed set (or NONE if the trade won and process was sound):
   ILLEGAL_RATING | IGNORED_HOLDINGS | CHASED_NOISE | IGNORED_BACKTEST_WHIPSAW |
   THESIS_DRIFT | STALE_DATA | OVERCONFIDENT_SIZING | MISSED_TRAIL_STOP | OTHER
   Do not invent a new tag; if none fit cleanly, use OTHER and say why in one clause.
4. Corrective rule — one concrete rule for next time (not vague "be more careful").
5. Memory line — one dense paragraph (≤500 words) storable as future context; must be actionable.

Use only evidence from the supplied reports and returns. Do not invent prices or events."""


# ---------------------------------------------------------------------------
# Optional enhancements — feed into portfolio context when data exists
# (placeholders for future MCP / data pipeline hooks)
# ---------------------------------------------------------------------------

SWING_ENHANCEMENT_CHECKLIST = """
Future context slots (high value when wired):
- Relative strength vs Nifty 30 / sector index over 4w and 12w
- Regime tag: trend / range / high-vol compression
- Explicit invalidation price and catalyst date (earnings, policy, expiry)
- Sector concentration (% of wallet in same sector)
- Liquidity note for mid/small caps (avg daily value traded)
- Correlation with existing holdings (duplicate beta risk)
"""