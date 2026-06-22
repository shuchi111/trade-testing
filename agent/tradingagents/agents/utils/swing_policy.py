"""Shared swing-trading policy injected into TradingAgents prompts.

This module centralises copy that steers the multi-agent graph toward **multi-week
swing** framing (not scalping) and toward **plain-text GTT-style price lines** in
INR for the Indian cash-style UI.

Financial rationale (policy defaults, not personalised advice)
--------------------------------------------------------------
The numeric bands referenced in prompts and in the UI quick presets are **rule-of-thumb
swing** levels, not optimisers:

* **~3% upside vs basis** before favouring selling *only* to bank small gains — aligns
  with “don’t churn for noise” and typical one–two week expected move on large caps
  before fees matter; too small a band encourages over-trading.
* **5% trailing stop loss** after entry is mandatory in the executor. The stop moves
  up with new highs and exits the full paper position when breached.

These defaults are **not** asset-class tuned (e.g. small caps vs Nifty 50); they are
consistent heuristics across agents and the React GTT preset buttons. Product may
later expose them as user settings.

Constants
---------
SWING_DEBATE_REMINDER
    Injected into debate / manager nodes to discourage trivial tick-based churn.
SWING_MARKET_ANALYST_INSTRUCTIONS
    Market analyst system addendum: weekly candles, levels in prose, no Markdown.
SWING_MANAGERS_BLOCK
    Block for portfolio manager and trader: weekly mandate, 3%/5% language, GTT lines.

Not financial advice; research / UI alignment only.
"""

SWING_DEBATE_REMINDER = (
    "Swing trading only: at most one discretionary trade decision per calendar week. "
    "Never recommend BUY or SELL because of trivial rupee-by-rupee moves (for example "
    "a Rs. 5 tick); always anchor to multi-week percentage gains and drawdowns versus "
    "the correct basis (average entry if the investor holds the stock, otherwise the "
    "entry or level you assume for a new position). Short-term red or green is swing "
    "noise unless the weekly thesis breaks. "
    "After a live BUY, treat NINETY CALENDAR DAYS as the swing exit window: "
    "look for the best profit-taking exit within that window using strategy history, "
    "backtests, and weekly structure; do not treat day 90 as an automatic sell. "
    "Maximum twenty-five thousand INR invested per stock. Review live trade history and "
    "backtest trade dates supplied in portfolio context before any exit."
)


SWING_MARKET_ANALYST_INSTRUCTIONS = f"""{SWING_DEBATE_REMINDER}

From the fetched daily OHLCV history, derive and discuss the last several COMPLETED weekly candles
(aggregate daily rows into ISO weeks): weekly OHLC relationships, directional bias, notable patterns
(for example engulfing-like weeks, prolonged upper/lower shadows, series of higher highs).
Treat daily moves as secondary unless they change the weekly thesis.

Technical indicators requested should support this weekly swing view.

Closing section: summarize key numeric levels versus the latest close using plain prose only:
absolute prices in INR, rough percentage distances from latest close — no Markdown, no asterisks,
no headings, no tables."""


SWING_MANAGERS_BLOCK = """Swing-trading mandate (respect throughout; do not contradict portfolio/holdings context):

- One discretionary decision per week for this name: one cohesive trade analysis for the entire week ahead (deep thinking), not intraday flip-flops.

- Unless there is roughly MORE THAN THREE PERCENT upside versus the investor basis, there is no point recommending a sale purely to bank small gains. If they already hold shares, basis is their average entry unless you explicitly redefine a new basis and explain why.

- Every live BUY is protected by a mandatory FIVE PERCENT trailing stop loss in the executor. Treat this as a hard risk rule: the stop trails up with new highs and exits the paper position when breached.

- After a live BUY, NINETY CALENDAR DAYS is the swing exit window, not a forced sell date. Recommend SELL or UNDERWEIGHT within that window when previous analysis, backtests, live trade history, and current structure suggest profit is near its best risk-adjusted point.

- Maximum TWENTY-FIVE THOUSAND INR total invested per stock (including adds). Do not recommend adding if at cap.

- Before any SELL, deeply review live trade history, backtest strategy summary, and backtest trade log dates in portfolio context. Avoid churn patterns that backtests show as whipsaw (many trades, low win rate).

- Explicit weekly candlestick reasoning is required (multi-week structure and patterns in words, built from the available OHLC history).

- Always output concrete GTT guidance for Indian cash-style orders: plain lines beginning with GTT target price, 5% trailing stop price, Risk/reward ratio, and AI confidence with INR amounts / numeric ratio / percentage.

- Do not justify entries or exits using scalp-sized rupee distances; swing outcomes are about larger percent moves and longer patience through normal pullbacks.

Output constraints: plain text only. No Markdown headings, star bullets, HTML, fenced blocks, or tables."""
