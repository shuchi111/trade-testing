"""Shared configuration for the VectorBT backtest engine."""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Resolve project roots so imports work from any cwd
AGENT_ROOT = Path(__file__).resolve().parent.parent
SWING_TRADER_ROOT = AGENT_ROOT.parent
os.chdir(AGENT_ROOT)
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from dotenv import load_dotenv

# Prefer swing-trading-dev/.env (canonical for this app). Optional .env.local
# only fills vars that .env does not already set.
load_dotenv(SWING_TRADER_ROOT / ".env", override=True)
load_dotenv(SWING_TRADER_ROOT / ".env.local", override=False)

# Supabase credentials (same vars as the Next.js app)
SUPABASE_URL = os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()

# Backtest defaults
DEFAULT_INIT_CASH = 100_000
DEFAULT_FEES = 0.001       # 0.1% brokerage
DEFAULT_SLIPPAGE = 0.002   # 0.2% slippage

# Timeframe: daily only — AI agent produces 1 signal per day
TIMEFRAME = "1d"
FREQ_LABEL = "1D"

# Long-horizon backtests: ~15 years of daily bars (Yahoo → market_daily_bars)
BACKTEST_YEARS = 15
BACKTEST_END = datetime.now().strftime("%Y-%m-%d")
# Use day-count (not replace(year=…)) so leap days never break the start date.
BACKTEST_START = (datetime.now() - timedelta(days=int(365.25 * BACKTEST_YEARS))).strftime(
    "%Y-%m-%d"
)

# All tickers — NSE India stocks + Crypto
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "WIPRO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "LT.NS",
    "MARUTI.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "ITC.NS", "SUNPHARMA.NS",
    "AXISBANK.NS", "NESTLEIND.NS", "KOTAKBANK.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "M&M.NS", "TITAN.NS", "NTPC.NS", "ULTRACEMCO.NS",
    "ASIANPAINT.NS", "LTTS.NS", "TECHM.NS", "CIPLA.NS",
    "BTC-USD", "ETH-USD",
]

# ═══════════════════════════════════════════════════════════════════
# Strategy definitions — what each strategy does and when it trades
# ═══════════════════════════════════════════════════════════════════

STRATEGY_DEFINITIONS = {
    "ai_agent": {
        "full_name": "AI Trading Agent",
        "description": (
            "Replays your stored AI decisions from the database (not a fresh LLM run). "
            "Prefers paper fills in ai_trade_executions; otherwise uses "
            "ai_recommendation_history + ai_recommendation_cache. "
            "0 trades means that ticker has no BUY/SELL rows stored yet — only HOLD or empty."
        ),
        "buy_trigger": "Stored BUY / OVERWEIGHT recommendation, or executed paper BUY",
        "sell_trigger": "Stored SELL / UNDERWEIGHT recommendation, or executed paper SELL",
        "best_for": "Swing trades lasting 2-6 weeks",
        "signal_source": "ai_trade_executions → else ai_recommendation_history + cache",
    },

    "buy_hold": {
        "full_name": "Buy & Hold (Baseline)",
        "description": (
            "Buys on the very first day of the backtest period and holds until the end. "
            "No selling, no signals, no thinking. This is the simplest possible strategy "
            "and serves as the baseline. If your AI agent or any technical strategy cannot "
            "beat buy & hold, it is not worth using — you would have made more money doing nothing."
        ),
        "buy_trigger": "Buys on day 1 of the backtest period",
        "sell_trigger": "Never sells — holds until the end",
        "best_for": "Measuring whether active trading adds any value at all",
        "signal_source": "None — buys once, never sells",
    },

    "sma_crossover": {
        "full_name": "Simple Moving Average (SMA) Crossover",
        "description": (
            "Calculates two moving averages of the closing price — a fast one (short window) "
            "and a slow one (long window). This is one of the oldest and most widely used "
            "trend-following strategies. It works well in trending markets but gives false "
            "signals in sideways/choppy markets."
        ),
        "buy_trigger": "Fast MA (e.g. 20-day) crosses ABOVE slow MA (e.g. 50-day) — uptrend starting",
        "sell_trigger": "Fast MA crosses BELOW slow MA — uptrend fading, downtrend beginning",
        "best_for": "Trending markets (sustained up or down moves)",
        "weakness": "Whipsawed in sideways markets — many small losses with no clear trend",
        "variants": [
            {"fast": 20, "slow": 50, "label": "Standard 20/50 — medium-term swing"},
            {"fast": 10, "slow": 30, "label": "Fast 10/30 — shorter-term, more trades, more noise"},
        ],
        "signal_source": "Price data only (no external data needed)",
    },

    "rsi": {
        "full_name": "Relative Strength Index (RSI)",
        "description": (
            "RSI measures how fast and how much the price has moved recently, on a scale of 0 to 100. "
            "Above 70 means the stock is overbought (rose too fast, likely to pull back). "
            "Below 30 means the stock is oversold (fell too fast, likely to bounce). "
            "This is a mean-reversion strategy — it bets that extreme moves will reverse. "
            "Works well in range-bound markets but can get crushed in strong trends."
        ),
        "buy_trigger": "RSI drops below 30 (oversold) — price fell too fast, expects a bounce",
        "sell_trigger": "RSI rises above 70 (overbought) — price rose too fast, expects a pullback",
        "best_for": "Range-bound or choppy markets where price bounces between support and resistance",
        "weakness": "In a strong trend, RSI stays overbought/oversold for a long time — signals are too early",
        "signal_source": "Price data only",
    },

    "bollinger": {
        "full_name": "Bollinger Bands",
        "description": (
            "Bollinger Bands draw a channel around the price using a moving average (middle band) "
            "and two standard deviation bands above and below it. The bands widen when price is volatile "
            "and narrow when it is calm. Like RSI, this is a mean-reversion strategy. "
            "It works best when price oscillates within a range."
        ),
        "buy_trigger": "Price touches or drops below the LOWER band — price is unusually cheap",
        "sell_trigger": "Price touches or exceeds the UPPER band — price is unusually expensive",
        "best_for": "Volatile markets that snap back to average after big moves",
        "weakness": "Sells too early in strong trends — the band expands with the trend",
        "signal_source": "Price data only",
    },

    "composite": {
        "full_name": "Composite (RSI + MACD + Bollinger + SMA)",
        "description": (
            "Combines four technical indicators into a single weighted score. "
            "Each indicator votes bullish (+1), bearish (-1), or neutral (0). "
            "BUY when the composite score crosses above the buy threshold; "
            "SELL when it crosses below the sell threshold."
        ),
        "buy_trigger": "Composite score crosses above buy threshold (default +0.5)",
        "sell_trigger": "Composite score crosses below sell threshold (default -0.5)",
        "best_for": "Confirmation layer for AI signals; reduces false positives",
        "signal_source": "Price data only",
    },

    "macd": {
        "full_name": "Moving Average Convergence Divergence (MACD)",
        "description": (
            "MACD takes the difference between a 12-day and 26-day exponential moving average (the MACD line), "
            "then smooths that difference with a 9-day average (the signal line). MACD is both a trend-following "
            "and momentum strategy. It catches trend changes earlier than SMA crossover but produces more "
            "signals, which means more trades and potentially more false positives."
        ),
        "buy_trigger": "MACD line crosses ABOVE signal line — momentum turning bullish",
        "sell_trigger": "MACD line crosses BELOW signal line — momentum turning bearish",
        "best_for": "Catching trend reversals early with momentum confirmation",
        "weakness": "Produces many signals — higher trade count means more fees and more small losses",
        "signal_source": "Price data only",
    },

    "ml_forecast": {
        "full_name": "LightGBM ML Forecast (Qlib-style)",
        "description": (
            "A gradient-boosted (LightGBM) machine-learning model trained on Qlib Alpha158-inspired factor "
            "features — momentum, volatility, mean-reversion, and volume ratios computed from pure OHLCV. "
            "It predicts the N-day forward return and goes long when the prediction exceeds a small positive "
            "threshold, flat/short when it falls below the negative threshold. Trained walk-forward on an "
            "expanding window, so the prediction at any bar only uses data strictly before it (no look-ahead). "
            "This is the quant ML baseline from Microsoft's Qlib; the whole point is to see whether your AI "
            "agent or the technical strategies can beat a plain ML model."
        ),
        "buy_trigger": "LightGBM predicts N-day forward return above +0.5% threshold — bullish forecast",
        "sell_trigger": "LightGBM predicts N-day forward return below -0.5% threshold — bearish forecast",
        "best_for": "A non-LLM systematic baseline that learns from engineered price features",
        "weakness": "Needs enough history to train (>= ~120 bars) and retraining to track regime drift",
        "signal_source": "Qlib Alpha158-style factor frame (agent/backtest/factors.py)",
    },

    "prophet_forecast": {
        "full_name": "Prophet Forecast",
        "description": (
            "Uses Facebook Prophet (or a lightweight seasonal-trend fallback) to forecast the next N days "
            "of price. Goes long when the predicted return is above a buy threshold and exits when the "
            "forecast turns negative. Retrains periodically so each signal only uses history available then."
        ),
        "buy_trigger": "Prophet N-day forecast return above +0.5%",
        "sell_trigger": "Prophet N-day forecast return below -0.5%",
        "best_for": "Trend + seasonality aware swing entries without hand-tuned indicators",
        "weakness": "Slower than pure technicals; can lag sudden regime breaks",
        "signal_source": "Prophet / seasonal-trend model on closing prices",
    },

    "ensemble": {
        "full_name": "Ensemble (SMA + RSI + Bollinger + MACD)",
        "description": (
            "Combines four classic technical strategies by majority vote. A trade opens when at least "
            "min_votes components are simultaneously long, and closes when votes fall back below the threshold. "
            "Reduces single-indicator noise — similar to an ensemble layer in multi-strategy research dashboards."
        ),
        "buy_trigger": "At least 2 of SMA / RSI / Bollinger / MACD are in a long state",
        "sell_trigger": "Fewer than 2 components remain long",
        "best_for": "Filtering weak single-indicator signals",
        "signal_source": "Votes from SMA crossover, RSI, Bollinger, MACD",
    },
}

# ═══════════════════════════════════════════════════════════════════
# How to read backtest results
# ═══════════════════════════════════════════════════════════════════
#
# RETURN — Total % gain or loss over the entire backtest period.
#   Positive = strategy made money. Negative = lost money.
#   This is the bottom line. If this is negative, nothing else matters.
#
# SHARPE RATIO — Risk-adjusted return. Measures return per unit of risk.
#   < 0.5 = poor (risk is not worth the return)
#   0.5-1.0 = decent
#   > 1.0 = good
#   > 2.0 = excellent
#   A strategy with +30% return and Sharpe 0.5 is worse than one with
#   +20% return and Sharpe 1.5, because the first one took much more risk.
#
# MAX DRAWDOWN — The worst peak-to-valley drop your portfolio experienced.
#   If you started with 100,000 and at some point your portfolio was worth
#   75,000, your max drawdown is -25%. This tells you the maximum pain you
#   would have had to sit through. A strategy with +50% return but -40%
#   drawdown means at some point you lost nearly half your money before
#   it recovered.
#
# WIN RATE — % of trades that were profitable (return > 0).
#   IMPORTANT: High win rate does NOT mean the strategy is profitable.
#   You can have 75% win rate but lose money if your losses are much bigger
#   than your wins. You can have 30% win rate but make money if your wins
#   are much bigger than your losses. Win rate alone is meaningless.
#
# EXPECTANCY — The average % you expect to make (or lose) per trade.
#   Formula: (win_rate × avg_win) + (loss_rate × avg_loss)
#   Positive = over many trades, the strategy makes money.
#   Negative = over many trades, the strategy loses money.
#   This is the single most important metric after total return.
#
# RISK:REWARD — How much bigger your average win is vs your average loss.
#   If avg win is +6% and avg loss is -2%, your R:R is 3.0.
#   This means every dollar you risk, you expect to win 3 dollars.
#   Higher is better. Below 1.0 means your losses are bigger than your wins
#   (you need a very high win rate to survive).
#
# PROFIT FACTOR — Total gross profit divided by total gross loss.
#   > 1.0 = profitable. < 1.0 = losing money.
#   1.5 = good. 2.0+ = excellent.
#   Similar to expectancy but as a ratio instead of a difference.
# ═══════════════════════════════════════════════════════════════════
