"""
VectorBT Backtest Runner — orchestrates all strategies for one or more tickers.

Usage:
  python -m backtest.runner --ticker TCS.NS
  python -m backtest.runner --all
  python -m backtest.runner --ticker TCS.NS --start 2023-01-01 --end 2025-06-01
  python -m backtest.runner --ticker BTC-USD --no-store
"""
from __future__ import annotations

import argparse
import logging
import math
import traceback

import pandas as pd
import vectorbt as vbt

from .config import (
    DEFAULT_FEES,
    DEFAULT_INIT_CASH,
    DEFAULT_SLIPPAGE,
    BACKTEST_END,
    BACKTEST_START,
    STRATEGY_DEFINITIONS,
    TICKERS,
)
from .data_loader import load_price_data, load_volume_data
from .db_store import store_backtest_result, store_trade_logs
from .signal_builder import compute_signal_accuracy, compute_ic, compute_ic_from_predictions
from .strategies.ai_agent import AiAgentStrategy
from .strategies.buy_hold import BuyHoldStrategy
from .strategies.bollinger import BollingerStrategy
from .strategies.composite import CompositeStrategy
from .strategies.ensemble import EnsembleStrategy
from .strategies.macd import MacdStrategy
from .strategies.ml_forecast import MlForecastStrategy
from .strategies.prophet_forecast import ProphetForecastStrategy
from .strategies.rsi import RsiStrategy
from .strategies.sma_crossover import SmaCrossoverStrategy

logger = logging.getLogger(__name__)


def extract_metrics(pf: vbt.Portfolio) -> dict:
    """Extract all performance metrics from a VectorBT Portfolio object."""
    stats = pf.stats()
    trades = pf.trades.records_readable

    winning = trades[trades["Return"] > 0] if "Return" in trades.columns else pd.DataFrame()
    losing = trades[trades["Return"] <= 0] if "Return" in trades.columns else pd.DataFrame()

    def _safe_float(val, fallback=0.0):
        if val is None:
            return fallback
        try:
            f = float(val)
        except (ValueError, TypeError):
            return fallback
        if pd.isna(f) or math.isinf(f):
            return fallback
        return f

    # ── Win/Loss definition ─────────────────────────────────────
    # WIN  = trade return > 0  (made money after fees + slippage)
    # LOSS = trade return <= 0 (lost money OR broke even)
    #
    # Key insight:
    #   High win rate + negative return = wins are small, losses are big
    #   Low win rate + positive return  = wins are big, losses are small
    #
    # What matters: EXPECTANCY = (win_rate * avg_win) - (loss_rate * |avg_loss|)
    #   If expectancy > 0 → strategy is profitable over time
    # ─────────────────────────────────────────────────────────────
    win_rate = _safe_float(stats.get("Win Rate [%]", 0))
    loss_rate = 100.0 - win_rate
    avg_win = _safe_float(winning["Return"].mean() * 100) if len(winning) > 0 else 0
    avg_loss = _safe_float(losing["Return"].mean() * 100) if len(losing) > 0 else 0

    # Expectancy: avg $ expected per trade (in % terms)
    # Positive = strategy makes money over time, regardless of win rate
    expectancy = (win_rate / 100 * avg_win) + (loss_rate / 100 * avg_loss)

    # Risk-Reward Ratio: how much you win vs how much you lose per trade
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else None

    return {
        "total_return_pct": _safe_float(pf.total_return() * 100),
        "cagr_pct":         _safe_float(stats.get("Annualized Return [%]", 0)),
        "sharpe_ratio":     _safe_float(stats.get("Sharpe Ratio", 0)),
        "sortino_ratio":    _safe_float(stats.get("Sortino Ratio", 0)),
        "max_drawdown_pct": _safe_float(abs(pf.max_drawdown()) * 100),
        "calmar_ratio":     _safe_float(stats.get("Calmar Ratio", 0)),
        "win_rate_pct":     win_rate,
        "profit_factor":    _safe_float(stats.get("Profit Factor", 0)),
        "total_trades":     int(_safe_float(stats.get("Total Trades", 0))),
        "winning_trades":   len(winning),
        "losing_trades":    len(losing),
        "avg_win_pct":      avg_win,
        "avg_loss_pct":     avg_loss,
        "avg_holding_days": _safe_float(stats.get("Avg Holding Period", 0)),
        "final_value":      _safe_float(pf.value().iloc[-1]),
        "expectancy_pct":   _safe_float(expectancy),
        "risk_reward":      _safe_float(risk_reward) if risk_reward is not None else None,
    }


def extract_trade_logs(pf: vbt.Portfolio, ticker: str, strategy_name: str,
                       entry_reasons: pd.Series = None, exit_reasons: pd.Series = None,
                       entry_values: pd.Series = None, exit_values: pd.Series = None) -> list[dict]:
    """Extract individual trade records with trigger reasons for bt_trade_log table."""
    trades = pf.trades.records_readable
    logs = []
    for _, t in trades.iterrows():
        entry_ts = t.get("Entry Timestamp")
        exit_ts = t.get("Exit Timestamp")
        entry_date = entry_ts.date() if entry_ts is not None else None
        exit_date = exit_ts.date() if exit_ts is not None else None

        # Look up the reason string for this trade's entry/exit date
        e_reason = ""
        x_reason = ""
        e_val = None
        x_val = None

        if entry_reasons is not None and entry_date is not None:
            key = pd.Timestamp(entry_date)
            if key in entry_reasons.index:
                e_reason = entry_reasons.get(key, "")
            # Try matching by date (tz-aware index)
            if not e_reason:
                for idx, val in entry_reasons.items():
                    if hasattr(idx, 'date') and idx.date() == entry_date and val:
                        e_reason = val
                        break

        if exit_reasons is not None and exit_date is not None:
            key = pd.Timestamp(exit_date)
            if key in exit_reasons.index:
                x_reason = exit_reasons.get(key, "")
            if not x_reason:
                for idx, val in exit_reasons.items():
                    if hasattr(idx, 'date') and idx.date() == exit_date and val:
                        x_reason = val
                        break

        if entry_values is not None and entry_date is not None:
            for idx, val in entry_values.items():
                if hasattr(idx, 'date') and idx.date() == entry_date and pd.notna(val):
                    e_val = float(val)
                    break

        if exit_values is not None and exit_date is not None:
            for idx, val in exit_values.items():
                if hasattr(idx, 'date') and idx.date() == exit_date and pd.notna(val):
                    x_val = float(val)
                    break

        logs.append({
            "ticker":                 ticker,
            "strategy_name":          strategy_name,
            "entry_date":             str(entry_date) if entry_date else None,
            "exit_date":              str(exit_date) if exit_date else None,
            "entry_price":            float(t.get("Avg Entry Price", 0)),
            "exit_price":             float(t.get("Avg Exit Price", 0)) if t.get("Avg Exit Price") is not None else None,
            "direction":              "long",
            "pnl":                    float(t.get("PnL", 0)) if t.get("PnL") is not None else None,
            "return_pct":             float(t.get("Return", 0) * 100) if t.get("Return") is not None else None,
            "is_win":                 bool(t.get("Return", 0) > 0) if t.get("Return") is not None else None,
            "entry_reason":           e_reason or None,
            "exit_reason":            x_reason or None,
            "entry_indicator_value":  e_val,
            "exit_indicator_value":   x_val,
        })
    return logs


def build_strategies(ticker: str) -> list:
    """Build all strategy instances for a given ticker."""
    return [
        AiAgentStrategy(ticker),
        BuyHoldStrategy(),
        SmaCrossoverStrategy(fast=20, slow=50),
        SmaCrossoverStrategy(fast=10, slow=30),
        RsiStrategy(window=14, oversold=30, overbought=70),
        BollingerStrategy(window=20, std_dev=2.0),
        MacdStrategy(fast=12, slow=26, signal=9),
        CompositeStrategy(),
        EnsembleStrategy(min_votes=2),
        MlForecastStrategy(),
        ProphetForecastStrategy(horizon=5, retrain_step=42),
    ]


def generate_diagnostic(metrics: dict, trade_logs: list[dict] | None = None) -> str:
    """
    Generate a plain-English explanation of the backtest result.
    No LLM needed — pure rules-based, instant.

    Explains:
      - What the strategy did overall
      - Why win rate and return might disagree
      - Whether the strategy is worth using
      - What triggered each individual trade
    """
    strategy = metrics.get("strategy_name", "unknown")
    total_ret = metrics.get("total_return_pct") or 0
    win_rate = metrics.get("win_rate_pct") or 0
    total_trades = metrics.get("total_trades", 0)
    wins = metrics.get("winning_trades", 0)
    losses = metrics.get("losing_trades", 0)
    avg_win = metrics.get("avg_win_pct") or 0
    avg_loss = metrics.get("avg_loss_pct") or 0
    expectancy = metrics.get("expectancy_pct") or 0
    rr = metrics.get("risk_reward") or 0
    sharpe = metrics.get("sharpe_ratio") or 0
    max_dd = metrics.get("max_drawdown_pct") or 0

    lines = []

    # ── 1. Strategy description from config ──────────────────
    sdef = STRATEGY_DEFINITIONS.get(strategy, {})
    if sdef:
        lines.append(f"[What is this strategy?]")
        lines.append(sdef.get("description", "").strip())

    # ── 1b. What triggers a BUY or SELL ──────────────────────
    if sdef:
        buy_t = sdef.get("buy_trigger", "")
        sell_t = sdef.get("sell_trigger", "")
        if buy_t or sell_t:
            lines.append("")
            lines.append("[What triggers a trade?]")
            if buy_t:
                lines.append(f"  BUY  -> {buy_t}")
            if sell_t:
                lines.append(f"  SELL -> {sell_t}")

    # ── 2. Win/Loss definition ───────────────────────────────
    lines.append("")
    lines.append("[How wins and losses are defined]")
    lines.append(
        "WIN  = a trade where exit price > entry price (after fees and slippage). "
        "The strategy made money on this trade."
    )
    lines.append(
        "LOSS = a trade where exit price <= entry price. "
        "The strategy lost money or broke even on this trade."
    )

    # ── 3. Result summary ────────────────────────────────────
    lines.append("")
    lines.append(f"[Result for {strategy}]")
    lines.append(
        f"{total_trades} total trades: {wins} wins ({win_rate:.1f}%) "
        f"and {losses} losses ({100 - win_rate:.1f}%)."
    )

    if total_trades == 0:
        lines.append(
            "No trades were generated. This usually means the strategy "
            "conditions were never met in this date range."
        )
        return "\n".join(lines)

    # ── 4. The win-rate vs return contradiction ──────────────
    lines.append("")
    if total_ret >= 0 and win_rate >= 50:
        lines.append(
            f"VERDICT: Good. Both return ({total_ret:+.1f}%) and win rate ({win_rate:.1f}%) "
            f"are positive. This strategy is working as expected."
        )
    elif total_ret >= 0 and win_rate < 50:
        lines.append(
            f"VERDICT: Profitable despite low win rate. Return is {total_ret:+.1f}% "
            f"but only {win_rate:.1f}% of trades are winners. This means the {wins} winning trades "
            f"made +{avg_win:.1f}% each on average, while the {losses} losing trades lost "
            f"only {avg_loss:.1f}% each. The wins are much bigger than the losses (R:R = 1:{rr:.1f}). "
            f"Your strategy lets winners run and cuts losers early — this is the correct approach."
        )
    elif total_ret < 0 and win_rate >= 50:
        loss_mult = f"{1/rr:.1f}x bigger" if rr else "much bigger"
        lines.append(
            f"VERDICT: Losing money despite high win rate. Return is {total_ret:.1f}% "
            f"even though {win_rate:.1f}% of trades are winners. The problem: {wins} small wins "
            f"of +{avg_win:.1f}% each cannot cover {losses} big losses of {avg_loss:.1f}% each. "
            f"Risk:Reward is 1:{rr:.1f} — your average loss is {loss_mult} than your average win. "
            f"You are taking small profits and letting losses grow. To fix this: use stop-losses, "
            f"or let winning trades run longer instead of selling too early."
        )
    else:
        lines.append(
            f"VERDICT: Both return ({total_ret:.1f}%) and win rate ({win_rate:.1f}%) are poor. "
            f"The strategy is losing on most trades and the losses are bigger than the wins. "
            f"This strategy does not work for this ticker in this time period."
        )

    # ── 5. Expectancy interpretation ─────────────────────────
    lines.append("")
    if expectancy > 0:
        lines.append(
            f"Expectancy is {expectancy:+.2f}% per trade — positive. "
            f"Over many trades, this strategy is expected to make money. "
            f"The more trades you take, the more reliable this becomes."
        )
    elif expectancy < 0:
        lines.append(
            f"Expectancy is {expectancy:.2f}% per trade — negative. "
            f"Every trade you take, on average, loses {abs(expectancy):.2f}%. "
            f"The more you trade, the more you lose. Stop using this strategy "
            f"or adjust its parameters."
        )
    else:
        lines.append("Expectancy is zero — the strategy breaks even before fees.")

    # ── 6. Risk warning ──────────────────────────────────────
    if max_dd > 20:
        lines.append("")
        lines.append(
            f"WARNING: Max drawdown is {max_dd:.1f}%. "
            f"At some point your portfolio dropped {max_dd:.1f}% from its peak. "
            f"If you started with 100,000, you would have seen it fall to "
            f"{100000 * (1 - max_dd/100):,.0f}. "
            f"Ask yourself: would you hold through that, or would you panic-sell?"
        )

    return "\n".join(lines)


def run_backtest_for_ticker(
    ticker: str,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    init_cash: float = DEFAULT_INIT_CASH,
    fees: float = DEFAULT_FEES,
    slippage: float = DEFAULT_SLIPPAGE,
    store: bool = True,
    strategy_names: list[str] | None = None,
) -> list[dict]:
    """Run strategies for one ticker, optionally store results to Supabase."""
    label = ", ".join(strategy_names) if strategy_names else "all strategies"
    print(f"\n{'='*60}")
    print(f"  Backtesting: {ticker}  ({start} to {end})  [{label}]")
    print(f"{'='*60}")

    price = load_price_data(ticker, start, end)
    if price.empty:
        print(f"  No price data for {ticker}, skipping.")
        return []

    try:
        from .price_history import coverage_summary

        cov = coverage_summary(ticker)
        print(
            f"  Price bars: {len(price)} closes | DB cache: {cov.get('bars', 0)} bars "
            f"({cov.get('date_from')} → {cov.get('date_to')}, {cov.get('years')}y)"
        )
    except Exception:
        print(f"  Price bars: {len(price)} closes ({price.index.min().date()} → {price.index.max().date()})")

    volume = load_volume_data(ticker, start, end)

    # AI accuracy/IC use the same DB signals the ai_agent strategy replays
    # (executions → history+cache), computed after strategies are built.
    ai_accuracy = {}
    ai_ic = {}

    strategies = build_strategies(ticker)
    if strategy_names:
        allowed = set(strategy_names)
        strategies = [s for s in strategies if s.name in allowed]
        if not strategies:
            print(f"  No matching strategies for filter: {strategy_names}")
            return []

    ai_strategy = next((s for s in strategies if s.name == "ai_agent"), None)
    if ai_strategy is not None:
        ai_recs = getattr(ai_strategy, "_recommendations", pd.DataFrame())
        if ai_recs is not None and not ai_recs.empty:
            ai_accuracy = compute_signal_accuracy(ai_recs, price)
            ai_ic = compute_ic(ai_recs, price)
            cfg = getattr(ai_strategy, "config", {}) or {}
            print(
                f"  AI signal source: {cfg.get('source')} | "
                f"rows={cfg.get('rows')} buy={cfg.get('buy_signals')} sell={cfg.get('sell_signals')}"
            )
        else:
            print("  AI signal source: none (no BUY/SELL in executions/history/cache for this ticker)")

    results = []

    for strategy in strategies:
        print(f"\n  Strategy: {strategy.name}")
        if strategy.name == "ml_forecast" and volume is not None:
            strategy._volume = volume.reindex(price.index)  # type: ignore[attr-defined]
        try:
            pf = strategy.run(price, init_cash=init_cash, fees=fees, slippage=slippage)
        except Exception as e:
            logger.error(
                "Strategy %s failed for %s: %s\n%s", strategy.name, ticker, e,
                traceback.format_exc(),
            )
            print(f"    ERROR: {e}")
            continue

        metrics = extract_metrics(pf)
        metrics["strategy_name"] = strategy.name
        metrics["ticker"] = ticker
        metrics["date_from"] = start
        metrics["date_to"] = end
        metrics["init_cash"] = init_cash
        metrics["fees_pct"] = fees
        metrics["slippage_pct"] = slippage
        metrics["strategy_config"] = getattr(strategy, "config", {})

        if strategy.name == "ai_agent" and ai_accuracy:
            metrics["buy_precision"] = ai_accuracy.get("buy_precision")
            metrics["sell_precision"] = ai_accuracy.get("sell_precision")
            metrics["directional_acc"] = ai_accuracy.get("directional_acc")

        # IC / Rank-IC for the AI agent (Qlib §1.4). None on <3 scoreable signals.
        if strategy.name == "ai_agent" and ai_ic:
            metrics["ic"] = ai_ic.get("ic")
            metrics["rank_ic"] = ai_ic.get("rank_ic")

        # ML metadata for the LightGBM strategy (migration 009). Read from the
        # instance attributes set during generate_signals; safe-guarded getattr.
        if strategy.name == "ml_forecast":
            metrics["ml_horizon"] = getattr(strategy, "last_horizon", None)
            metrics["ml_train_rows"] = getattr(strategy, "last_train_rows", None)
            metrics["ml_feature_count"] = getattr(strategy, "last_feature_count", None)
            _retrain = getattr(strategy, "last_retrain_date", None)
            metrics["ml_retrain_date"] = (
                str(_retrain.date()) if hasattr(_retrain, "date") else (str(_retrain) if _retrain else None)
            )
            preds = getattr(strategy, "last_predictions", None)
            if preds is not None:
                horizon = getattr(strategy, "last_horizon", None) or 5
                ml_ic = compute_ic_from_predictions(preds, price, hold_days=int(horizon))
                metrics["ic"] = ml_ic.get("ic")
                metrics["rank_ic"] = ml_ic.get("rank_ic")

        _ret = metrics.get('total_return_pct')
        _sharpe = metrics.get('sharpe_ratio')
        _trades = metrics.get('total_trades')
        _wr = metrics.get('win_rate_pct')
        _aw = metrics.get('avg_win_pct')
        _al = metrics.get('avg_loss_pct')
        _exp = metrics.get('expectancy_pct')
        _rr = metrics.get('risk_reward')
        print(
            f"    Return: {_ret:+.1f}%  |  " if _ret is not None else "    Return: N/A  |  "
            f"Sharpe: {_sharpe:.2f}  |  " if _sharpe is not None else "Sharpe: N/A  |  "
            f"Trades: {_trades}  |  "
            f"Win Rate: {_wr:.1f}%  |  " if _wr is not None else "Win Rate: N/A  |  "
            f"Avg Win: +{_aw:.1f}%  |  " if _aw is not None else "Avg Win: N/A  |  "
            f"Avg Loss: {_al:.1f}%  |  " if _al is not None else "Avg Loss: N/A  |  "
            f"Expectancy: {_exp:+.2f}%  |  " if _exp is not None else "Expectancy: N/A  |  "
            f"R:R = 1:{_rr:.1f}" if _rr is not None else "R:R = N/A"
        )

        # Build trade logs with reasons (needed for diagnostic display AND DB storage)
        e_reasons = x_reasons = e_vals = x_vals = None
        try:
            e_reasons, x_reasons, e_vals, x_vals = strategy.build_trade_reasons(price)
        except Exception:
            pass  # not all strategies implement this

        trade_logs = extract_trade_logs(
            pf, ticker, strategy.name,
            entry_reasons=e_reasons, exit_reasons=x_reasons,
            entry_values=e_vals, exit_values=x_vals,
        )

        diagnostic = generate_diagnostic(metrics, trade_logs)
        metrics["diagnostic"] = diagnostic
        print(f"    {diagnostic}")

        if store:
            try:
                result_id = store_backtest_result(metrics)
                if trade_logs:
                    store_trade_logs(result_id, trade_logs)
                metrics["db_result_id"] = result_id
            except Exception as e:
                logger.error(
                    "DB store failed for %s/%s: %s\n%s", ticker, strategy.name, e,
                    traceback.format_exc(),
                )
                print(f"    DB store failed: {e}")

        results.append(metrics)

    return results


def run_all(
    tickers: list[str] = TICKERS,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    store: bool = True,
) -> list[dict]:
    """Run backtests for all tickers and all strategies."""
    all_results = []
    for ticker in tickers:
        results = run_backtest_for_ticker(ticker, start, end, store=store)
        all_results.extend(results)

    # Print summary
    if all_results:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        for r in sorted(all_results, key=lambda x: x.get("total_return_pct", 0), reverse=True):
            print(
                f"  {r['ticker']:15s} {r['strategy_name']:20s} "
                f"Return: {r['total_return_pct']:+8.1f}%  "
                f"Sharpe: {r['sharpe_ratio']:5.2f}  "
                f"Trades: {r['total_trades']:4d}"
            )

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VectorBT Backtest Runner")
    parser.add_argument("--ticker", type=str, help="Single ticker (e.g. TCS.NS, BTC-USD)")
    parser.add_argument("--all", action="store_true", help="Run all tickers")
    parser.add_argument("--start", default=BACKTEST_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=BACKTEST_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-store", action="store_true", help="Don't write results to DB")
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        help="Limit to strategy name(s), e.g. --strategy ai_agent (repeatable)",
    )
    args = parser.parse_args()

    store = not args.no_store
    strategy_names = args.strategies or None

    # Show which Supabase project this run uses (from swing-trading-dev/.env)
    from .config import SUPABASE_URL
    host = (SUPABASE_URL or "").replace("https://", "").split(".")[0] or "(missing)"
    print(f"Supabase project: {host}")

    if args.all:
        run_all(start=args.start, end=args.end, store=store)
    elif args.ticker:
        run_backtest_for_ticker(
            args.ticker,
            start=args.start,
            end=args.end,
            store=store,
            strategy_names=strategy_names,
        )
    else:
        parser.print_help()
