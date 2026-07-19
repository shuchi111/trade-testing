from __future__ import annotations

import json
import os
import sys
import time
from typing import Any
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# Load env vars from swing-trader/.env.local (one level up from agent/)
# Falls back to swing-trader/.env if .env.local doesn't exist
SWING_TRADER_ROOT = ROOT.parent
load_dotenv(SWING_TRADER_ROOT / ".env.local")
load_dotenv(SWING_TRADER_ROOT / ".env")

import psycopg2  # type: ignore[reportMissingModuleSource]

from canonical_decision import resolve_canonical_decision
from db_url import resolve_psycopg2_url
from portfolio_db import build_analysis_context
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.confidence_extraction import (
    build_quick_llm_from_config,
    extract_confidence_pct,
)
from tradingagents.graph.report_builder import build_complete_report
from tradingagents.graph.signal_processing import is_transient_propagate_error
from tradingagents.graph.trading_graph import TradingAgentsGraph

class RunPropagateApp:
    """Coordinates one stdin-driven TradingAgents propagation run."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.ticker = str(payload.get("ticker") or "NVDA").strip()
        self.trade_date = str(payload.get("trade_date") or "").strip()
        self.debug = bool(payload.get("debug", False))
        self.holding_qty = float(payload.get("holding_quantity") or 0)
        self.holding_entry = float(payload.get("holding_avg_entry") or 0)

    @staticmethod
    def print_error(message: str, **extra: Any) -> None:
        print(json.dumps({"ok": False, "error": message, **extra}, ensure_ascii=True))

    def build_config(self) -> dict:
        config = DEFAULT_CONFIG.copy()
        # All model/provider settings come from env so deployments can swap models without source edits.
        config["llm_provider"] = os.getenv("LLM_PROVIDER", "anthropic").strip()
        backend = (
            (os.getenv("LLM_BACKEND_URL") or "").strip()
            or (os.getenv("ANTHROPIC_BASE_URL") or "").strip()
            or "https://api.z.ai/api/anthropic"
        )
        config["backend_url"] = backend.rstrip("/")
        config["deep_think_llm"] = os.getenv("DEEP_THINK_LLM", "glm-5.2")
        config["quick_think_llm"] = os.getenv("QUICK_THINK_LLM", "glm-5.2")
        config["max_debate_rounds"] = int(os.getenv("MAX_DEBATE_ROUNDS", "1"))
        config["api_key"] = (
            (os.getenv("Z_API_KEY") or "").strip()
            or (os.getenv("GLM_API_KEY") or "").strip()
            or (os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()
            or (os.getenv("ANTHROPIC_API_KEY") or "").strip()
            or ""
        )
        config["data_vendors"].update({
            "core_stock_apis": os.getenv("DATA_VENDOR_STOCKS", "yfinance"),
            "technical_indicators": os.getenv("DATA_VENDOR_INDICATORS", "yfinance"),
            "fundamental_data": os.getenv("DATA_VENDOR_FUNDAMENTALS", "yfinance"),
            "news_data": os.getenv("DATA_VENDOR_NEWS", "yfinance"),
        })
        return config

    def simple_portfolio_context(self) -> str:
        if self.holding_qty > 0 and self.holding_entry > 0:
            return (
                f"You currently hold {self.holding_qty:.0f} units of {self.ticker} "
                f"at an average entry price of {self.holding_entry:,.2f}. "
                "Use this average entry as the percentage basis for swing framing. "
                "Look for the best risk-adjusted exit inside the 90-day swing window, "
                "while respecting the mandatory 5% trailing stop as the hard risk guard. "
                "Factor the open position into whether to add, hold, trim, or exit."
            )
        return f"You currently have no open position in {self.ticker}."

    def build_portfolio_context(self) -> str:
        fallback = self.simple_portfolio_context()
        db_url = resolve_psycopg2_url()
        if not db_url:
            return fallback

        try:
            conn = psycopg2.connect(db_url)
        except Exception:
            return fallback

        try:
            context = build_analysis_context(conn, self.ticker, trade_date=self.trade_date)
            conn.rollback()
            if not context.strip():
                return fallback
            return (
                f"{context}\n\n"
                "=== LIVE REPORT REQUIREMENT ===\n"
                "In the final full report, include concrete portfolio and wallet observations: "
                "current position status, cash/reserve, position cap room, active trailing stop, "
                "recent live trades, backtest evidence, and whether the decision adds risk or reduces risk."
            )
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return fallback
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def propagate_with_retry(self, config: dict, portfolio_context: str) -> tuple[dict | None, Any]:
        max_attempts = int(os.getenv("PROPAGATE_MAX_ATTEMPTS", "3"))
        retry_delay = float(os.getenv("PROPAGATE_RETRY_DELAY_SEC", "4"))

        for attempt in range(max_attempts):
            try:
                graph = TradingAgentsGraph(debug=self.debug, config=config)
                return graph.propagate(
                    self.ticker,
                    self.trade_date,
                    portfolio_context=portfolio_context,
                )
            except Exception as exc:
                if attempt < max_attempts - 1 and is_transient_propagate_error(exc):
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
        return None, None

    def run(self) -> int:
        if not self.trade_date:
            self.print_error("trade_date is required (YYYY-MM-DD)")
            return 1

        config = self.build_config()
        if not config.get("api_key"):
            self.print_error("Missing Z_API_KEY or GLM_API_KEY - add it to swing-trader/.env.local")
            return 1

        portfolio_context = self.build_portfolio_context()
        try:
            final_state, decision = self.propagate_with_retry(
                config,
                portfolio_context,
            )
        except Exception as exc:
            self.print_error(str(exc), ticker=self.ticker, trade_date=self.trade_date)
            return 1

        if final_state is None:
            self.print_error(
                "TradingAgents propagate produced no state",
                ticker=self.ticker,
                trade_date=self.trade_date,
            )
            return 1

        report_text = str(final_state.get("final_trade_decision") or "")
        decision = resolve_canonical_decision(
            str(decision) if decision is not None else "",
            report_text,
        )
        full_report = build_complete_report(
            final_state,
            portfolio_context=portfolio_context,
            canonical_decision=decision,
        )
        confidence_llm = build_quick_llm_from_config(config)
        ai_confidence_pct = extract_confidence_pct(
            full_report,
            decision=decision,
            final_trade_decision=report_text,
            llm=confidence_llm,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "ticker": self.ticker,
                    "trade_date": self.trade_date,
                    "decision": decision,
                    "final_trade_decision": report_text,
                    "full_report": full_report,
                    "portfolio_context_snapshot": portfolio_context,
                    "ai_confidence_pct": ai_confidence_pct,
                },
                ensure_ascii=True,
            )
        )
        return 0


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        RunPropagateApp.print_error(f"Invalid JSON stdin: {exc}")
        sys.exit(1)
        return

    sys.exit(RunPropagateApp(payload).run())


if __name__ == "__main__":
    main()
