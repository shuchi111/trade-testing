import os

# ---------------------------------------------------------------------------
# Env-var -> config-key overrides (additive; mirrors upstream TRADINGAGENTS_*).
# Every override is optional — when the env var is unset the hard-coded default
# below is used, so existing behaviour is preserved exactly.
# ---------------------------------------------------------------------------
_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")


def _env_str(key: str, default: str) -> str:
    """Return a trimmed env var, or ``default`` when unset/empty."""
    val = os.getenv(key)
    return val.strip() if val and val.strip() else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if not val or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _data_vendors_from_env(defaults: dict) -> dict:
    """Allow per-category vendor overrides via TRADINGAGENTS_VENDOR_<CATEGORY>.

    Example: TRADINGAGENTS_VENDOR_CORE_STOCK_APIS="yfinance,alpha_vantage"
    Categories are uppercased with spaces -> underscores. Unknown categories
    are ignored, so typos cannot corrupt the config.
    """
    env_prefix = "TRADINGAGENTS_VENDOR_"
    vendors = dict(defaults)
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(env_prefix) or not env_val or not env_val.strip():
            continue
        category = env_key[len(env_prefix):].lower()
        if category in vendors:
            vendors[category] = env_val.strip()
    return vendors


DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": _env_str("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": _env_str("TRADINGAGENTS_LLM_PROVIDER", "***REMOVED***"),
    "deep_think_llm": _env_str("TRADINGAGENTS_DEEP_THINK_LLM", "glm-5.2"),
    "quick_think_llm": _env_str("TRADINGAGENTS_QUICK_THINK_LLM", "glm-5.2"),
    "backend_url": _env_str("TRADINGAGENTS_BACKEND_URL", "***REMOVED***"),
    # Provider-specific thinking configuration
    "google_thinking_level": os.getenv("TRADINGAGENTS_GOOGLE_THINKING_LEVEL") or None,
    "glm_reasoning_effort": os.getenv("TRADINGAGENTS_GLM_REASONING_EFFORT") or None,
    "***REMOVED***_effort": os.getenv("TRADINGAGENTS_ANTHROPIC_EFFORT") or None,
    # Optional explicit API key override (otherwise providers read their own env vars)
    "api_key": os.getenv("TRADINGAGENTS_API_KEY") or None,
    # Debate and discussion settings
    "max_debate_rounds": _env_int("TRADINGAGENTS_MAX_DEBATE_ROUNDS", 1),
    "max_risk_discuss_rounds": _env_int("TRADINGAGENTS_MAX_RISK_DISCUSS_ROUNDS", 1),
    "max_recur_limit": _env_int("TRADINGAGENTS_MAX_RECUR_LIMIT", 1000),
    # Data vendor configuration (exact chain; use "yfinance,alpha_vantage" for explicit fallback)
    "strict_vendor_chain": os.getenv("TRADINGAGENTS_STRICT_VENDOR_CHAIN", "1").strip().lower()
    not in ("0", "false", "no", "off"),
    "data_vendors": _data_vendors_from_env(
        {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
            "macro_data": "fred",
            "prediction_markets": "polymarket",
        }
    ),
    "tool_vendors": {},
    # Optional persistent decision log (markdown). None/"" disables (default).
    # Set TRADINGAGENTS_DECISION_LOG_DIR to enable the TradingMemoryLog.
    "decision_log_dir": os.getenv("TRADINGAGENTS_DECISION_LOG_DIR") or None,
}
