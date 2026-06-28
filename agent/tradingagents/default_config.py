import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "anthropic",
    "deep_think_llm": "glm-5.2",
    "quick_think_llm": "glm-5.2",
    "backend_url": "https://api.z.ai/api/anthropic",
    # Provider-specific thinking configuration
    "google_thinking_level": None,
    "glm_reasoning_effort": None,
    "anthropic_effort": None,
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 1000,
    # Data vendor configuration (exact chain; use "yfinance,alpha_vantage" for explicit fallback)
    "strict_vendor_chain": True,
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    },
    "tool_vendors": {},
}
