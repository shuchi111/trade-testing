"""Model name validators for each provider."""

VALID_MODELS = {
    "openai": ["gpt-5.4-pro", "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
    "***REMOVED***": [
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5",
        # Z.ai Anthropic gateway (glm-* model names)
        "glm-5.2", "glm-5.1", "glm-4.7", "glm-4-plus", "glm-4-flash", "glm-4.5", "glm-4.5-air",
    ],
    "google": ["gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview", "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "xai": ["grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning", "grok-4-0709", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning"],
    "glm": ["glm-5.2", "glm-5.1", "glm-4.7", "glm-4-plus", "glm-4-flash", "glm-4.5", "glm-4.5-air"],
    # Additive (upstream parity): additional OpenAI-compatible providers.
    # Lists are best-effort and non-exhaustive; validate_model stays permissive.
    "nvidia": ["deepseek-ai/deepseek-r1", "meta/llama-3.1-405b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct"],
    "kimi": ["moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k", "kimi-latest"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "mistral": ["mistral-large-latest", "mistral-small-latest", "open-mixtral-8x22b", "open-mistral-7b", "codestral-latest"],
}


def validate_model(provider: str, model: str) -> bool:
    provider_lower = provider.lower()
    # Self-hosted / aggregator providers accept arbitrary model names.
    if provider_lower in ("ollama", "openrouter"):
        return True
    if provider_lower not in VALID_MODELS:
        return True
    return model in VALID_MODELS[provider_lower]
