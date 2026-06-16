"""Model name validators for each provider."""

VALID_MODELS = {
    "openai": ["gpt-5.4-pro", "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
    "anthropic": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
    "google": ["gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview", "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "xai": ["grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning", "grok-4-0709", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning"],
    "glm": ["glm-5.1", "glm-4.7", "glm-4-plus", "glm-4-flash", "glm-4.5", "glm-4.5-air"],
}


def validate_model(provider: str, model: str) -> bool:
    provider_lower = provider.lower()
    if provider_lower in ("ollama", "openrouter"):
        return True
    if provider_lower not in VALID_MODELS:
        return True
    return model in VALID_MODELS[provider_lower]
