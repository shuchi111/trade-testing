from typing import Any, Optional
from .base_client import BaseLLMClient

_OPENAI_COMPATIBLE = {"openai", "xai", "openrouter", "ollama", "glm"}


def create_llm_client(provider: str, model: str, base_url: Optional[str] = None, **kwargs) -> BaseLLMClient:
    """Factory function to create the appropriate LLM client.
    
    Supported providers: openai, anthropic, google, xai, openrouter, ollama, glm
    """
    provider_lower = provider.lower()

    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model=model, base_url=base_url, **kwargs)

    if provider_lower == "google":
        from .google_client import GoogleClient
        return GoogleClient(model=model, base_url=base_url, **kwargs)

    if provider_lower in _OPENAI_COMPATIBLE:
        from .openai_client import OpenAIClient
        return OpenAIClient(model=model, base_url=base_url, provider=provider_lower, **kwargs)

    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Supported providers: anthropic, google, openai, xai, openrouter, ollama, glm"
    )
