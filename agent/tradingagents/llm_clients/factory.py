from typing import Any, Optional
from .***REMOVED***_client import AnthropicClient
from .base_client import BaseLLMClient
from .google_client import GoogleClient
from .openai_client import OpenAIClient

_OPENAI_COMPATIBLE = {
    "openai", "xai", "openrouter", "ollama", "glm",
    # Additive (upstream parity): additional OpenAI-compatible gateways.
    "nvidia", "kimi", "groq", "mistral",
}


def create_llm_client(provider: str, model: str, base_url: Optional[str] = None, **kwargs) -> BaseLLMClient:
    """Factory function to create the appropriate LLM client.

    Supported providers: openai, ***REMOVED***, google, xai, openrouter, ollama, glm,
    nvidia, kimi (Moonshot), groq, mistral.
    """
    provider_lower = provider.lower()

    if provider_lower == "***REMOVED***":
        return AnthropicClient(model=model, base_url=base_url, **kwargs)

    if provider_lower == "google":
        return GoogleClient(model=model, base_url=base_url, **kwargs)

    if provider_lower in _OPENAI_COMPATIBLE:
        return OpenAIClient(model=model, base_url=base_url, provider=provider_lower, **kwargs)

    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Supported providers: ***REMOVED***, google, openai, xai, openrouter, ollama, glm, "
        f"nvidia, kimi, groq, mistral"
    )
