import os
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = ("timeout", "max_retries", "reasoning_effort", "api_key", "callbacks", "http_client", "http_async_client")

_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "glm": ("https://api.z.ai/api/paas/v4/", "GLM_API_KEY"),
    # Additive (upstream parity): additional OpenAI-compatible providers.
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
}


class NormalizedChatOpenAI(ChatOpenAI):
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None, provider: str = "openai", **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        llm_kwargs = {"model": self.model}

        if self.provider in _PROVIDER_CONFIG:
            default_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            resolved = (self.base_url or "").strip()
            llm_kwargs["base_url"] = resolved.rstrip("/") + "/" if resolved else default_url
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if not api_key and api_key_env == "GLM_API_KEY":
                    api_key = (
                        os.environ.get("Z_API_KEY")
                        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                        or os.environ.get("ANTHROPIC_API_KEY")
                    )
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        if self.provider == "openai":
            flag = (os.getenv("OPENAI_USE_RESPONSES_API") or "1").strip().lower()
            llm_kwargs["use_responses_api"] = flag not in ("0", "false", "no", "off")
        else:
            llm_kwargs["use_responses_api"] = False

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)
