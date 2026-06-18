from typing import Any, Optional
import os
import time

from langchain_***REMOVED*** import ChatAnthropic
from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = ("timeout", "max_retries", "api_key", "max_tokens", "callbacks", "http_client", "http_async_client", "effort")


class NormalizedChatAnthropic(ChatAnthropic):
    def invoke(self, input, config=None, **kwargs):
        gap = float(os.getenv("LLM_REQUEST_GAP_SEC", "0") or "0")
        if gap > 0:
            time.sleep(gap)
        return normalize_content(super().invoke(input, config, **kwargs))


class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        llm_kwargs = {"model": self.model}
        if self.base_url:
            url = self.base_url.rstrip("/")
            llm_kwargs["base_url"] = url
            llm_kwargs["***REMOVED***_api_url"] = url
        if "api_key" in self.kwargs:
            llm_kwargs["api_key"] = self.kwargs["api_key"]
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        # Z.ai 529 overload: fail fast at SDK layer; propagate retry waits minutes between graph runs.
        if "max_retries" not in llm_kwargs and "api.z.ai" in str(self.base_url or ""):
            llm_kwargs["max_retries"] = int(os.getenv("LLM_HTTP_MAX_RETRIES", "2"))
        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("***REMOVED***", self.model)
