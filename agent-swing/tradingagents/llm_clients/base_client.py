from abc import ABC, abstractmethod
from typing import Any, Optional


def normalize_content(response):
    """Normalize LLM response content to a plain string.
    
    Handles list-of-blocks responses from OpenAI Responses API and Google Gemini 3.
    """
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
    return response


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    @abstractmethod
    def get_llm(self) -> Any:
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        pass
