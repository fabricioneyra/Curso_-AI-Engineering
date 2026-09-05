"""Punto de entrada único: elige e instancia el proveedor correcto."""

import os
from typing import AsyncGenerator, List, Optional

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse

_PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
}


class AsyncLLMManager:
    """Envuelve a OpenAIClient / AnthropicClient detrás de una única interfaz."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        provider = (provider or os.getenv("PROVIDER", "openai")).lower()
        if provider not in _PROVIDERS:
            raise ValueError(
                f"Proveedor '{provider}' no soportado. Usá 'openai' o 'anthropic'."
            )
        self.provider = provider
        self._client: BaseLLMClient = _PROVIDERS[provider](api_key=api_key)

    async def generate(self, messages: List[ChatMessage], config: ModelConfig) -> ModelResponse:
        return await self._client.generate(messages, config)

    async def generate_stream(self, messages: List[ChatMessage], config: ModelConfig) -> AsyncGenerator[str, None]:
        async for chunk in self._client.generate_stream(messages, config):
            yield chunk
