"""Punto de entrada único: elige e instancia el proveedor correcto."""

import os
from typing import AsyncGenerator, List, Optional

from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient
from .schemas import ChatMessage, LLMConfig, ModelConfig, ModelResponse

_PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
}


class AsyncLLMManager:
    """Envuelve a OpenAIClient / AnthropicClient / GeminiClient detrás de una única interfaz."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        provider = (provider or os.getenv("PROVIDER", "openai")).lower()
        if provider not in _PROVIDERS:
            raise ValueError(
                f"Proveedor '{provider}' no soportado. Usá 'openai', 'anthropic' o 'gemini'."
            )
        self.provider = provider

        if api_key is not None:
            # Permite seguir inyectando una key a mano (tests, notebooks, etc.)
            resolved_key = api_key
        else:
            # Camino normal: valida con Pydantic ANTES de tocar ningún SDK.
            # Si falta la key, esto lanza un ValueError con mensaje claro en
            # vez de dejar que cada SDK explote con su propio error interno.
            config = LLMConfig.from_env(provider)
            resolved_key = config.get_active_key()

        self._client: BaseLLMClient = _PROVIDERS[provider](api_key=resolved_key)

    async def generate(self, messages: List[ChatMessage], config: ModelConfig) -> ModelResponse:
        return await self._client.generate(messages, config)

    async def generate_stream(self, messages: List[ChatMessage], config: ModelConfig) -> AsyncGenerator[str, None]:
        async for chunk in self._client.generate_stream(messages, config):
            yield chunk

    async def aclose(self) -> None:
        """Cierra la conexión HTTP del proveedor activo. Llamar siempre al terminar."""
        await self._client.aclose()
