"""Implementación del cliente unificado para Anthropic."""

import asyncio
from typing import AsyncGenerator, List, Optional, Tuple

from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)

from .base import BaseLLMClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Role

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0


class AnthropicClient(BaseLLMClient):
    """Cliente asíncrono para modelos de Anthropic (usa AsyncAnthropic internamente).

    Notas de compatibilidad con la API de Anthropic:
    - A diferencia de OpenAI, no acepta mensajes con role="system" dentro de la
      lista `messages`; van aparte en el parámetro `system`.
    - Los modelos recientes (Claude 4.7 en adelante, incluido claude-sonnet-5)
      eliminaron los parámetros de sampling `temperature`, `top_p` y `top_k`.
      El SDK actual ni siquiera los acepta como argumento (tira TypeError), así
      que NO se reenvía `config.temperature` en la llamada. Si en el futuro
      querés aproximar ese control, el reemplazo oficial es
      `output_config={"effort": "low" | "medium" | "high"}`.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(self, messages: List[ChatMessage], config: ModelConfig) -> ModelResponse:
        kwargs = self._build_kwargs(messages, config)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.messages.create(**kwargs)
                text = "".join(block.text for block in response.content if block.type == "text")
                return ModelResponse(content=text, provider="anthropic", model=config.model)
            except AuthenticationError as e:
                return self._error(config, f"API key inválida: {e}")
            except RateLimitError as e:
                if attempt == MAX_RETRIES:
                    return self._error(config, f"Límite de tasa excedido tras {MAX_RETRIES} intentos: {e}")
                await asyncio.sleep(BASE_DELAY_SECONDS * attempt)
            except APIConnectionError as e:
                if attempt == MAX_RETRIES:
                    return self._error(config, f"Error de conexión tras {MAX_RETRIES} intentos: {e}")
                await asyncio.sleep(BASE_DELAY_SECONDS * attempt)
            except APIStatusError as e:
                return self._error(config, f"Error de la API de Anthropic: {e}")
            except Exception as e:
                return self._error(config, f"Error inesperado: {e}")

        return self._error(config, "Se agotaron los reintentos")

    async def generate_stream(self, messages: List[ChatMessage], config: ModelConfig) -> AsyncGenerator[str, None]:
        kwargs = self._build_kwargs(messages, config)
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except (AuthenticationError, RateLimitError, APIConnectionError, APIStatusError) as e:
            yield f"[ERROR] {e}"
        except Exception as e:
            yield f"[ERROR inesperado] {e}"

    @staticmethod
    def _build_kwargs(messages: List[ChatMessage], config: ModelConfig) -> dict:
        system, chat_messages = AnthropicClient._split_system(messages)
        # No se pasa `temperature`: el SDK actual de Anthropic ya no lo admite
        # para los modelos recientes (ver docstring de la clase).
        kwargs = dict(
            model=config.model,
            messages=chat_messages,
            max_tokens=config.max_tokens,
        )
        if system:
            kwargs["system"] = system
        return kwargs

    @staticmethod
    def _split_system(messages: List[ChatMessage]) -> Tuple[str, list]:
        system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
        chat_messages = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != Role.SYSTEM
        ]
        return "\n".join(system_parts), chat_messages

    @staticmethod
    def _error(config: ModelConfig, message: str) -> ModelResponse:
        return ModelResponse(content="", provider="anthropic", model=config.model, success=False, error=message)
