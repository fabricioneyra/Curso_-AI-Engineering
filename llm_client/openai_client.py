"""Implementación del cliente unificado para OpenAI."""

import asyncio
from typing import AsyncGenerator, List, Optional

from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

from .base import BaseLLMClient
from .schemas import ChatMessage, ModelConfig, ModelResponse

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0


class OpenAIClient(BaseLLMClient):
    """Cliente asíncrono para modelos de OpenAI (usa AsyncOpenAI internamente)."""

    def __init__(self, api_key: Optional[str] = None):
        # Si no se pasa api_key, el SDK la toma automáticamente de OPENAI_API_KEY
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(self, messages: List[ChatMessage], config: ModelConfig) -> ModelResponse:
        payload = [{"role": m.role.value, "content": m.content} for m in messages]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=config.model,
                    messages=payload,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return ModelResponse(
                    content=response.choices[0].message.content or "",
                    provider="openai",
                    model=config.model,
                )
            except AuthenticationError as e:
                # No tiene sentido reintentar si la key está mal
                return self._error(config, f"API key inválida: {e}")
            except RateLimitError as e:
                if attempt == MAX_RETRIES:
                    return self._error(config, f"Límite de tasa excedido tras {MAX_RETRIES} intentos: {e}")
                await asyncio.sleep(BASE_DELAY_SECONDS * attempt)
            except APIConnectionError as e:
                if attempt == MAX_RETRIES:
                    return self._error(config, f"Error de conexión tras {MAX_RETRIES} intentos: {e}")
                await asyncio.sleep(BASE_DELAY_SECONDS * attempt)
            except APIError as e:
                return self._error(config, f"Error de la API de OpenAI: {e}")
            except Exception as e:  # red de seguridad: nunca crashear el loop principal
                return self._error(config, f"Error inesperado: {e}")

        return self._error(config, "Se agotaron los reintentos")

    async def generate_stream(self, messages: List[ChatMessage], config: ModelConfig) -> AsyncGenerator[str, None]:
        payload = [{"role": m.role.value, "content": m.content} for m in messages]
        try:
            stream = await self._client.chat.completions.create(
                model=config.model,
                messages=payload,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (AuthenticationError, RateLimitError, APIConnectionError, APIError) as e:
            yield f"[ERROR] {e}"
        except Exception as e:
            yield f"[ERROR inesperado] {e}"

    @staticmethod
    def _error(config: ModelConfig, message: str) -> ModelResponse:
        return ModelResponse(content="", provider="openai", model=config.model, success=False, error=message)
