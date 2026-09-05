"""Implementación del cliente unificado para Google Gemini."""

import asyncio
from typing import AsyncGenerator, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from .base import BaseLLMClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Role

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0


class GeminiClient(BaseLLMClient):
    """Cliente asíncrono para modelos de Gemini (usa el namespace `.aio` del SDK).

    Notas de compatibilidad con la API de Gemini:
    - No usa role="assistant" como OpenAI/Anthropic: los turnos del modelo
      llevan role="model". Se traduce automáticamente en `_build_contents`.
    - Los mensajes con role="system" tampoco van en la lista de contenidos;
      se pasan aparte como `system_instruction` (similar a Anthropic).
    - A diferencia de Anthropic, Gemini sí sigue soportando `temperature` y
      `max_output_tokens` en su config (`GenerateContentConfig`), así que acá
      `config.temperature` sí se reenvía.
    """

    def __init__(self, api_key: Optional[str] = None):
        # Si no se pasa api_key, el SDK la toma de GOOGLE_API_KEY o GEMINI_API_KEY
        self._client = genai.Client(api_key=api_key)

    async def generate(self, messages: List[ChatMessage], config: ModelConfig) -> ModelResponse:
        gen_config = self._build_config(messages, config)
        _, contents = self._build_contents(messages)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=config.model,
                    contents=contents,
                    config=gen_config,
                )
                return ModelResponse(content=response.text or "", provider="gemini", model=config.model)
            except ClientError as e:
                if e.code == 429:
                    if attempt == MAX_RETRIES:
                        return self._error(config, f"Límite de tasa excedido tras {MAX_RETRIES} intentos: {e}")
                    await asyncio.sleep(BASE_DELAY_SECONDS * attempt)
                    continue
                # 401/403 y otros errores de cliente no tiene sentido reintentarlos
                return self._error(config, f"Error de la API de Gemini ({e.code}): {e}")
            except ServerError as e:
                if attempt == MAX_RETRIES:
                    return self._error(config, f"Error del servidor tras {MAX_RETRIES} intentos: {e}")
                await asyncio.sleep(BASE_DELAY_SECONDS * attempt)
            except APIError as e:
                return self._error(config, f"Error de la API de Gemini: {e}")
            except Exception as e:  # red de seguridad: nunca crashear el loop principal
                return self._error(config, f"Error inesperado: {e}")

        return self._error(config, "Se agotaron los reintentos")

    async def generate_stream(self, messages: List[ChatMessage], config: ModelConfig) -> AsyncGenerator[str, None]:
        gen_config = self._build_config(messages, config)
        _, contents = self._build_contents(messages)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=config.model,
                contents=contents,
                config=gen_config,
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except (ClientError, ServerError, APIError) as e:
            yield f"[ERROR] {e}"
        except Exception as e:
            yield f"[ERROR inesperado] {e}"

    @staticmethod
    def _build_config(messages: List[ChatMessage], config: ModelConfig) -> "types.GenerateContentConfig":
        system, _ = GeminiClient._build_contents(messages)
        return types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            system_instruction=system,
        )

    @staticmethod
    def _build_contents(messages: List[ChatMessage]) -> Tuple[Optional[str], list]:
        system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
        contents = [
            types.Content(
                role="model" if m.role == Role.ASSISTANT else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != Role.SYSTEM
        ]
        system = "\n".join(system_parts) if system_parts else None
        return system, contents

    @staticmethod
    def _error(config: ModelConfig, message: str) -> ModelResponse:
        return ModelResponse(content="", provider="gemini", model=config.model, success=False, error=message)
