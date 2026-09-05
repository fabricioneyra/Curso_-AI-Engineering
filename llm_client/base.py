"""Interfaz común que deben implementar todos los clientes de LLM."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

from .schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    """Clase base abstracta. Cada proveedor implementa estos dos métodos."""

    @abstractmethod
    async def generate(self, messages: List[ChatMessage], config: ModelConfig) -> ModelResponse:
        """Devuelve la respuesta completa del modelo (sin streaming)."""
        raise NotImplementedError

    @abstractmethod
    def generate_stream(self, messages: List[ChatMessage], config: ModelConfig) -> AsyncGenerator[str, None]:
        """Devuelve un generador asíncrono que va emitiendo fragmentos de texto."""
        raise NotImplementedError
