"""Modelos Pydantic para mensajes, configuración y respuestas del LLM."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Roles válidos para un mensaje de chat."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje individual dentro de la conversación."""

    role: Role
    content: str = Field(..., min_length=1, description="Contenido del mensaje, no puede estar vacío")


class ModelConfig(BaseModel):
    """Configuración de la llamada al modelo. Pydantic valida los rangos automáticamente."""

    provider: str = Field(..., description="Proveedor a usar: 'openai' o 'anthropic'")
    model: str = Field(..., description="Nombre del modelo, ej. 'gpt-4o-mini' o 'claude-sonnet-5'")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Creatividad del modelo (0 a 2)")
    max_tokens: int = Field(default=1024, gt=0, description="Cantidad máxima de tokens en la respuesta")


class ModelResponse(BaseModel):
    """Respuesta unificada, sin importar de qué proveedor haya venido."""

    content: str
    provider: str
    model: str
    success: bool = True
    error: Optional[str] = None
