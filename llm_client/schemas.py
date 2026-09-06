"""Modelos Pydantic para mensajes, configuración y respuestas del LLM."""

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, SecretStr, model_validator


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


class LLMConfig(BaseModel):
    """Carga y valida las API keys ANTES de instanciar cualquier cliente.

    Usar SecretStr en vez de str evita que la key quede expuesta si alguien
    hace print(), la loguea por accidente, o la serializa en un dict/JSON:
    SecretStr la enmascara en cualquiera de esos casos y solo la revela con
    .get_secret_value() explícito.
    """

    provider: str
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    gemini_api_key: Optional[SecretStr] = None

    @model_validator(mode="after")
    def validar_key_del_proveedor_elegido(self) -> "LLMConfig":
        keys_por_proveedor = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
        }
        if self.provider not in keys_por_proveedor:
            raise ValueError(
                f"Proveedor '{self.provider}' no soportado. "
                f"Usá uno de: {', '.join(keys_por_proveedor)}."
            )
        if keys_por_proveedor[self.provider] is None:
            env_var = f"{self.provider.upper()}_API_KEY"
            raise ValueError(
                f"Falta la API key para el proveedor '{self.provider}'. "
                f"Definí {env_var} en tu archivo .env (mirá .env.example)."
            )
        return self

    def get_active_key(self) -> str:
        """Devuelve en texto plano la key del proveedor ya elegido y validado."""
        key: SecretStr = getattr(self, f"{self.provider}_api_key")
        return key.get_secret_value()

    @classmethod
    def from_env(cls, provider: str) -> "LLMConfig":
        """Arma la config leyendo las keys directamente de las variables de entorno."""

        def read_secret(env_var: str) -> Optional[SecretStr]:
            value = os.getenv(env_var)
            return SecretStr(value) if value else None

        return cls(
            provider=provider,
            openai_api_key=read_secret("OPENAI_API_KEY"),
            anthropic_api_key=read_secret("ANTHROPIC_API_KEY"),
            gemini_api_key=read_secret("GEMINI_API_KEY"),
        )
