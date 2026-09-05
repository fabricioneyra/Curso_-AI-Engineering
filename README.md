# Unified Async LLM Client

Cliente asíncrono unificado para modelos de OpenAI, Anthropic y Gemini, con
streaming y validación de datos con Pydantic. Permite intercambiar de
proveedor sin cambiar el código que lo consume.

## Estructura del proyecto

```
async-llm-client/
├── llm_client/
│   ├── __init__.py
│   ├── schemas.py          # Modelos Pydantic: ChatMessage, ModelConfig, ModelResponse
│   ├── base.py              # Clase base abstracta BaseLLMClient
│   ├── openai_client.py     # Implementación para OpenAI (AsyncOpenAI)
│   ├── anthropic_client.py  # Implementación para Anthropic (AsyncAnthropic)
│   ├── gemini_client.py     # Implementación para Gemini (google-genai, namespace .aio)
│   └── manager.py           # AsyncLLMManager: elige el proveedor según config/env
├── main.py                   # Script de validación (modo normal + streaming)
├── .env.example
├── requirements.txt
└── README.md
```

## Instalación

1. Cloná el repo y entrá en la carpeta.
2. Creá un entorno virtual con **Python 3.12** (verificá tu versión con `py --version` en Windows o `python3 --version` en Mac/Linux):
   ```bash
   py -m venv venv          # Windows
   venv\Scripts\activate
   # python3 -m venv venv    # Mac/Linux
   # source venv/bin/activate
   ```
3. Instalá las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Configuración

1. Copiá `.env.example` a `.env`:
   ```bash
   copy .env.example .env   # Windows
   # cp .env.example .env    # Mac/Linux
   ```
2. Completá tu(s) API key(s) y elegí el proveedor por defecto:
   ```
   PROVIDER=openai
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GEMINI_API_KEY=...
   ```

## Uso

Correr el script de prueba (modo normal + streaming) con el proveedor definido en `.env`:
```bash
python main.py
```

Para probar el otro proveedor sin editar el `.env`:
```bash
set PROVIDER=anthropic && python main.py     # Windows
# PROVIDER=anthropic python main.py           # Mac/Linux
```

## Diseño

- **Interfaz común (`BaseLLMClient`)**: `OpenAIClient` y `AnthropicClient` implementan
  los mismos dos métodos (`generate`, `generate_stream`), así que `AsyncLLMManager`
  y el código que lo consume no necesitan saber con qué proveedor están hablando.
- **Asincronía real**: se usan `AsyncOpenAI` y `AsyncAnthropic` (no las versiones
  síncronas), y todas las llamadas van con `await` para no bloquear el event loop.
- **Streaming**: `generate_stream` es un generador asíncrono (`async for ... yield`)
  agnóstico del proveedor.
- **Validación con Pydantic**: `ChatMessage` valida que el contenido no esté vacío;
  `ModelConfig` valida `temperature` (0–2) y `max_tokens` (> 0) automáticamente.

## Manejo de errores

- **API key inválida**: se captura (`AuthenticationError`) y se devuelve un
  `ModelResponse` con `success=False` y el detalle en `error`, sin crashear el
  programa ni el loop principal.
- **Rate limiting / errores de red**: se reintenta hasta 3 veces con backoff
  progresivo antes de devolver el error controlado.
- **Cualquier excepción no prevista**: hay un `except Exception` de red de
  seguridad en ambos clientes, para que un fallo inesperado nunca tumbe el
  programa.

## Nota importante: `temperature` en Anthropic

Durante el desarrollo se detectó que el SDK actual de Anthropic (`anthropic`
≥ 1.4) **ya no acepta el parámetro `temperature`** para los modelos recientes
(Claude 4.7 en adelante, incluido `claude-sonnet-5`): Anthropic los reemplazó
por un control de "esfuerzo" de razonamiento (`output_config.effort`, con
niveles `low` / `medium` / `high` / etc.). Pasar `temperature` a
`client.messages.create()` directamente rompe con un `TypeError`, ni siquiera
llega a la API.

Por eso:
- `ModelConfig.temperature` se sigue validando con Pydantic (0 a 2) porque el
  enunciado lo pide y porque OpenAI sí lo sigue soportando.
- `AnthropicClient` **no reenvía** `temperature` en la llamada real, para
  evitar el crash. Queda documentado en el docstring de la clase.
- Si más adelante querés controlar el comportamiento del modelo de Anthropic,
  el reemplazo oficial es `output_config={"effort": "low"|"medium"|"high"}`.

Esto no afecta a OpenAI: `temperature` se sigue enviando normalmente en
`openai_client.py`. Gemini (agregado después) tampoco se ve afectado: su SDK
sigue soportando `temperature` sin problema.

## Nota sobre Gemini

Al agregar el tercer proveedor aparecieron un par de diferencias más respecto
a OpenAI/Anthropic, todas ya resueltas en `gemini_client.py`:

- El SDK (`google-genai`) es asíncrono a través de un namespace `.aio`
  (`client.aio.models.generate_content(...)`), no con una clase `Async...`
  aparte como en los otros dos proveedores.
- Gemini no usa `role="assistant"` para los turnos del modelo: usa
  `role="model"`. Se traduce automáticamente.
- Al igual que Anthropic, los mensajes de sistema no van en la lista de
  mensajes: se pasan aparte como `system_instruction`.
- El cliente busca la key en `GEMINI_API_KEY` o `GOOGLE_API_KEY` si no se le
  pasa una explícitamente.
