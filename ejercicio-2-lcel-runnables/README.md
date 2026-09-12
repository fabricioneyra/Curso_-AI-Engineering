# Pipeline de Extracción de Entidades Técnicas (LCEL)

Pipeline que recibe un texto sin procesar (descripción de arquitectura, log de
error, etc.) y devuelve un objeto **validado con Pydantic**, usando una cadena
LCEL (`prompt | modelo | validación`) con reintento automático ante salidas
mal formadas o incompletas.

## Estructura del proyecto

```
ejercicio-2-lcel-runnables/
├── schemas.py       # Modelo Pydantic de salida (EntidadesTecnicas)
├── chain.py         # Prompt, cadena LCEL, with_structured_output y with_retry
├── main.py          # Mini-script de prueba asíncrono
├── requirements.txt
├── .env.example
└── README.md
```

## Instalación

1. Creá un entorno virtual (Python 3.12) e instalá las dependencias:
   ```bash
   py -3.12 -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Copiá `.env.example` a `.env` y completá tu key:
   ```
   PROVIDER=openai
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

## Uso

```bash
python main.py
```

Corre el pipeline sobre 3 textos de ejemplo: una descripción de arquitectura
clara, un texto ambiguo (la "prueba de estrés") y un log de error.

Para usar el pipeline en tu propio código:

```python
import asyncio
from chain import process_text

async def main():
    resultado = await process_text("Tu texto acá...")
    print(resultado.model_dump_json(indent=2))

asyncio.run(main())
```

## Ejemplo de salida esperada

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

## Diseño

- **`schemas.py`**: `EntidadesTecnicas` valida que `tecnologias` no quede vacía
  (y limpia duplicados/espacios), que `nivel_de_criticidad` sea uno de
  `baja`/`media`/`alta` (via `Enum`), y que `resumen_tecnico` tenga contenido
  mínimo real.
- **`chain.py`**: usa `ChatPromptTemplate` (nunca f-strings sueltos) para que
  LangChain gestione la variable `{texto}` de forma modular, y
  `model.with_structured_output(EntidadesTecnicas, include_raw=True)` para que
  el LLM devuelva directamente un objeto validado.
- **`include_raw=True`** es la pieza clave para no "ignorar el finish_reason":
  sin esto, solo veríamos el objeto ya parseado (o `None` si falló) y no
  tendríamos forma de distinguir "el modelo no dijo nada" de "el modelo dijo
  algo pero se cortó por límite de tokens". Con `include_raw=True` accedemos a
  `raw.response_metadata["finish_reason"]` y lo chequeamos explícitamente
  antes de confiar en el resultado.
- **Reintento (`.with_retry()`)**: envuelve toda la cadena
  (`prompt | modelo | validación`) y reintenta hasta 3 veces
  (`stop_after_attempt=3`, con backoff exponencial) específicamente ante
  `OutputParserException` (JSON mal formado, o detectado como truncado por
  nuestro chequeo de `finish_reason`) o `pydantic.ValidationError` (el modelo
  devolvió algo que no cumple el esquema, ej. una tecnología vacía). Errores
  no recuperables como una API key inválida **no** se reintentan — se
  probó explícitamente que no consumen los 3 intentos en vano.

## Nota sobre `temperature` por proveedor

Igual que en el Módulo 1: los modelos recientes de Anthropic (4.7+, incluido
`claude-sonnet-5`) ya no aceptan `temperature` explícito. `chain.py` solo lo
setea (`temperature=0`, para respuestas más determinísticas) cuando el
proveedor es OpenAI.
