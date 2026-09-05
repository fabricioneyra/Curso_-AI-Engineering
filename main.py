"""Script de validación: prueba el cliente en modo normal y en streaming."""

import asyncio
import os

from dotenv import load_dotenv

from llm_client import AsyncLLMManager, ChatMessage, ModelConfig, Role

load_dotenv()

QUESTION = "¿Qué es la entropía?"

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-2.5-flash",
}


async def run_normal(manager: AsyncLLMManager, config: ModelConfig) -> None:
    print(f"\n--- Modo normal ({manager.provider} / {config.model}) ---")
    messages = [ChatMessage(role=Role.USER, content=QUESTION)]
    response = await manager.generate(messages, config)
    if response.success:
        print(response.content)
    else:
        print(f"[Error controlado] {response.error}")


async def run_streaming(manager: AsyncLLMManager, config: ModelConfig) -> None:
    print(f"\n--- Modo streaming ({manager.provider} / {config.model}) ---")
    messages = [ChatMessage(role=Role.USER, content=QUESTION)]
    async for chunk in manager.generate_stream(messages, config):
        print(chunk, end="", flush=True)
    print()


async def main() -> None:
    provider = os.getenv("PROVIDER", "openai").lower()
    manager = AsyncLLMManager(provider=provider)

    config = ModelConfig(
        provider=provider,
        model=DEFAULT_MODELS[provider],
        temperature=0.7,
        max_tokens=300,
    )

    await run_normal(manager, config)
    await run_streaming(manager, config)


if __name__ == "__main__":
    asyncio.run(main())
