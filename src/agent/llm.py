"""Lightweight LLM client for sub-agent tasks (summarization, analysis, research).

This is intentionally separate from agent.py to avoid circular imports —
tool definitions need to call this, and agent.py imports from the tools package.
"""

import json
import logging
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

MAX_SUB_AGENT_TOKENS = 8_000


class SubAgentLLM:
    """A simple LLM completion client for sub-agent tools.

    Supports Anthropic and OpenAI-compatible APIs (Moonshot/Kimi, OpenAI, etc.).
    No tool calling — just text in, text out.
    """

    def __init__(
        self,
        api: Literal["anthropic", "openai", "openapi"],
        model: str,
        api_key: str,
        endpoint: str | None = None,
    ) -> None:
        self._api = api
        self._model = model
        self._api_key = api_key
        self._endpoint = endpoint

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = MAX_SUB_AGENT_TOKENS,
    ) -> str:
        """Single-shot text completion. Returns the response text."""
        if self._api == "anthropic":
            return await self._complete_anthropic(prompt, system, max_tokens)
        else:
            return await self._complete_openai(prompt, system, max_tokens)

    async def _complete_anthropic(
        self, prompt: str, system: str | None, max_tokens: int
    ) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            msg = await client.messages.create(**kwargs)
            return "".join(
                block.text for block in msg.content if hasattr(block, "text")
            )
        finally:
            await client.close()

    async def _complete_openai(
        self, prompt: str, system: str | None, max_tokens: int
    ) -> str:
        endpoint = self._endpoint or "https://api.openai.com/v1"
        endpoint = endpoint.rstrip("/")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
            )
            if not resp.is_success:
                logger.error(
                    "Sub-agent API error %d: %s", resp.status_code, resp.text[:500]
                )
                resp.raise_for_status()

            data = resp.json()
            return data["choices"][0]["message"]["content"]
