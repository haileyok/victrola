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
        api: Literal["anthropic", "openai", "openapi", "umans"],
        model: str,
        api_key: str | None,
        endpoint: str | None = None,
        auth_header: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """``auth_header`` (openapi only) is sent instead of ``Authorization: Bearer <api_key>``."""
        if auth_header and api != "openapi":
            raise ValueError("a custom auth header is only supported for openapi")
        if not auth_header and not api_key:
            raise ValueError("either api_key or auth_header is required")
        self._api = api
        self._model = model
        self._api_key = api_key
        self._endpoint = endpoint
        self._auth_header = dict(auth_header) if auth_header else None
        self._reasoning_effort = reasoning_effort or None

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = MAX_SUB_AGENT_TOKENS,
    ) -> str:
        """Single-shot text completion. Returns the response text."""
        if self._api in ("anthropic", "umans"):
            return await self._complete_anthropic(prompt, system, max_tokens)
        else:
            return await self._complete_openai(prompt, system, max_tokens)

    async def _complete_anthropic(
        self, prompt: str, system: str | None, max_tokens: int
    ) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=self._api_key, base_url=self._endpoint
        )
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
                    **(self._auth_header or {"Authorization": f"Bearer {self._api_key}"}),
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    **({"reasoning_effort": self._reasoning_effort} if self._reasoning_effort else {}),
                },
            )
            if not resp.is_success:
                logger.error(
                    "Sub-agent API error %d: %s", resp.status_code, resp.text[:500]
                )
                resp.raise_for_status()

            data = resp.json()
            return data["choices"][0]["message"]["content"]
