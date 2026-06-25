import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import anthropic
import httpx
from anthropic.types import TextBlock, ToolUseBlock

from src.agent.prompt import build_system_prompt
from src.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


@dataclass
class AgentTextBlock:
    text: str


@dataclass
class AgentToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AgentResponse:
    content: list[AgentTextBlock | AgentToolUseBlock]
    stop_reason: Literal["end_turn", "tool_use"]
    reasoning_content: str | None = None
    usage: dict[str, int] | None = None


@dataclass
class AgentEvent:
    kind: Literal["llm_start", "llm_done", "tool_start", "tool_done"]
    data: dict[str, Any] = field(default_factory=dict)


class AgentClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        pass


class AnthropicClient(AgentClient):
    def __init__(
        self, api_key: str, model_name: str = "claude-sonnet-4-5-20250929"
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model_name = model_name

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        system_text = system or build_system_prompt()
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": 16_000,
            "system": [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": self._inject_cache_breakpoints(messages),
        }

        if tools:
            tools = [dict(t) for t in tools]
            tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = tools

        async with self._client.messages.stream(**kwargs) as stream:  # type: ignore
            msg = await stream.get_final_message()

        content: list[AgentTextBlock | AgentToolUseBlock] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                content.append(AgentTextBlock(text=block.text))
            elif isinstance(block, ToolUseBlock):
                content.append(
                    AgentToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=block.input,  # type: ignore
                    )
                )

        usage: dict[str, int] = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        if hasattr(msg.usage, "cache_creation_input_tokens") and msg.usage.cache_creation_input_tokens:
            usage["cache_creation_input_tokens"] = msg.usage.cache_creation_input_tokens
        if hasattr(msg.usage, "cache_read_input_tokens") and msg.usage.cache_read_input_tokens:
            usage["cache_read_input_tokens"] = msg.usage.cache_read_input_tokens

        logger.info("API usage: %s", usage)

        return AgentResponse(
            content=content,
            stop_reason=msg.stop_reason or "end_turn",  # type: ignore TODO: fix this
            usage=usage,
        )

    @staticmethod
    def _inject_cache_breakpoints(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        a helper that adds cache_control breakpoints to the conversation so that
        the conversation prefix is cached across successive calls. we place a single
        breakpoint in the last message's content block, combined with the sys-prompt
        and tool defs breakpoints. ensures that we stay in the 4-breakpoint limit
        that anthropic requires
        """
        if not messages:
            return messages

        # shallow-copy the list so we don't mutate the caller's conversation
        messages = list(messages)
        last_msg = dict(messages[-1])
        content = last_msg["content"]

        if isinstance(content, str):
            last_msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            content = [dict(b) for b in content]
            content[-1] = dict(content[-1])
            content[-1]["cache_control"] = {"type": "ephemeral"}
            last_msg["content"] = content

        messages[-1] = last_msg
        return messages


class OpenAICompatibleClient(AgentClient):
    """client for openapi compatible apis like openai, moonshot, etc"""

    def __init__(self, api_key: str, model_name: str, endpoint: str) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._endpoint = endpoint.rstrip("/")
        self._http = httpx.AsyncClient(timeout=300.0)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        oai_messages = self._convert_messages(messages, system or build_system_prompt())

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": oai_messages,
            "max_tokens": 16_000,
        }

        if tools:
            payload["tools"] = self._convert_tools(tools)

        resp = await self._http.post(
            f"{self._endpoint}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if not resp.is_success:
            logger.error("API error %d: %s", resp.status_code, resp.text[:1000])
            resp.raise_for_status()
        data = resp.json()

        return self._parse_response(data)

    def _convert_messages(
        self, messages: list[dict[str, Any]], system: str
    ) -> list[dict[str, Any]]:
        """for anthropic chats, we'll convert the outputs into a similar format"""
        result: list[dict[str, Any]] = [{"role": "system", "content": system}]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            tool_calls.append(
                                {
                                    "id": block["id"],
                                    "type": "function",
                                    "function": {
                                        "name": block["name"],
                                        "arguments": json.dumps(block["input"]),
                                    },
                                }
                            )
                    oai_msg: dict[str, Any] = {"role": "assistant"}
                    if msg.get("reasoning_content"):
                        oai_msg["reasoning_content"] = msg["reasoning_content"]
                    # some openai-compatible apis reject content: null on
                    # assistant messages with tool_calls, so omit it when empty
                    if text_parts:
                        oai_msg["content"] = "\n".join(text_parts)
                    else:
                        oai_msg["content"] = ""
                    if tool_calls:
                        oai_msg["tool_calls"] = tool_calls
                    result.append(oai_msg)
                elif role == "user":
                    if content and content[0].get("type") == "tool_result":
                        for block in content:
                            raw = block.get("content", "")
                            if isinstance(raw, list):
                                # image content blocks — convert to OAI format
                                oai_parts: list[dict[str, Any]] = []
                                for part in raw:
                                    if part.get("type") == "text":
                                        oai_parts.append(
                                            {"type": "text", "text": part["text"]}
                                        )
                                    elif part.get("type") == "image":
                                        src = part.get("source", {})
                                        data_url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
                                        oai_parts.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": data_url},
                                            }
                                        )
                                    else:
                                        oai_parts.append(
                                            {"type": "text", "text": str(part)}
                                        )
                                result.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block["tool_use_id"],
                                        "content": oai_parts,
                                    }
                                )
                            else:
                                result.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block["tool_use_id"],
                                        "content": raw,
                                    }
                                )
                    else:
                        # User message with multi-block content (e.g. images
                        # pasted alongside text). Convert any Anthropic image
                        # blocks to OAI `image_url` form; pass text through.
                        has_image = any(
                            isinstance(b, dict) and b.get("type") == "image"
                            for b in content
                        )
                        if has_image:
                            oai_parts: list[dict[str, Any]] = []
                            for b in content:
                                if not isinstance(b, dict):
                                    oai_parts.append({"type": "text", "text": str(b)})
                                elif b.get("type") == "text":
                                    oai_parts.append(
                                        {"type": "text", "text": b.get("text", "")}
                                    )
                                elif b.get("type") == "image":
                                    src = b.get("source", {})
                                    data_url = (
                                        f"data:{src.get('media_type', 'image/png')};"
                                        f"base64,{src.get('data', '')}"
                                    )
                                    oai_parts.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": data_url},
                                        }
                                    )
                                else:
                                    oai_parts.append(
                                        {"type": "text", "text": str(b)}
                                    )
                            result.append({"role": "user", "content": oai_parts})
                        else:
                            text = " ".join(b.get("text", str(b)) for b in content)
                            result.append({"role": "user", "content": text})

        return result

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """convert anthropic tool defs to oai function calling format"""
        result = []
        for t in tools:
            func: dict[str, Any] = {
                "name": t["name"],
                "description": t.get("description", ""),
            }
            if "input_schema" in t:
                func["parameters"] = t["input_schema"]
            result.append({"type": "function", "function": func})
        return result

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """convert an oai chat completion resp to agentresponse"""
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        content: list[AgentTextBlock | AgentToolUseBlock] = []

        if message.get("content"):
            content.append(AgentTextBlock(text=message["content"]))

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}
                content.append(
                    AgentToolUseBlock(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        input=args,
                    )
                )

        usage: dict[str, int] | None = None
        raw_usage = data.get("usage")
        if raw_usage:
            usage = {
                "input_tokens": raw_usage.get("prompt_tokens", 0),
                "output_tokens": raw_usage.get("completion_tokens", 0),
            }

        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"
        reasoning_content = message.get("reasoning_content")
        return AgentResponse(
            content=content,
            stop_reason=stop_reason,
            reasoning_content=reasoning_content,
            usage=usage,
        )


MAX_TOOL_RESULT_LENGTH = 4_000


def _format_tool_result(result: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Format a tool result for the conversation.

    If the result contains an image_result, returns a list of content blocks
    with both text and image data. Otherwise returns a truncated string.
    """
    # check if any nested value is an image result (the Deno executor
    # wraps tool outputs, so the image_result may be inside "output")
    image_data = None
    text_parts: list[str] = []

    def _extract_image(obj: Any) -> None:
        nonlocal image_data
        if isinstance(obj, dict):
            if obj.get("type") == "image_result" and "image" in obj:
                image_data = obj["image"]
                if obj.get("text"):
                    text_parts.append(obj["text"])
            else:
                for v in obj.values():
                    _extract_image(v)
        elif isinstance(obj, list):
            for item in obj:
                _extract_image(item)

    _extract_image(result)

    if image_data and image_data.get("type") == "base64":
        blocks: list[dict[str, Any]] = []
        # add any text context
        non_image_str = str({k: v for k, v in result.items() if k != "image" and not (isinstance(v, dict) and v.get("type") == "image_result")})
        if non_image_str and non_image_str != "{}":
            if len(non_image_str) > MAX_TOOL_RESULT_LENGTH:
                non_image_str = non_image_str[:MAX_TOOL_RESULT_LENGTH] + "\n... (truncated)"
            blocks.append({"type": "text", "text": non_image_str})
        if text_parts:
            blocks.append({"type": "text", "text": "\n".join(text_parts)})
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_data.get("media_type", "image/png"),
                "data": image_data["data"],
            },
        })
        return blocks

    content_str = str(result)
    if len(content_str) > MAX_TOOL_RESULT_LENGTH:
        content_str = content_str[:MAX_TOOL_RESULT_LENGTH] + "\n... (truncated)"
    return content_str


def _timestamp_prefix(message: str) -> str:
    """prefix user mesages with a formatted timestamp for agent awareness"""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%a %b %d, %Y %I:%M%p %Z")
    return f"[{ts}] {message}"


class Agent:
    def __init__(
        self,
        model_api: Literal["anthropic", "openai", "openapi"],
        model_name: str,
        model_api_key: str | None,
        model_endpoint: str | None = None,
        tool_executor: ToolExecutor | None = None,
        max_iterations: int = 30,
        system_prompt: str | None = None,
        system_prompt_provider: Callable[[], Awaitable[str]] | None = None,
        sub_llm_client: Any | None = None,
        compact_threshold_chars: int = 240_000,
    ) -> None:
        match model_api:
            case "anthropic":
                if not model_api_key:
                    raise ValueError("model_api_key is required for anthropic")
                self._client: AgentClient = AnthropicClient(
                    api_key=model_api_key, model_name=model_name
                )
            case "openai":
                if not model_api_key:
                    raise ValueError("model_api_key is required for openai")
                self._client = OpenAICompatibleClient(
                    api_key=model_api_key,
                    model_name=model_name,
                    endpoint="https://api.openai.com/v1",
                )
            case "openapi":
                if not model_api_key:
                    raise ValueError("model_api_key is required for openapi")
                if not model_endpoint:
                    raise ValueError("model_endpoint is required for openapi")
                self._client = OpenAICompatibleClient(
                    api_key=model_api_key,
                    model_name=model_name,
                    endpoint=model_endpoint,
                )

        self._tool_executor = tool_executor
        self._max_iterations = max_iterations
        self._system_prompt = system_prompt
        self._system_prompt_provider = system_prompt_provider
        self._sub_llm_client = sub_llm_client
        self._compact_threshold_chars = compact_threshold_chars

    # -- public read-only properties for TUI/Discord --

    @property
    def client(self) -> AgentClient:
        """Return the underlying LLM client."""
        return self._client

    @property
    def system_prompt(self) -> str | None:
        """Return the current system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str | None) -> None:
        self._system_prompt = value

    @property
    def system_prompt_provider(self) -> Any:
        return self._system_prompt_provider

    @system_prompt_provider.setter
    def system_prompt_provider(self, value: Any) -> None:
        self._system_prompt_provider = value

    async def refresh_system_prompt(self) -> str:
        """Call the system prompt provider and update the cached prompt."""
        if self._system_prompt_provider is not None:
            self._system_prompt = await self._system_prompt_provider()
        return self._system_prompt or ""

    # number of messages from the end to preserve full tool results
    TOOL_RESULT_PRESERVE_COUNT = 4

    def _get_tools(self) -> list[dict[str, Any]] | None:
        """get tool definitions for the agent"""

        if self._tool_executor is None:
            return None

        return [self._tool_executor.get_execute_code_tool_definition()]

    async def _handle_tool_call(self, tool_use: AgentToolUseBlock) -> dict[str, Any]:
        """handle a tool call from the model"""
        if tool_use.name == "execute_code" and self._tool_executor:
            code = tool_use.input.get("code", "")
            result = await self._tool_executor.execute_code(code)
            return result
        else:
            return {"error": f"Unknown tool: {tool_use.name}"}

    def _repair_conversation(self, conversation: list[dict[str, Any]]) -> None:
        """Fix broken tool_use/tool_result pairs in conversation history.

        If a previous chat() call crashed mid-tool-execution, or concurrent
        calls interleaved messages, the conversation may contain assistant
        messages with tool_use blocks that lack matching tool_result responses.
        The Anthropic API rejects these. This method patches them up.
        """
        i = 0
        while i < len(conversation):
            msg = conversation[i]
            if msg["role"] != "assistant":
                i += 1
                continue

            content = msg.get("content", [])
            if not isinstance(content, list):
                i += 1
                continue

            tool_use_ids = [
                b["id"] for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            if not tool_use_ids:
                i += 1
                continue

            # collect tool_result ids from next message (if it exists and is a user message)
            existing_result_ids: set[str] = set()
            next_msg = conversation[i + 1] if i + 1 < len(conversation) else None
            if next_msg and next_msg["role"] == "user":
                next_content = next_msg.get("content", [])
                if isinstance(next_content, list):
                    existing_result_ids = {
                        b.get("tool_use_id")
                        for b in next_content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }

            missing = [tid for tid in tool_use_ids if tid not in existing_result_ids]
            if missing:
                logger.warning(
                    "Repairing %d orphaned tool_use block(s) at conversation index %d",
                    len(missing), i,
                )
                patch = [
                    {"type": "tool_result", "tool_use_id": tid, "content": "[result unavailable]"}
                    for tid in missing
                ]
                if existing_result_ids and next_msg and next_msg["role"] == "user":
                    # append missing results to existing tool_result message
                    next_content = next_msg.get("content", [])
                    if isinstance(next_content, list):
                        next_content.extend(patch)
                    i += 2
                else:
                    # insert a new tool_result message
                    conversation.insert(i + 1, {"role": "user", "content": patch})
                    i += 2
            else:
                i += 1

    def _trim_old_tool_results(self, conversation: list[dict[str, Any]]) -> None:
        """Replace verbose tool_result content with a short summary for older messages.

        Keeps the last TOOL_RESULT_PRESERVE_COUNT messages intact so the LLM
        has recent context, but shrinks older tool results to save tokens.
        """
        cutoff = len(conversation) - self.TOOL_RESULT_PRESERVE_COUNT
        if cutoff <= 0:
            return

        for i in range(cutoff):
            msg = conversation[i]
            if msg["role"] != "user":
                continue
            content = msg["content"]
            if not isinstance(content, list):
                continue
            changed = False
            new_content = []
            for block in content:
                if block.get("type") != "tool_result":
                    new_content.append(block)
                    continue
                raw = block.get("content", "")
                # skip already-trimmed results and image blocks
                if isinstance(raw, list) or (isinstance(raw, str) and raw.startswith("[tool result:")):
                    new_content.append(block)
                    continue
                size = len(str(raw))
                new_content.append({
                    "type": "tool_result",
                    "tool_use_id": block["tool_use_id"],
                    "content": f"[tool result: {size / 1024:.1f}KB]",
                })
                changed = True
            if changed:
                conversation[i] = {**msg, "content": new_content}

    @staticmethod
    def _accumulate_usage(total: dict[str, int], usage: dict[str, int]) -> None:
        for k, v in usage.items():
            total[k] = total.get(k, 0) + v

    async def _maybe_compact(self, conversation: list[dict[str, Any]]) -> None:
        """Summarize older conversation messages when the char budget is exceeded.

        Keeps the most recent ~25% of the budget as raw messages; replaces
        everything older with a single summary turn produced by the sub-agent.
        No-ops if no sub-agent LLM is wired in.
        """
        if self._sub_llm_client is None or not conversation:
            return

        total = sum(len(str(m.get("content", ""))) for m in conversation)
        if total <= self._compact_threshold_chars:
            return

        keep_chars = self._compact_threshold_chars // 4
        cumulative = 0
        split_idx = len(conversation)
        for i in range(len(conversation) - 1, -1, -1):
            cumulative += len(str(conversation[i].get("content", "")))
            if cumulative >= keep_chars:
                split_idx = i
                break

        older = conversation[:split_idx]
        recent = conversation[split_idx:]
        if not older:
            return

        older_text_lines: list[str] = []
        for m in older:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            parts.append(b.get("text", ""))
                        elif b.get("type") == "tool_use":
                            parts.append(f"[tool_use:{b.get('name', '?')}]")
                        elif b.get("type") == "tool_result":
                            r = str(b.get("content", ""))[:300]
                            parts.append(f"[tool_result]: {r}")
                content_str = " ".join(parts)
            else:
                content_str = str(content)
            if len(content_str) > 800:
                content_str = content_str[:800] + "..."
            older_text_lines.append(f"{role}: {content_str}")

        older_text = "\n".join(older_text_lines)
        if len(older_text) > 60_000:
            older_text = older_text[:60_000] + "\n... (truncated)"

        try:
            summary = await self._sub_llm_client.complete(
                (
                    "Summarize this prior conversation history for later recall. "
                    "Preserve key facts, decisions, identities, tool calls with their "
                    "outcomes, and any open threads or pending tasks. Output a dense "
                    "concise summary — no preamble.\n\n" + older_text
                ),
                system=(
                    "You are a conversation summarizer for an AI agent's memory. "
                    "Your output replaces the conversation prefix."
                ),
            )
        except Exception:
            logger.exception("compaction sub-agent call failed; keeping raw conversation")
            return

        logger.info(
            "Compacted conversation: %d messages (%d chars) -> summary (%d chars); "
            "kept %d recent messages",
            len(older),
            sum(len(str(m.get("content", ""))) for m in older),
            len(summary),
            len(recent),
        )
        # Use slice assignment so the caller's list is mutated in place
        # (a plain `conversation = ...` would only rebind the local).
        conversation[:] = [
            {
                "role": "user",
                "content": (
                    f"[Conversation summary of {len(older)} earlier messages]\n\n{summary}"
                ),
            }
        ] + recent

    async def chat(
        self,
        user_message: str,
        conversation: list[dict[str, Any]],
        on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
        images: list[dict[str, str]] | None = None,
    ) -> str:
        """Send a message and get a response, handling tool calls.

        `conversation` is a list owned by the caller and mutated in place —
        each surface (TUI, Discord, scheduler) maintains its own conversation
        list and passes it here. The agent does not hold shared conversation
        state.

        When `images` is provided, the current user turn is sent as a multi-
        block message with images first, then text. Each image dict should be
        `{"media_type": "image/png", "data": "<base64>"}`. Images are ephemeral
        — they only affect this turn; nothing is persisted to the conversation
        store for them.
        """

        async def _emit(event: AgentEvent) -> None:
            if on_event is not None:
                await on_event(event)

        return await self._chat_impl(user_message, conversation, _emit, images=images)

    async def _chat_impl(
        self,
        user_message: str,
        conversation: list[dict[str, Any]],
        _emit: Callable[[AgentEvent], Awaitable[None]],
        images: list[dict[str, str]] | None = None,
    ) -> str:
        """Inner chat loop. `conversation` is the caller-owned list that
        will be mutated in place. Do not call directly — go through `chat()`."""
        # refresh the system prompt before each operator turn so that newly
        # added secrets, approved custom tools, self-note edits, and skills
        # show up without restarting the harness.
        if self._system_prompt_provider is not None:
            try:
                self._system_prompt = await self._system_prompt_provider()
            except Exception:
                logger.warning(
                    "system prompt refresh failed; using cached version",
                    exc_info=True,
                )

        # compact the conversation if it's gotten huge
        try:
            await self._maybe_compact(conversation)
        except Exception:
            logger.exception("compaction failed; continuing with raw conversation")

        timestamped = _timestamp_prefix(user_message)
        if images:
            # Multi-block content: images first, text last (per Gemma 4 docs,
            # and Anthropic handles either order). Uses Anthropic's source
            # schema; the OpenAI-compat client converts on the way out.
            content_blocks: list[dict[str, Any]] = []
            for img in images:
                content_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.get("media_type", "image/png"),
                            "data": img.get("data", ""),
                        },
                    }
                )
            content_blocks.append({"type": "text", "text": timestamped})
            conversation.append({"role": "user", "content": content_blocks})
        else:
            conversation.append({"role": "user", "content": timestamped})

        total_usage: dict[str, int] = {}
        iteration = 0
        while iteration < self._max_iterations:
            iteration += 1

            self._repair_conversation(conversation)
            self._trim_old_tool_results(conversation)

            await _emit(AgentEvent(kind="llm_start"))
            resp = await self._client.complete(
                messages=conversation,
                system=self._system_prompt,
                tools=self._get_tools(),
            )
            if resp.usage:
                self._accumulate_usage(total_usage, resp.usage)
            await _emit(AgentEvent(
                kind="llm_done",
                data={"usage": resp.usage} if resp.usage else {},
            ))

            assistant_content: list[dict[str, Any]] = []
            text_response = ""

            for block in resp.content:
                if isinstance(block, AgentTextBlock):
                    assistant_content.append({"type": "text", "text": block.text})
                    text_response += block.text
                elif isinstance(block, AgentToolUseBlock):  # type: ignore
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }
            if resp.reasoning_content:
                assistant_msg["reasoning_content"] = resp.reasoning_content
            conversation.append(assistant_msg)

            # find any tool calls that we need to handle
            if resp.stop_reason == "tool_use":
                tool_results: list[dict[str, Any]] = []
                for block in resp.content:
                    if isinstance(block, AgentToolUseBlock):
                        code = block.input.get("code", "")
                        logger.info("Tool call: %s\n%s", block.name, code)

                        await _emit(AgentEvent(
                            kind="tool_start",
                            data={"tool": block.name, "code": code},
                        ))
                        result = await self._handle_tool_call(block)
                        is_error = "error" in result
                        summary = str(result)[:500]
                        logger.info(
                            "Tool result (%s): %s",
                            "error" if is_error else "ok",
                            summary,
                        )
                        await _emit(AgentEvent(
                            kind="tool_done",
                            data={
                                "tool": block.name,
                                "success": not is_error,
                                "result": result,
                            },
                        ))

                        content = _format_tool_result(result)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": content,
                            }
                        )

                conversation.append({"role": "user", "content": tool_results})
            else:
                # once there are no more tool calls, we proceed to the text response
                if total_usage:
                    logger.info("Chat total usage: %s", total_usage)
                return text_response

        # hit max iterations — one final LLM call without tools
        logger.warning(
            "Hit max iterations (%d), forcing final response", self._max_iterations
        )
        conversation.append(
            {
                "role": "user",
                "content": "[System: Max tool calls reached. Provide final response now.]",
            }
        )
        self._repair_conversation(conversation)
        self._trim_old_tool_results(conversation)
        await _emit(AgentEvent(kind="llm_start"))
        resp = await self._client.complete(
            messages=conversation,
            system=self._system_prompt,
            tools=None,  # no tools for final call
        )
        if resp.usage:
            self._accumulate_usage(total_usage, resp.usage)
        await _emit(AgentEvent(
            kind="llm_done",
            data={"usage": resp.usage} if resp.usage else {},
        ))
        text_response = ""
        for block in resp.content:
            if isinstance(block, AgentTextBlock):
                text_response += block.text
        if total_usage:
            logger.info("Chat total usage: %s", total_usage)
        return text_response
