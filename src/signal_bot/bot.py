"""Signal bot giving the operator a persistent chat surface via Signal.

Polls signal-cli-rest-api for incoming messages, routes them through the
agent, and sends responses back via Signal. Uses a single persistent chat
session keyed by a fixed rkey (``signal-persistent``).

Requires signal-cli-rest-api running as an external service (typically
Docker). The operator sets it up, registers or links a Signal account,
and configures victrola with the service address and phone numbers.

Important: signal-cli-rest-api's /v1/receive endpoint is DESTRUCTIVE —
it fetches and consumes messages from the Signal server. Each message
is returned exactly once, so no dedup logic is needed. Do NOT set
AUTO_RECEIVE_SCHEDULE in the Docker container, or it will consume
messages out from under this bot.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from src.config import CONFIG

if TYPE_CHECKING:
    from src.agent.agent import Agent
    from src.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)

# Signal message body cap (similar to Discord's 2000 char limit)
MAX_CHUNK = 1900
POLL_INTERVAL_SECONDS = 2.0
TYPING_REFRESH_SECONDS = 3.0


class SignalBot:
    """Long-running Signal bot that polls signal-cli-rest-api for messages.

    Polls GET /v1/receive/{phone} every ~2 seconds. Sends responses via
    POST /v2/send with the bot's number in the body. Single persistent
    chat session with the operator.
    """

    def __init__(
        self,
        signal_service: str,
        bot_phone: str,
        operator_phone: str,
        agent: "Agent",
        executor: "ToolExecutor",
    ) -> None:
        self._agent = agent
        self._executor = executor
        self._bot_phone = bot_phone
        self._operator_phone = operator_phone
        self._signal_service = signal_service
        self._session_rkey = CONFIG.signal_session_rkey
        self._session_lock = asyncio.Lock()
        self._stopped = asyncio.Event()

    @property
    def _base_url(self) -> str:
        return f"http://{self._signal_service}"

    @property
    def _http_client(self) -> httpx.AsyncClient:
        return self._executor.ctx.http_client

    def _send_url(self) -> str:
        return f"{self._base_url}/v2/send"

    def _receive_url(self) -> str:
        # send_read_receipts=true marks messages as read as they're consumed
        # from the destructive receive endpoint. This applies to all fetched
        # messages before sender validation — non-operator messages are also
        # marked read, since the API doesn't support per-message receipts.
        return (
            f"{self._base_url}/v1/receive/{quote(self._bot_phone, safe='')}"
            "?send_read_receipts=true"
        )

    def _typing_url(self) -> str:
        return f"{self._base_url}/v1/typing-indicator/{quote(self._bot_phone, safe='')}"

    async def _start_typing(self) -> None:
        """Send a typing indicator to the operator."""
        try:
            await self._http_client.request(
                "PUT",
                self._typing_url(),
                json={"recipient": self._operator_phone},
                timeout=5.0,
            )
        except Exception:
            logger.debug("Failed to send typing indicator", exc_info=True)

    async def _stop_typing(self) -> None:
        """Clear the typing indicator."""
        try:
            await self._http_client.request(
                "DELETE",
                self._typing_url(),
                json={"recipient": self._operator_phone},
                timeout=5.0,
            )
        except Exception:
            logger.debug("Failed to stop typing indicator", exc_info=True)

    async def _typing_loop(self, stop: asyncio.Event) -> None:
        """Refresh the typing indicator periodically until stop is set."""
        while not stop.is_set():
            await self._start_typing()
            try:
                await asyncio.wait_for(stop.wait(), timeout=TYPING_REFRESH_SECONDS)
            except asyncio.TimeoutError:
                pass

    async def start(self) -> None:
        """Poll signal-cli-rest-api for incoming messages until close() is called."""
        logger.info(
            "Signal bot started, polling %s for messages to %s",
            self._signal_service,
            self._bot_phone,
        )
        while not self._stopped.is_set():
            try:
                await self._poll_once()
            except Exception:
                logger.exception("Signal poll failed; will retry")
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=POLL_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass  # normal — timeout means keep polling

    async def close(self) -> None:
        self._stopped.set()

    async def _poll_once(self) -> None:
        """Fetch new messages from signal-cli-rest-api and dispatch them.

        Each message is handled in its own try/except so one failing message
        doesn't abandon the rest of the batch — since /v1/receive is
        destructive, abandoned messages are permanently lost.
        """
        try:
            resp = await self._http_client.get(
                self._receive_url(),
                timeout=10.0,
            )
        except httpx.HTTPError as e:
            logger.warning("Signal receive request failed: %s", e)
            return

        if resp.status_code != 200:
            logger.warning("Signal receive returned HTTP %d", resp.status_code)
            return

        try:
            messages = resp.json()
        except (ValueError, TypeError) as e:
            logger.error(
                "Signal receive returned malformed JSON (messages may be lost): %s", e
            )
            return

        if not isinstance(messages, list):
            return

        for msg in messages:
            try:
                await self._handle_message(msg)
            except Exception:
                logger.exception("Failed to handle Signal message; continuing to next")

    async def _handle_message(self, msg: dict) -> None:
        """Parse a signal-cli-rest-api message and route to the agent if it's
        from the operator."""
        # signal-cli-rest-api wraps messages in an envelope structure
        envelope = msg.get("envelope", msg)

        # Match the operator by any available identifier. For phone-based
        # accounts `source` is the E.164 number; for username-only accounts
        # `source` is a UUID and `sourceNumber` is null. Check all three
        # fields so the operator can be identified by either.
        sender_ids = {
            envelope.get("source", ""),
            envelope.get("sourceNumber", ""),
            envelope.get("sourceUuid", ""),
        }
        sender_ids.discard(None)
        sender_ids.discard("")

        if self._operator_phone not in sender_ids:
            return

        # The text is always in dataMessage regardless of envelope type
        data_msg = envelope.get("dataMessage", {})
        if not data_msg:
            return

        # Extract text
        user_text = data_msg.get("message", "")

        # Extract image attachments
        images = await self._extract_attachments(data_msg)

        if not user_text and not images:
            return

        await self._run_agent(user_text, images)

    async def _extract_attachments(self, data_msg: dict) -> list[dict[str, str]]:
        """Download image attachments from signal-cli-rest-api as base64 blobs."""
        attachments = data_msg.get("attachments", [])
        if not attachments:
            return []

        images: list[dict[str, str]] = []
        for att in attachments:
            # signal-cli-rest-api returns attachment metadata with an id;
            # download via GET /v1/attachments/{id}
            att_id = att.get("id")
            content_type = att.get("contentType", "").lower()
            if not content_type.startswith("image/"):
                continue
            if not att_id:
                continue
            try:
                resp = await self._http_client.get(
                    f"{self._base_url}/v1/attachments/{att_id}",
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    images.append(
                        {
                            "media_type": content_type,
                            "data": base64.b64encode(resp.content).decode("ascii"),
                        }
                    )
            except Exception:
                logger.exception("Failed to download Signal attachment %s", att_id)

        return images

    async def _run_agent(self, user_text: str, images: list[dict[str, str]]) -> None:
        """Route an operator message into the persistent session and run the agent."""
        async with self._session_lock:
            store = self._executor.store
            if store is None or store.chat is None:
                logger.error("Store not initialized for Signal bot")
                return

            # Ensure session exists
            await store.chat.ensure_session(rkey=self._session_rkey, title="Signal")

            # Save user message
            await store.chat.create_message(
                session_id=self._session_rkey,
                sender="user",
                content=json.dumps({"role": "user", "content": user_text}, default=str),
            )

            # Load conversation with IDs for compaction checkpoint tracking
            from src.agent.conversation import ConversationManager

            conv_manager = ConversationManager(
                ctx=self._executor.ctx, llm_client=self._executor.llm_client
            )
            messages, msg_ids = await conv_manager.load_session_with_ids(self._session_rkey)

            # Define on_compact callback — persists the checkpoint so the
            # summary is reused on reload instead of re-summarizing.
            async def on_compact(summary: str, split_idx: int) -> None:
                if 0 < split_idx <= len(msg_ids):
                    last_id = msg_ids[split_idx - 1]
                    if last_id >= 0:
                        await store.chat.set_compaction_checkpoint(
                            self._session_rkey, last_id, summary
                        )

            # Run agent with a typing indicator while it thinks
            typing_stop = asyncio.Event()
            typing_task = asyncio.create_task(self._typing_loop(typing_stop))
            try:
                response = await self._agent.chat(
                    user_text,
                    conversation=messages,
                    on_compact=on_compact,
                    images=images or None,
                )
            finally:
                typing_stop.set()
                with contextlib.suppress(asyncio.CancelledError):
                    await typing_task
                await self._stop_typing()

            # Save assistant response
            if response:
                await store.chat.create_message(
                    session_id=self._session_rkey,
                    sender="assistant",
                    content=json.dumps(
                        {"role": "assistant", "content": response}, default=str
                    ),
                )
                await self._send_response(response)
            else:
                await self._send_response("(empty response)")

    async def _send_response(self, text: str) -> None:
        """Send a message to the operator via signal-cli-rest-api, chunking if needed."""
        from src.utils.text import _chunk

        for chunk in _chunk(text, limit=MAX_CHUNK):
            try:
                resp = await self._http_client.post(
                    self._send_url(),
                    json={
                        "message": chunk,
                        "number": self._bot_phone,
                        "recipients": [self._operator_phone],
                    },
                    timeout=10.0,
                )
                if resp.status_code >= 400:
                    logger.error(
                        "Signal send failed: HTTP %d — %s",
                        resp.status_code,
                        resp.text[:200],
                    )
                    return
            except Exception:
                logger.exception("Failed to send Signal message")
                return
