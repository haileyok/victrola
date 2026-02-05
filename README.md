# Victrola

> [!WARNING]
> No guarantees about security are made. I have reasonable confidence that things like secrets cannot be accessed by the agent (unless, of course, you allow it access through one of its custom tools...), but again make no guarantees. Don't do stupid things with agents. Don't give agents access to sensitive information. Do we really need to warn about anything else?
>
> You are also trusting [the Deno sandbox](https://docs.deno.com/runtime/fundamentals/security/). For a truly hardened setup, you should absolutely be executing code in something like Firecracker VMs, not Deno alone.

A general-purpose personal AI agent harness. You chat with it through a TUI or through Discord. It writes and runs code to do tasks for you!

Victrola is a single-operator agent runtime. You (the operator) drive it through chat. The agent has persistent memory, a scheduler, and can write its own tools in TypeScript that run in a sandboxed Deno process. You can also pre-write your own tools in Python using a decorator.

Victrola supports Anthropic, OpenAI, or any OpenAPI compatible endpoint like Ollama or llama-cpp.


## Tool Calling

The agent interacts with its tools via a single `execute_code` primitive. It writes TypeScript that calls `tools.namespace.method(...)`, which round-trips to Python handlers. This lets the agent chain multiple tool calls in one turn instead of paying a round trip per call (see [Cloudflare's "code mode" post](https://blog.cloudflare.com/code-mode/) for the rationale).

Deno runs with the bare minimum of permissions: no filesystem writes, no network, no env access, 256 MB V8 heap, 60 s timeout, max 25 inner tool calls per execution. All actual network / storage work happens in Python.

## Built-in tools

| Namespace | Tool | What it does |
|-----------|------|---|
| `notes` | `note_upsert`, `note_get`, `note_list` | Persistent memory. `self` note holds the agent's personality; `operator` note holds what it knows about you; `skill:*` holds reusable procedures. |
| `scheduler` | `list_schedules`, `get_schedule` | View scheduled tasks. Creation is via the TUI. |
| `notify` | `discord` | Send a message to Discord via webhook (requires `DISCORD_WEBHOOK_URL` secret). |
| `summarize` | `summarize` | Summarize text using the sub-agent model. |
| `web` | search tools via [Exa](https://exa.ai) (requires `EXA_API_KEY`). |
| `image` | `view_image` | Fetch an image URL and include it inline. |
| `custom_tools` | `create_custom_tool`, `call_tool`, etc. | Agent-written tools (see below). |

## Writing your own tools

Drop a file in `src/tools/definitions/`, decorate an async function, and add it to `src/tools/definitions/__init__.py`. Example:

```python
# src/tools/definitions/weather.py
from typing import Any
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="weather.current",
    description="Look up the current weather for a city.",
    parameters=[
        ToolParameter(name="city", type="string", description="City name"),
    ],
)
async def current(ctx: ToolContext, city: str) -> dict[str, Any]:
    resp = await ctx.http_client.get(
        "https://wttr.in/" + city, params={"format": "j1"}
    )
    return resp.json()
```

The harness will generate a TypeScript stub and add the tool to the agent's system prompt.

## Agent-written custom tools

The agent can propose new tools by calling `custom_tools.create_custom_tool` during a chat turn. These are stored as pending and **do not appear in the agent's tool list until you approve them.** Tools can be reviewed via the TUI:

- From a chat session, press `T` if there's a pending-tools banner.
- From the session list, press `t`.
- Select a tool and press Enter to view its code + requested secrets.
- `a` to approve, `r` to revoke, `d` to delete.

If a tool references a secret that isn't configured yet, the approval flow walks you through setting each missing secret before completing approval.

## Storage

Everything persistent lives under `./data/`:
- `store.db` — SQLite: notes, skills, chat sessions + messages, custom tool definitions, namespaced records
- `secrets.json` — named secrets (injected as env vars into custom tool Deno processes)
- `schedules.json` — scheduled prompts

Nothing leaves this directory unless you wire a tool to send it somewhere.

## Prerequisites

- [Deno](https://deno.com/) runtime
- [uv](https://github.com/astral-sh/uv) package manager
- A model API key (Anthropic) or a local inference endpoint (Ollama, etc.)

## Install

```bash
git clone https://github.com/haileyok/victrola.git
cd victrola
uv sync --frozen
```

## Configuration

Copy `.env.example` to `.env` and fill in. Minimal setup for Anthropic:

```env
MODEL_API=anthropic
MODEL_API_KEY=sk-ant-...
MODEL_NAME=claude-sonnet-4-5-20250929
```

For local Ollama (Gemma 4, etc.):

```env
MODEL_API=openapi
MODEL_ENDPOINT=http://localhost:11434/v1
MODEL_NAME=gemma4:26b-moe
MODEL_API_KEY=ollama      # Ollama ignores the value; just can't be empty
```

Sub-agent (used by `summarize` and anything else that needs a lighter model) defaults to the same key as the main model if left empty:

```env
SUB_MODEL_API=anthropic
SUB_MODEL_NAME=claude-haiku-4-5-20251001
```

## Usage

```bash
uv run python main.py tui
uv run python main.py main  # headless; runs the scheduler + Discord bot only
```

All commands accept `--model-api`, `--model-name`, `--model-api-key`, `--model-endpoint` to override config at launch.

## Discord chat (optional)

You can chat with the agent from Discord in addition to the TUI. Each thread in a dedicated channel is a chat session.

**Setup (one-time):**

1. [discord.com/developers/applications](https://discord.com/developers/applications) → New Application → "Victrola"
2. **Bot tab** → Reset Token → copy (save as `DISCORD_BOT_TOKEN` secret in the TUI)
3. **Bot tab → Privileged Gateway Intents** → enable **"Message Content Intent"** (required — otherwise the bot can't read message text)
4. **OAuth2 → URL Generator:**
   - Scopes: `bot`
   - Bot Permissions (principle of least privilege):
     - View Channel
     - Read Message History
     - Create Public Threads
     - Send Messages in Threads
   - Visit the generated URL → pick your server → Authorize
5. In your server, create a text channel (default name `victrola-sessions` — configurable via `DISCORD_SESSIONS_CHANNEL` in `.env`)

**Usage:**

- Post a top-level message in the channel → the bot creates a thread *from* that message and the agent responds inside.
- Or create a thread yourself (with any first message) → the agent responds in the thread.
- Reply in an existing thread to continue that session.
- Only the agent's final text response appears in the thread. Tool activity (TypeScript the agent writes + tool results) is hidden from Discord to keep threads readable — use the TUI if you want to review that.

The bot only starts when `DISCORD_BOT_TOKEN` is configured. Without it the TUI and scheduler still run normally.
