# Victrola

> [!WARNING]
> No guarantees about security are made. I have reasonable confidence that things like secrets cannot be accessed by the agent (unless, of course, you allow it access through one of its custom tools...), but again make no guarantees. Don't do stupid things with agents. Don't give agents access to sensitive information. Do we really need to warn about anything else?
>
> You are also trusting [the Deno sandbox](https://docs.deno.com/runtime/fundamentals/security/). For a truly hardened setup, you should absolutely be executing code in something like Firecracker VMs, not Deno alone.

A general-purpose personal AI agent harness. You chat with it through a local web interface or through Discord. It writes and runs code to do tasks for you!

Victrola is a single-operator agent runtime. You (the operator) drive it through chat. The agent has persistent memory, a scheduler, and can write its own tools in TypeScript that run in a sandboxed Deno process. You can also pre-write your own tools in Python using a decorator.

Victrola supports Anthropic, OpenAI, or any OpenAPI compatible endpoint like Ollama or llama-cpp.


## Tool Calling

The agent interacts with its tools via a single `execute_code` primitive. It writes TypeScript that calls `tools.namespace.method(...)`, which round-trips to Python handlers. This lets the agent chain multiple tool calls in one turn instead of paying a round trip per call (see [Cloudflare's "code mode" post](https://blog.cloudflare.com/code-mode/) for the rationale).

Deno runs with the bare minimum of permissions: scoped filesystem access limited to the workspace directory, no network, no env access (except explicitly declared secrets in custom tools), 256 MB V8 heap, 60 s timeout, max 25 inner tool calls per execution. All actual network / storage work happens in Python.

## Built-in tools

| Namespace | Tool | What it does |
|-----------|------|---|
| `memory` | `add`, `update`, `delete`, `search`, `get`, `list_skills`, `get_skill` | Entry-based memory with hybrid search. `self` holds the agent's personality; `operator` holds facts about you; `skill:*` holds reusable procedures; `episodic` and `factual` are RAG-retrieved per turn. |
| `scheduler` | `list_schedules`, `get_schedule` | View scheduled tasks. Creation is via the web interface or `scheduler.create_schedule`. |
| `notify` | `discord` | Send a message to Discord via webhook (requires `DISCORD_WEBHOOK_URL` secret). |
| `summarize` | `summarize` | Summarize text using the sub-agent model. |
| `web` | search tools via [Exa](https://exa.ai) (requires `EXA_API_KEY`). |
| `image` | `view_image` | Fetch an image URL and include it inline. |
| `custom_tools` | `create_custom_tool`, `call_tool`, etc. | Agent-written tools (see below). |
| `system` | `get_tool_docs` | Fetch full parameter docs for any tool (typically MCP tools, which are listed in a compact catalog in the system prompt). |

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

The agent can propose new tools by calling `custom_tools.create_custom_tool` during a chat turn. These are stored as pending and **do not appear in the agent's tool list until you approve them.** Tools can be reviewed via the web interface:

- Navigate to the Tools page from the sidebar.
- Click a tool to view its code, parameters, and requested secrets.
- Approve, revoke, or delete from the detail view.

If a tool references a secret that isn't configured yet, the approval flow walks you through setting each missing secret before completing approval.

## Scheduling and Triggers

Scheduled tasks fire a prompt on a recurring schedule (hourly, daily, weekly, cron). When a schedule fires, the agent runs the prompt in a fresh conversation with full tool access. Tasks are stored in SQLite (`store.db`).

### Triggers (conditional schedules)

A scheduled task can optionally have a **condition script** — TypeScript that runs on schedule *before* waking the agent. This avoids spending tokens on recurring checks that usually find nothing to act on.

- The condition script calls `output({ wake: true })` to wake the agent, or `output({ wake: false })` to skip this cycle.
- Condition code runs in the same Deno sandbox as custom tools, with optional network access and secrets.
- Condition code requires **operator approval** before it will fire (same gate as custom tools). Until approved, the task skips silently.
- If a condition script fails 3 consecutive times, the task is auto-disabled.
- The operator can test condition code via the web interface before approving.

Tasks without a condition script behave exactly as before — they always wake the agent on schedule.

### One-time migration

On startup, existing `schedules.json` entries are migrated to SQLite and the file is renamed to `schedules.json.migrated`. This is idempotent — re-running with a stale JSON file does not error or duplicate.

## Memory

The agent has an entry-based memory system with hybrid search and RAG recall. All memory lives in SQLite (`memory_entries` table).

### Memory types

| Type | Structure | Consumption |
|------|-----------|-------------|
| `self` | Single entry | Always in system prompt |
| `operator` | Multi-entry (individual facts) | All entries in system prompt every turn |
| `skill` | Single entry per skill | Name + preview in prompt; full content loaded on demand |
| `episodic` | Multi-entry | RAG-retrieved per turn based on relevance |
| `factual` | Multi-entry | RAG-retrieved per turn based on relevance |

### Search

`memory.search` combines FTS5 keyword matching with vector cosine similarity over embeddings (via local Ollama). If either subsystem is unavailable, it degrades gracefully to keyword-only or semantic-only.

### RAG recall

Before each turn, the user's message is embedded and relevant `episodic` + `factual` entries are retrieved and injected into the system prompt as a `# Relevant Memories` section. This is transient — it's not persisted or visible in the static system prompt viewer.

### Embeddings

Embeddings are generated via a local Ollama instance using `nomic-embed-text` (768 dimensions). If Ollama isn't running, memory writes store NULL embeddings and searches fall back to keyword-only. Embeddings are backfilled automatically when Ollama becomes available.

### Web UI

Memory entries can also be managed through the web interface at `/memory`, which provides browse, search, create, edit, and delete alongside the `memory.*` agent tools.

## Workspace

The agent has a persistent workspace directory (default: `data/workspace/`) where it can read and write files using Deno's native file APIs. This lets the agent build multi-file projects, create shared libraries, test code before deploying it as a custom tool, and generate artifacts.

The workspace path is injected into every `execute_code` and custom tool block as the `WORKSPACE` TypeScript constant. The agent uses `Deno.writeTextFile`, `Deno.readTextFile`, `Deno.readDir`, and other native APIs to interact with files. Modules in the workspace can be dynamically imported via `await import(WORKSPACE + "/lib/parsers.ts")`.

### Security model

- Workspace access is scoped via Deno's `--allow-write=<workspace>` and `--allow-read=<workspace>` — the agent cannot write outside this directory
- Symlink creation is not permitted
- The workspace is visible to the operator via the web interface at `/workspace`
- No npm packages — the agent vendors TypeScript source manually if needed
- A configurable size limit monitors for disk exhaustion (soft limit with warning)

### Configuration

```env
WORKSPACE_DIR=data/workspace
WORKSPACE_MAX_SIZE_MB=1024
```

### Saving web pages as PDFs (`web.save_url_as_pdf`)

The agent can turn a URL into a PDF in its workspace: if the URL is already a
PDF it's downloaded; otherwise the page is rendered with a headless Chromium
(Playwright). It then reads the file from the workspace and can hand it to an
upload tool.

**Deploy (Linux):** `playwright` is a dependency; install the browser once:

```bash
playwright install --with-deps chromium
```

If Chromium won't launch on a locked-down host, set
`WEB_PDF_CHROMIUM_NO_SANDBOX=1` (prefer running victrola as a non-root user / in
a container over disabling Chromium's sandbox).

**SSRF — required egress filtering.** Only public addresses are allowed. The
*download* path is pinned in code (rebinding-safe). The *render* path uses a
real browser that does its own DNS/redirects, so its authoritative protection is
**host egress filtering**: the victrola process MUST be denied outbound access
to loopback, link-local (169.254/16, incl. cloud metadata), RFC1918,
CGNAT/tailnet (100.64/10), and unique-local ranges. Without that the render path
is rebinding-exploitable. An nftables example is in the `web_pdf.py` module
docstring.

> End-to-end "send a web page to my device" also needs an upload tool (e.g. a
> registered Supernote MCP server) for the agent to call after the PDF is saved.

## Storage

Everything persistent lives under `./data/`:
- `store.db` — SQLite: memory entries (+ FTS5 index + embeddings), chat sessions + messages, custom tool definitions, scheduled tasks, namespaced records
- `secrets.json` — named secrets (injected as env vars into custom tool Deno processes)
- `schedules.json.migrated` — renamed after one-time migration to SQLite (kept as backup)
- `workspace/` — agent file storage (read/write scoped to Deno sandbox)

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

## Umans AI (optional)

[Umans AI](https://app.umans.ai) is a subscription-based inference provider with an Anthropic-compatible endpoint. To use it:

```env
MODEL_API=umans
MODEL_API_KEY=sk-...
MODEL_NAME=umans-glm-5.2
```

Available models include `umans-glm-5.2`, `umans-kimi-k2.7`, `umans-coder`, and `umans-flash`.

### Server-side web search

Umans can run web search server-side via the `X-Umans-Websearch-Provider` header. Set `UMANS_WEBSEARCH_PROVIDER` to `exa`, `native`, or `none` (default):

```env
UMANS_WEBSEARCH_PROVIDER=exa
```

When set to `exa` or `native`, a `web_search` tool is sent to the model alongside `execute_code`. Umans intercepts it server-side on models that support the Umans-owned search step. On models where the header is a no-op, the tool falls back to the local `exa-py` client (requires `EXA_API_KEY`).

When set to `none` (default), web search works through the existing `execute_code` → Deno sandbox → `exa-py` path.

### Sub-agent via Umans

```env
SUB_MODEL_API=umans
SUB_MODEL_NAME=umans-flash
SUB_MODEL_API_KEY=sk-...
```

The sub-agent endpoint defaults to `https://api.code.umans.ai` when `SUB_MODEL_API=umans` and no explicit `SUB_MODEL_ENDPOINT` is set.

### Custom endpoint

Override the Umans API base URL if needed:

```env
UMANS_ENDPOINT=https://api.code.umans.ai
```

## Embeddings (optional)

Embeddings (for memory search and RAG recall) use a local Ollama instance:

```env
EMBEDDING_ENDPOINT=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
```

If Ollama isn't running, memory writes store NULL embeddings and searches fall back to keyword-only.

## Usage

```bash
uv run python main.py serve  # web interface at http://localhost:8000
uv run python main.py main   # headless; runs the scheduler + Discord bot only
```

All commands accept `--model-api`, `--model-name`, `--model-api-key`, `--model-endpoint` to override config at launch.

### Development

For frontend development, run the backend and Vite dev server simultaneously:

```bash
# terminal 1 — backend
uv run python main.py serve

# terminal 2 — frontend dev server (hot reload, proxies /api to :8000)
cd web && npm run dev
```

The Vite dev server runs at `http://localhost:5173` with API requests proxied to the backend on port 8000. For production, build the frontend (`cd web && npm run build`) and it will be served directly by FastAPI at `http://localhost:8000`.

## Discord chat (optional)

You can chat with the agent from Discord in addition to the web interface. Each thread in a dedicated channel is a chat session.

**Setup (one-time):**

1. [discord.com/developers/applications](https://discord.com/developers/applications) → New Application → "Victrola"
2. **Bot tab** → Reset Token → copy (save as `DISCORD_BOT_TOKEN` secret in the web interface)
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
- Only the agent's final text response appears in the thread. Tool activity (TypeScript the agent writes + tool results) is hidden from Discord to keep threads readable — use the web interface if you want to review that.

The bot only starts when `DISCORD_BOT_TOKEN` is configured. Without it the web interface and scheduler still run normally.

## Signal chat (optional)

You can chat with the agent from Signal in addition to the web interface. Signal becomes the default notification channel when configured — scheduled task results and `notify.send` calls route there automatically.

**Requires `signal-cli-rest-api`** running as an external service. This is a REST wrapper around signal-cli that victrola polls for incoming messages and uses to send responses.

**Setup:**

1. Run signal-cli-rest-api via Docker:
   ```bash
   docker run -d \
     -e MODE=normal \
     -p 8080:8080 \
     -v ./signal-cli-config:/home/.local/share/signal-cli \
     bbernhard/signal-cli-rest-api:latest
   ```
   **Do NOT set `AUTO_RECEIVE_SCHEDULE`** — the bot polls `/v1/receive` itself, and the auto-receive schedule would consume messages out from under it, causing permanent message loss.

2. Register a new Signal account (or link to an existing one via QR code). A dedicated phone number for the bot is simplest — see the signal-cli-rest-api docs for registration/linking details.

3. Configure `.env`:
   ```env
   SIGNAL_SERVICE=127.0.0.1:8080
   SIGNAL_BOT_PHONE=+1234567890
   SIGNAL_OPERATOR_PHONE=+0987654321
   ```
   `SIGNAL_OPERATOR_PHONE` can be a phone number or a Signal UUID — username-only accounts are matched via `sourceUuid`.

4. Restart victrola. The bot polls for messages every ~2 seconds and maintains a single persistent chat session (`signal-persistent`) that survives restarts.

**Usage:**

- Send a Signal message to the bot's number — the agent processes it and responds via Signal.
- The session is persistent: conversation history and compaction state survive restarts.
- Long responses are chunked into multiple messages at ~1900 chars.
- Image attachments are supported (ephemeral — not persisted to the conversation store).
- The agent's `notify.signal` and `notify.send` tools work independently of the chat loop, so you can send notifications even when the bot isn't running.

## Compaction (optional)

When a conversation grows too large, older messages are summarized by the sub-agent into a single summary turn. The most recent ~25% of the threshold is kept as raw messages; everything older is replaced with the summary. Compaction checkpoints are persisted to the store, so summaries are reused on reload instead of re-summarizing.

The threshold is configurable in `.env`:

```env
COMPACT_THRESHOLD_CHARS=240000   # ~60k tokens at ~4 chars/token
```

Requires a sub-agent LLM (`SUB_MODEL_*` config). If no sub-agent is configured, compaction is a no-op and conversations grow unbounded.
