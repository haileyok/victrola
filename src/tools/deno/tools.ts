// Auto-generated - do not edit
import { callTool } from "./runtime.ts";

export const custom_tools = {
  /** Execute an approved custom tool by name with the given parameters. */
  call_tool: (name: string, params: Record<string, unknown>): Promise<unknown> => callTool("custom_tools.call_tool", { name, params }),

  /** Create a new custom tool. The tool is stored locally and must be approved by the operator before it appears in the LLM's tool list.

The `parameters` field must be a valid JSON Schema object describing the tool's input parameters.
The `code` field is TypeScript source that will run in Deno. It receives a `params` object with the input parameters and has access to `output()` and `debug()` functions, plus the `tools` namespace.

Use `list_available_secrets` to see what secrets the operator has configured. Reference them by name in the `secrets` field — they'll be injected as env vars (accessed via `Deno.env.get("SECRET_NAME")`). */
  create_custom_tool: (name: string, description: string, parameters: Record<string, unknown>, code: string, response_schema?: Record<string, unknown>, secrets?: unknown[]): Promise<unknown> => callTool("custom_tools.create_custom_tool", { name, description, parameters, code, response_schema, secrets }),

  /** Delete a custom tool. */
  delete_custom_tool: (name: string): Promise<unknown> => callTool("custom_tools.delete_custom_tool", { name }),

  /** Get full details of a custom tool including its code. */
  get_custom_tool: (name: string): Promise<unknown> => callTool("custom_tools.get_custom_tool", { name }),

  /** List the names of secrets that have been configured by the human operator. These can be referenced in the `secrets` field when creating custom tools. Secret values are never exposed — only names are returned. */
  list_available_secrets: (): Promise<unknown> => callTool("custom_tools.list_available_secrets", {}),

  /** List all custom tools with their name, description, and approval status. */
  list_custom_tools: (): Promise<unknown> => callTool("custom_tools.list_custom_tools", {}),

  /** Update an existing custom tool. If code or parameters change, approval will be reset. */
  update_custom_tool: (name: string, description?: string, parameters?: Record<string, unknown>, code?: string, response_schema?: Record<string, unknown>, secrets?: unknown[]): Promise<unknown> => callTool("custom_tools.update_custom_tool", { name, description, parameters, code, response_schema, secrets }),
};

export const image = {
  /** View an image from a URL. The image will be shown to you directly so you can see its contents. */
  view_image: (url: string): Promise<unknown> => callTool("image.view_image", { url }),
};

export const notes = {
  /** Retrieve the full content of one or more notes by rkey. Works for any rkey — `self`, `operator`, `skill:*`, `task:*`, or any free-form note name.

Call this to load full content before relying on a note. The system prompt only shows skill names with short previews — you must `note_get` to see a skill's actual content before executing it. The `operator` note is not preloaded at all — call `note_get(['operator'])` when you need to recall context about the operator.

Pass an array of rkeys to batch multiple reads into one call. */
  note_get: (rkeys: unknown[]): Promise<unknown> => callTool("notes.note_get", { rkeys }),

  /** List all stored notes with short content previews.

Use this to discover what notes exist when you're not sure, or to audit your memory. The system prompt shows `skill:*` notes by name but not your other notes — call `note_list` to see everything you've saved, then `note_get` for full content of anything interesting. */
  note_list: (limit?: number, cursor?: string): Promise<unknown> => callTool("notes.note_list", { limit, cursor }),

  /** Create or update a note by rkey (note name).

Use this to persist information across sessions. If the note exists, it will be updated (overwritten); otherwise created. To append rather than overwrite, read the existing note first and write the combined content.

Conventional rkeys:
- `self` — your identity, personality, and behavior instructions (loaded into your system prompt at boot)
- `operator` — what you know about the human operator (preferences, context, ongoing projects)
- `skill:<name>` — a reusable procedure you can load and execute
- `task:<name>` — progress / state for a long-running task
- any other rkey — free-form facts, reminders, scratch space

Rkey rules: 1-512 chars, alphanumeric with `-_.~:` */
  note_upsert: (rkey: string, content: string): Promise<unknown> => callTool("notes.note_upsert", { rkey, content }),
};

export const notify = {
  /** Send a message to Discord via webhook. Useful for alerting the operator about background events — scheduled task results, findings that need attention, errors, etc.

Requires a secret named `DISCORD_WEBHOOK_URL` (a webhook URL from a Discord channel's integration settings). The operator configures it via the Secrets screen in the TUI.

If `title` is provided the message renders as an embed with the title as a heading; otherwise as a plain message. Discord limits content to 2000 characters and this tool truncates beyond that. */
  discord: (content: string, title?: string): Promise<unknown> => callTool("notify.discord", { content, title }),
};

export const scheduler = {
  /** Create a new scheduled task that will fire a prompt on a recurring schedule. When the schedule fires, the agent runs the prompt in a fresh conversation (no prior history) with full access to tools and notes, and the run is saved as its own chat session.

Use this to give yourself recurring work — daily summaries, periodic checks, reminders. The prompt should be self-contained since scheduled runs start with no conversation context.

Supported schedule expressions:
- Duration: `30m`, `2h`, `1h30m`, `90s` — fires every interval (min 1 minute)
- Keywords: `hourly`, `daily`, `weekly`
- Daily at time: `daily@9:00`, `daily@14:30` — 24h clock, UTC
- Weekly on day: `weekly@monday`, `weekly@fri`
- Weekly on day at time: `weekly@monday@9:00`
- Cron: `cron:0 9 * * *` (if croniter installed) */
  create_schedule: (name: string, schedule: string, prompt: string): Promise<unknown> => callTool("scheduler.create_schedule", { name, schedule, prompt }),

  /** Delete a scheduled task. */
  delete_schedule: (name: string): Promise<unknown> => callTool("scheduler.delete_schedule", { name }),

  /** Disable a scheduled task so it will not fire until re-enabled. */
  disable_schedule: (name: string): Promise<unknown> => callTool("scheduler.disable_schedule", { name }),

  /** Enable a scheduled task so it will fire on its schedule. */
  enable_schedule: (name: string): Promise<unknown> => callTool("scheduler.enable_schedule", { name }),

  /** Get full details of a scheduled task. */
  get_schedule: (name: string): Promise<unknown> => callTool("scheduler.get_schedule", { name }),

  /** List all scheduled tasks with their status and next run time. */
  list_schedules: (): Promise<unknown> => callTool("scheduler.list_schedules", {}),

  /** Update an existing scheduled task. Any unspecified field is left unchanged.

Supported schedule expressions:
- Duration: `30m`, `2h`, `1h30m`, `90s` — fires every interval (min 1 minute)
- Keywords: `hourly`, `daily`, `weekly`
- Daily at time: `daily@9:00`, `daily@14:30` — 24h clock, UTC
- Weekly on day: `weekly@monday`, `weekly@fri`
- Weekly on day at time: `weekly@monday@9:00`
- Cron: `cron:0 9 * * *` (if croniter installed) */
  update_schedule: (name: string, schedule?: string, prompt?: string, enabled?: boolean): Promise<unknown> => callTool("scheduler.update_schedule", { name, schedule, prompt, enabled }),
};

export const summarize = {
  /** Summarize text using a sub-agent LLM. Useful for condensing long tool outputs, articles, or thread contents before presenting to the user. */
  summarize: (text: string, instructions?: string, max_length?: string): Promise<unknown> => callTool("summarize.summarize", { text, instructions, max_length }),
};

export const web = {
  /** Fetch and read the contents of a webpage. Use this to read articles, documentation, blog posts, or any URL you want to examine in detail. */
  fetch_page: (url: string, max_chars?: number): Promise<unknown> => callTool("web.fetch_page", { url, max_chars }),

  /** Plain HTTP GET of a URL. Returns the raw response body (truncated). Unlike `fetch_page`, this does NOT go through Exa — use it for JSON APIs, URLs behind auth, or when you just want the raw response.

Declare any required auth tokens as secrets (via `create_custom_tool`) if you need Authorization headers; this tool itself doesn't inject them. For simple GETs to public URLs this is the right tool. */
  http_get: (url: string, max_chars?: number): Promise<unknown> => callTool("web.http_get", { url, max_chars }),

  /** Search the web for current information via Exa AI. Returns results with titles, URLs, and text snippets.

If summary_focus is provided, the results will include a note about what to focus on when analyzing them. */
  web_search: (query: string, summary_focus?: string, num_results?: number): Promise<unknown> => callTool("web.web_search", { query, summary_focus, num_results }),
};
