import type {
  MCPServerDetail,
  MCPServerSummary,
  MemoryEntry,
  MemoryEntryList,
  MemorySearchResponse,
  MessageList,
  Schedule,
  Secret,
  Session,
  SessionList,
  Status,
  SystemPrompt,
  ToolDetail,
  ToolSummary,
} from "./types";

const API_BASE = "/api";

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    const message = typeof detail === "string"
      ? detail
      : (detail as { message?: string })?.message || `HTTP ${status}`;
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new ApiError(resp.status, body.detail ?? body);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json();
}

function enc(value: string): string {
  return encodeURIComponent(value);
}

// -- sessions --

export const api = {
  getStatus: () => fetchJSON<Status>(`${API_BASE}/status`),

  listSessions: (limit = 50, cursor?: string) =>
    fetchJSON<SessionList>(
      `${API_BASE}/sessions?limit=${limit}${cursor ? `&cursor=${enc(cursor)}` : ""}`,
    ),

  createSession: (title = "") =>
    fetchJSON<Session>(`${API_BASE}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),

  getSession: (id: string) => fetchJSON<Session>(`${API_BASE}/sessions/${enc(id)}`),

  deleteSession: (id: string) =>
    fetchJSON<void>(`${API_BASE}/sessions/${enc(id)}`, { method: "DELETE" }),

  listMessages: (id: string, limit = 100, cursor?: string) =>
    fetchJSON<MessageList>(
      `${API_BASE}/sessions/${enc(id)}/messages?limit=${limit}${cursor ? `&cursor=${enc(cursor)}` : ""}`,
    ),

  // -- chat (SSE via fetch + ReadableStream) --

  chat: (
    sessionId: string,
    message: string,
    images?: { media_type: string; data: string }[],
    onEvent?: (event: string, data: Record<string, unknown>) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    return fetch(`${API_BASE}/sessions/${enc(sessionId)}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, images }),
      signal,
    }).then(async (resp) => {
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new ApiError(resp.status, body.detail ?? body);
      }
      const reader = resp.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const lines = part.split("\n");
          let eventName = "message";
          let dataStr = "{}";
          for (const line of lines) {
            if (line.startsWith("event: ")) eventName = line.slice(7);
            else if (line.startsWith("data: ")) dataStr = line.slice(6);
          }
          if (onEvent) {
            onEvent(eventName, JSON.parse(dataStr));
          }
        }
      }
    });
  },

  // -- tools --

  listTools: () => fetchJSON<ToolSummary[]>(`${API_BASE}/tools`),
  getTool: (name: string) => fetchJSON<ToolDetail>(`${API_BASE}/tools/${enc(name)}`),
  approveTool: (name: string) =>
    fetchJSON<{ message: string }>(`${API_BASE}/tools/${enc(name)}/approve`, {
      method: "POST",
    }),
  revokeTool: (name: string) =>
    fetchJSON<{ message: string }>(`${API_BASE}/tools/${enc(name)}/revoke`, {
      method: "POST",
    }),
  deleteTool: (name: string) =>
    fetchJSON<void>(`${API_BASE}/tools/${enc(name)}`, { method: "DELETE" }),

  // -- secrets --

  listSecrets: () => fetchJSON<Secret[]>(`${API_BASE}/secrets`),
  setSecret: (name: string, value: string) =>
    fetchJSON<Secret>(`${API_BASE}/secrets`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, value }),
    }),
  deleteSecret: (name: string) =>
    fetchJSON<void>(`${API_BASE}/secrets/${enc(name)}`, { method: "DELETE" }),

  // -- schedules --

  listSchedules: () => fetchJSON<Schedule[]>(`${API_BASE}/schedules`),
  createSchedule: (
    name: string,
    schedule: string,
    prompt: string,
    condition_code?: string,
    requires_net?: boolean,
    secrets?: string[],
  ) =>
    fetchJSON<Schedule>(`${API_BASE}/schedules`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        schedule,
        prompt,
        condition_code: condition_code || null,
        requires_net: requires_net || false,
        secrets: secrets || [],
      }),
    }),
  enableSchedule: (name: string) =>
    fetchJSON<Schedule>(`${API_BASE}/schedules/${enc(name)}/enable`, {
      method: "POST",
    }),
  disableSchedule: (name: string) =>
    fetchJSON<Schedule>(`${API_BASE}/schedules/${enc(name)}/disable`, {
      method: "POST",
    }),
  deleteSchedule: (name: string) =>
    fetchJSON<void>(`${API_BASE}/schedules/${enc(name)}`, { method: "DELETE" }),
  approveSchedule: (name: string) =>
    fetchJSON<Schedule>(`${API_BASE}/schedules/${enc(name)}/approve`, {
      method: "POST",
    }),
  revokeSchedule: (name: string) =>
    fetchJSON<Schedule>(`${API_BASE}/schedules/${enc(name)}/revoke`, {
      method: "POST",
    }),
  testSchedule: (name: string) =>
    fetchJSON<Record<string, unknown>>(`${API_BASE}/schedules/${enc(name)}/test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    }),
  updateSchedule: (
    name: string,
    fields: {
      schedule?: string;
      prompt?: string;
      condition_code?: string;
      requires_net?: boolean;
      secrets?: string[];
    },
  ) =>
    fetchJSON<Schedule>(`${API_BASE}/schedules/${enc(name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fields),
    }),
  runScheduleNow: (name: string) =>
    fetchJSON<{ fired: boolean; name: string }>(
      `${API_BASE}/schedules/${enc(name)}/run-now`,
      { method: "POST" },
    ),
  testTool: (name: string, params: Record<string, unknown> = {}) =>
    fetchJSON<Record<string, unknown>>(`${API_BASE}/tools/${enc(name)}/test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params }),
    }),

  // -- system prompt --

  getSystemPrompt: () => fetchJSON<SystemPrompt>(`${API_BASE}/system-prompt`),

  // -- memory --

  listMemory: (type?: string, limit = 50, cursor?: number) =>
    fetchJSON<MemoryEntryList>(
      `${API_BASE}/memory?limit=${limit}${type ? `&type=${enc(type)}` : ""}${cursor ? `&cursor=${cursor}` : ""}`,
    ),

  getMemory: (id: number) => fetchJSON<MemoryEntry>(`${API_BASE}/memory/${id}`),

  createMemory: (type: string, scope: string, content: string, tags?: string[]) =>
    fetchJSON<MemoryEntry>(`${API_BASE}/memory`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, scope, content, tags: tags || [] }),
    }),

  updateMemory: (id: number, content?: string, tags?: string[]) =>
    fetchJSON<MemoryEntry>(`${API_BASE}/memory/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, tags }),
    }),

  deleteMemory: (id: number) =>
    fetchJSON<void>(`${API_BASE}/memory/${id}`, { method: "DELETE" }),

  searchMemory: (query: string, type?: string, types?: string[], scope?: string, tags?: string[], limit = 20) =>
    fetchJSON<MemorySearchResponse>(`${API_BASE}/memory/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, type, types, scope, tags, limit }),
    }),

  // -- MCP --

  listMCPServers: () => fetchJSON<MCPServerSummary[]>(`${API_BASE}/mcp/servers`),
  createMCPServer: (config: {
    name: string;
    transport: string;
    url?: string;
    command?: string;
    args?: string[];
    auth_type?: string;
    auth_token_secret?: string;
    env_secrets?: string[];
    enabled?: boolean;
  }) =>
    fetchJSON<MCPServerSummary>(`${API_BASE}/mcp/servers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    }),
  getMCPServer: (name: string) =>
    fetchJSON<MCPServerDetail>(`${API_BASE}/mcp/servers/${enc(name)}`),
  deleteMCPServer: (name: string) =>
    fetchJSON<void>(`${API_BASE}/mcp/servers/${enc(name)}`, { method: "DELETE" }),
  connectMCPServer: (name: string) =>
    fetchJSON<MCPServerDetail>(`${API_BASE}/mcp/servers/${enc(name)}/connect`, {
      method: "POST",
    }),
  disconnectMCPServer: (name: string) =>
    fetchJSON<MCPServerDetail>(`${API_BASE}/mcp/servers/${enc(name)}/disconnect`, {
      method: "POST",
    }),
  refreshMCPServer: (name: string) =>
    fetchJSON<MCPServerDetail>(`${API_BASE}/mcp/servers/${enc(name)}/refresh`, {
      method: "POST",
    }),
  approveMCPTool: (serverName: string, toolName: string) =>
    fetchJSON<{ message: string }>(
      `${API_BASE}/mcp/servers/${enc(serverName)}/tools/approve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tool_name: toolName }),
      },
    ),
  revokeMCPTool: (serverName: string, toolName: string) =>
    fetchJSON<{ message: string }>(
      `${API_BASE}/mcp/servers/${enc(serverName)}/tools/revoke`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tool_name: toolName }),
      },
    ),
  deauthorizeMCPOAuth: (name: string) =>
    fetchJSON<{ message: string }>(
      `${API_BASE}/mcp/servers/${enc(name)}/oauth/deauthorize`,
      { method: "POST" },
    ),
  getOAuthConsentUrl: (name: string) =>
    fetchJSON<{ consent_url: string | null; pending_callback: boolean }>(
      `${API_BASE}/mcp/servers/${enc(name)}/oauth/consent-url`,
    ),
  submitOAuthCallback: (name: string, redirectUrl: string) =>
    fetchJSON<{ message: string }>(
      `${API_BASE}/mcp/servers/${enc(name)}/oauth/callback`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ redirect_url: redirectUrl }),
      },
    ),
};
