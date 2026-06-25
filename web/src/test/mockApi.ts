import { vi } from "vitest";
import type { Status, Session, SessionList, MessageList, ToolSummary, ToolDetail, Secret, Schedule, SystemPrompt, MemoryEntry, MemoryEntryList, MemorySearchResponse } from "@/lib/types";

type MockApi = {
  getStatus: ReturnType<typeof vi.fn>;
  listSessions: ReturnType<typeof vi.fn>;
  createSession: ReturnType<typeof vi.fn>;
  getSession: ReturnType<typeof vi.fn>;
  deleteSession: ReturnType<typeof vi.fn>;
  listMessages: ReturnType<typeof vi.fn>;
  chat: ReturnType<typeof vi.fn>;
  listTools: ReturnType<typeof vi.fn>;
  getTool: ReturnType<typeof vi.fn>;
  approveTool: ReturnType<typeof vi.fn>;
  revokeTool: ReturnType<typeof vi.fn>;
  deleteTool: ReturnType<typeof vi.fn>;
  listSecrets: ReturnType<typeof vi.fn>;
  setSecret: ReturnType<typeof vi.fn>;
  deleteSecret: ReturnType<typeof vi.fn>;
  listSchedules: ReturnType<typeof vi.fn>;
  createSchedule: ReturnType<typeof vi.fn>;
  enableSchedule: ReturnType<typeof vi.fn>;
  disableSchedule: ReturnType<typeof vi.fn>;
  deleteSchedule: ReturnType<typeof vi.fn>;
  getSystemPrompt: ReturnType<typeof vi.fn>;
  listMemory: ReturnType<typeof vi.fn>;
  getMemory: ReturnType<typeof vi.fn>;
  createMemory: ReturnType<typeof vi.fn>;
  updateMemory: ReturnType<typeof vi.fn>;
  deleteMemory: ReturnType<typeof vi.fn>;
  searchMemory: ReturnType<typeof vi.fn>;
};

export function makeMockApi(): MockApi {
  return {
    getStatus: vi.fn(),
    listSessions: vi.fn(),
    createSession: vi.fn(),
    getSession: vi.fn(),
    deleteSession: vi.fn(),
    listMessages: vi.fn(),
    chat: vi.fn(),
    listTools: vi.fn(),
    getTool: vi.fn(),
    approveTool: vi.fn(),
    revokeTool: vi.fn(),
    deleteTool: vi.fn(),
    listSecrets: vi.fn(),
    setSecret: vi.fn(),
    deleteSecret: vi.fn(),
    listSchedules: vi.fn(),
    createSchedule: vi.fn(),
    enableSchedule: vi.fn(),
    disableSchedule: vi.fn(),
    deleteSchedule: vi.fn(),
    getSystemPrompt: vi.fn(),
    listMemory: vi.fn(),
    getMemory: vi.fn(),
    createMemory: vi.fn(),
    updateMemory: vi.fn(),
    deleteMemory: vi.fn(),
    searchMemory: vi.fn(),
  };
}

// Factory helpers for mock data
export const mockStatus: Status = {
  model: "claude-sonnet-4-5",
  discord: true,
  schedules: 3,
  schedules_pending: 1,
  secrets: 5,
  custom_tools_approved: 2,
  custom_tools_pending: 1,
};

export const mockSession: Session = {
  rkey: "abc123",
  title: "Test Session",
  createdAt: "2024-01-01T00:00:00.000Z",
};

export const mockSessionList: SessionList = {
  sessions: [
    { rkey: "abc123", title: "Session One", createdAt: "2024-01-01T00:00:00.000Z" },
    { rkey: "def456", title: "Session Two", createdAt: "2024-01-02T00:00:00.000Z" },
  ],
  cursor: null,
};

export const mockMessageList: MessageList = {
  messages: [
    {
      id: 1,
      sessionId: "abc123",
      sender: "user",
      content: JSON.stringify({ role: "user", content: "Hello there" }),
      createdAt: "2024-01-01T00:00:01.000Z",
    },
    {
      id: 2,
      sessionId: "abc123",
      sender: "assistant",
      content: JSON.stringify({ role: "assistant", content: "Hi! How can I help?" }),
      createdAt: "2024-01-01T00:00:02.000Z",
    },
  ],
  cursor: null,
};

export const mockToolSummary: ToolSummary[] = [
  {
    name: "weather",
    description: "Get the weather",
    approved: true,
    requires_net: false,
    secrets: [],
  },
  {
    name: "fetcher",
    description: "Fetch a URL",
    approved: false,
    requires_net: true,
    secrets: ["API_KEY"],
  },
];

export const mockToolDetail: ToolDetail = {
  name: "weather",
  description: "Get the weather for a city",
  approved: false,
  requires_net: false,
  code: "const result = await tools.web.web_search({ query: 'weather' });\noutput(result);",
  parameters: { type: "object", properties: { city: { type: "string" } } },
  secrets: [
    { name: "API_KEY", status: "missing" },
    { name: "EXA_KEY", status: "set" },
  ],
};

export const mockSecrets: Secret[] = [
  { name: "OPENAI_API_KEY", masked_value: "********..." },
  { name: "DISCORD_TOKEN", masked_value: "****" },
];

export const mockSchedules: Schedule[] = [
  {
    name: "daily_report",
    schedule: "daily@9:00",
    prompt: "Generate a daily report",
    enabled: true,
    last_run: "2024-01-01T09:00:00.000Z",
    next_run: "2024-01-02T09:00:00.000Z",
    condition_code: null,
    requires_net: false,
    secrets: [],
    approved: false,
  },
  {
    name: "weekly_cleanup",
    schedule: "weekly@monday",
    prompt: "Clean up old sessions",
    enabled: false,
    last_run: null,
    next_run: null,
    condition_code: null,
    requires_net: false,
    secrets: [],
    approved: false,
  },
];

export const mockSystemPrompt: SystemPrompt = {
  text: "You are a helpful assistant.\n\nFollow instructions carefully.",
  char_count: 52,
  token_estimate: 13,
};

// -- memory mock data --

export const mockMemoryEntries: MemoryEntry[] = [
  {
    id: 1,
    type: "self",
    scope: "self",
    content: "You are Victrola, a personal AI assistant.",
    metadata: {},
    createdAt: "2024-01-01T00:00:00.000Z",
    updatedAt: "2024-01-01T00:00:00.000Z",
  },
  {
    id: 2,
    type: "factual",
    scope: "topic",
    content: "Deploys use blue-green strategy with 5-minute health checks.",
    metadata: { tags: ["deploy"] },
    createdAt: "2024-01-02T00:00:00.000Z",
    updatedAt: "2024-01-02T00:00:00.000Z",
  },
];

export const mockMemoryList: MemoryEntryList = {
  entries: mockMemoryEntries,
  cursor: null,
};

export const mockMemoryListWithMore: MemoryEntryList = {
  entries: mockMemoryEntries,
  cursor: 3,
};

export const mockMemorySearchResults: MemorySearchResponse = {
  results: [
    {
      id: 2,
      type: "factual",
      scope: "topic",
      content: "Deploys use blue-green strategy with 5-minute health checks.",
      score: 0.95,
      matched_by: "both",
    },
  ],
};
