// -- API response types --

export interface Session {
  rkey: string;
  title: string;
  createdAt: string;
}

export interface SessionList {
  sessions: Session[];
  cursor: string | null;
}

export interface Message {
  id: number;
  sessionId: string;
  sender: string;
  content: string;
  createdAt: string;
}

export interface MessageList {
  messages: Message[];
  cursor: string | null;
}

export interface Status {
  model: string;
  discord: boolean;
  schedules: number;
  schedules_pending: number;
  secrets: number;
  custom_tools_approved: number;
  custom_tools_pending: number;
}

export interface ToolSummary {
  name: string;
  description: string;
  approved: boolean;
  requires_net: boolean;
  secrets: string[];
}

export interface ToolDetail {
  name: string;
  description: string;
  approved: boolean;
  requires_net: boolean;
  code: string;
  parameters: Record<string, unknown>;
  secrets: { name: string; status: "set" | "missing" }[];
}

export interface Secret {
  name: string;
  masked_value: string;
}

export interface Schedule {
  name: string;
  schedule: string;
  prompt: string;
  enabled: boolean;
  last_run: string | null;
  next_run: string | null;
  condition_code: string | null;
  requires_net: boolean;
  secrets: string[];
  approved: boolean;
}

export interface SystemPrompt {
  text: string;
  char_count: number;
  token_estimate: number;
}

// -- SSE event types --

export interface ChatSSEEvent {
  event: string;
  data: Record<string, unknown>;
}

// Parsed message history (from raw store records)
export interface ParsedMessage {
  role: string;
  content: string;
}
