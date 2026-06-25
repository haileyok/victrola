import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { SystemPromptView } from "@/components/system-prompt/SystemPromptView";
import { mockSystemPrompt } from "@/test/mockApi";

const { mockApi, ApiError } = vi.hoisted(() => ({
  mockApi: {
    getStatus: vi.fn(), listSessions: vi.fn(), createSession: vi.fn(),
    getSession: vi.fn(), deleteSession: vi.fn(), listMessages: vi.fn(),
    chat: vi.fn(), listTools: vi.fn(), getTool: vi.fn(), approveTool: vi.fn(),
    revokeTool: vi.fn(), deleteTool: vi.fn(), listSecrets: vi.fn(),
    setSecret: vi.fn(), deleteSecret: vi.fn(), listSchedules: vi.fn(),
    createSchedule: vi.fn(), enableSchedule: vi.fn(), disableSchedule: vi.fn(),
    deleteSchedule: vi.fn(), getSystemPrompt: vi.fn(),
  },
  ApiError: class ApiError extends Error {
    status: number; detail: unknown;
    constructor(s: number, d: unknown) {
      const m = typeof d === "string" ? d : (d as { message?: string })?.message || `HTTP ${s}`;
      super(m); this.name = "ApiError"; this.status = s; this.detail = d;
    }
  },
}));
vi.mock("@/lib/api", () => ({ api: mockApi, ApiError }));

import { api } from "@/lib/api";

describe("SystemPromptView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getSystemPrompt.mockResolvedValue(mockSystemPrompt);
  });

  it("shows loading state initially", () => {
    render(<SystemPromptView />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("loads and displays the system prompt text", async () => {
    render(<SystemPromptView />);
    await waitFor(() => expect(screen.getByText(/You are a helpful assistant/)).toBeInTheDocument());
    expect(screen.getByText(/Follow instructions carefully/)).toBeInTheDocument();
  });

  it("displays char and token count", async () => {
    render(<SystemPromptView />);
    await waitFor(() => expect(screen.getByText(/52 chars/)).toBeInTheDocument());
    expect(screen.getByText(/~13 tokens/)).toBeInTheDocument();
  });

  it("refreshes the prompt on button click", async () => {
    render(<SystemPromptView />);
    await waitFor(() => expect(screen.getByText("Refresh")).toBeInTheDocument());
    api.getSystemPrompt.mockResolvedValue({ text: "Updated prompt", char_count: 14, token_estimate: 3 });
    fireEvent.click(screen.getByText("Refresh"));
    await waitFor(() => expect(screen.getByText("Updated prompt")).toBeInTheDocument());
    expect(api.getSystemPrompt).toHaveBeenCalledTimes(2);
  });

  it("shows error state when fetch fails", async () => {
    api.getSystemPrompt.mockRejectedValue(new Error("fetch failed"));
    render(<SystemPromptView />);
    await waitFor(() => expect(screen.getByText("Failed to load system prompt.")).toBeInTheDocument());
  });
});
