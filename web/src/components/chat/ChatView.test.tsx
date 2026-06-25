import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { ChatView } from "@/components/chat/ChatView";
import { mockMessageList } from "@/test/mockApi";

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

function renderChatView(sessionId: string = "abc123") {
  return render(
    <MemoryRouter initialEntries={[`/sessions/${sessionId}`]}>
      <Routes>
        <Route path="/sessions/:id" element={<ChatView />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("ChatView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listMessages.mockResolvedValue(mockMessageList);
    api.chat.mockResolvedValue(undefined);
  });

  it("shows loading state initially", () => {
    renderChatView();
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("loads and displays message history", async () => {
    renderChatView();
    await waitFor(() => expect(screen.getByText("Hello there")).toBeInTheDocument());
    expect(screen.getByText("Hi! How can I help?")).toBeInTheDocument();
  });

  it("shows empty state when no messages", async () => {
    api.listMessages.mockResolvedValue({ messages: [], cursor: null });
    renderChatView();
    await waitFor(() => expect(screen.getByText(/Send a message to start chatting/)).toBeInTheDocument());
  });

  it("disables input and shows thinking state while responding", async () => {
    api.chat.mockReturnValue(new Promise(() => {}));
    renderChatView();
    await waitFor(() => expect(screen.getByPlaceholderText("Type a message…")).toBeInTheDocument());
    const input = screen.getByPlaceholderText("Type a message…");
    fireEvent.change(input, { target: { value: "test message" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("Thinking…")).toBeInTheDocument());
    expect(input).toBeDisabled();
  });

  it("shows user message immediately after sending", async () => {
    renderChatView();
    await waitFor(() => expect(screen.getByPlaceholderText("Type a message…")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("Type a message…"), { target: { value: "my new message" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("my new message")).toBeInTheDocument());
  });

  it("handles SSE response event and shows assistant message", async () => {
    let cb: ((e: string, d: Record<string, unknown>) => void) | null = null;
    api.chat.mockImplementation((_s: string, _m: string, _i: unknown, onEvent?: (e: string, d: Record<string, unknown>) => void) => {
      cb = onEvent || null;
      return Promise.resolve();
    });
    renderChatView();
    await waitFor(() => expect(screen.getByPlaceholderText("Type a message…")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("Type a message…"), { target: { value: "hello agent" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(cb).not.toBeNull());
    act(() => { cb!("response", { text: "Here is my reply" }); });
    await waitFor(() => expect(screen.getByText("Here is my reply")).toBeInTheDocument());
  });

  it("handles SSE error event", async () => {
    let cb: ((e: string, d: Record<string, unknown>) => void) | null = null;
    api.chat.mockImplementation((_s: string, _m: string, _i: unknown, onEvent?: (e: string, d: Record<string, unknown>) => void) => {
      cb = onEvent || null;
      return Promise.resolve();
    });
    renderChatView();
    await waitFor(() => expect(screen.getByPlaceholderText("Type a message…")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("Type a message…"), { target: { value: "hi" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(cb).not.toBeNull());
    act(() => { cb!("error", { message: "Something broke" }); });
    await waitFor(() => expect(screen.getByText("Something broke")).toBeInTheDocument());
  });

  it("handles tool_start and tool_done events", async () => {
    let cb: ((e: string, d: Record<string, unknown>) => void) | null = null;
    api.chat.mockImplementation((_s: string, _m: string, _i: unknown, onEvent?: (e: string, d: Record<string, unknown>) => void) => {
      cb = onEvent || null;
      return Promise.resolve();
    });
    renderChatView();
    await waitFor(() => expect(screen.getByPlaceholderText("Type a message…")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("Type a message…"), { target: { value: "use a tool" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(cb).not.toBeNull());
    act(() => { cb!("tool_start", { tool: "execute_code", code: "const x = 1;" }); });
    await waitFor(() => {
      expect(screen.getByText("execute_code")).toBeInTheDocument();
      expect(screen.getByText("running…")).toBeInTheDocument();
    });
    expect(screen.getByText(/Running: execute_code/)).toBeInTheDocument();
    act(() => { cb!("tool_done", { tool: "execute_code", result: { output: 42 }, success: true }); });
    await waitFor(() => expect(screen.queryByText("running…")).not.toBeInTheDocument());
  });

  it("updates token stats on llm_done event", async () => {
    let cb: ((e: string, d: Record<string, unknown>) => void) | null = null;
    api.chat.mockImplementation((_s: string, _m: string, _i: unknown, onEvent?: (e: string, d: Record<string, unknown>) => void) => {
      cb = onEvent || null;
      return Promise.resolve();
    });
    renderChatView();
    await waitFor(() => expect(screen.getByPlaceholderText("Type a message…")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("Type a message…"), { target: { value: "hi" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(cb).not.toBeNull());
    act(() => { cb!("llm_start", {}); });
    act(() => { cb!("llm_done", { usage: { input_tokens: 100, output_tokens: 50 } }); });
    await waitFor(() => {
      expect(screen.getByText(/in:.*100/)).toBeInTheDocument();
      expect(screen.getByText(/out:.*50/)).toBeInTheDocument();
      expect(screen.getByText(/calls:.*1/)).toBeInTheDocument();
    });
  });

  it("does not send empty messages", async () => {
    renderChatView();
    await waitFor(() => expect(screen.getByText("Send")).toBeInTheDocument());
    expect(screen.getByText("Send")).toBeDisabled();
  });

  it("sends on Enter key press", async () => {
    renderChatView();
    await waitFor(() => expect(screen.getByPlaceholderText("Type a message…")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("Type a message…"), { target: { value: "enter test" } });
    fireEvent.keyDown(screen.getByPlaceholderText("Type a message…"), { key: "Enter", shiftKey: false });
    await waitFor(() => expect(api.chat).toHaveBeenCalled());
  });

  it("does not send on Shift+Enter", async () => {
    renderChatView();
    await waitFor(() => expect(screen.getByPlaceholderText("Type a message…")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("Type a message…"), { target: { value: "shift enter test" } });
    fireEvent.keyDown(screen.getByPlaceholderText("Type a message…"), { key: "Enter", shiftKey: true });
    expect(api.chat).not.toHaveBeenCalled();
  });
});
