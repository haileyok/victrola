import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { SessionList } from "@/components/sessions/SessionList";
import { mockStatus, mockSessionList, mockSession } from "@/test/mockApi";

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

function renderSessionList() {
  return render(<MemoryRouter><SessionList /></MemoryRouter>);
}

describe("SessionList", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getStatus.mockResolvedValue(mockStatus);
    api.listSessions.mockResolvedValue(mockSessionList);
    api.createSession.mockResolvedValue(mockSession);
    api.deleteSession.mockResolvedValue(undefined);
  });

  it("shows loading state initially", () => {
    renderSessionList();
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("loads and displays sessions with status banner", async () => {
    renderSessionList();
    await waitFor(() => expect(screen.getByText("Session One")).toBeInTheDocument());
    // Verify status banner content — check the banner div directly
    const banner = screen.getByText(/Model:/).closest("div");
    expect(banner?.textContent).toContain("claude-sonnet-4-5");
    expect(banner?.textContent).toContain("Discord:");
    expect(banner?.textContent).toContain("on");
    expect(banner?.textContent).toContain("Schedules:");
    expect(banner?.textContent).toContain("3");
  });

  it("shows empty state when no sessions", async () => {
    api.listSessions.mockResolvedValue({ sessions: [], cursor: null });
    renderSessionList();
    await waitFor(() => expect(screen.getByText(/No sessions yet/)).toBeInTheDocument());
  });

  it("creates a new session on button click", async () => {
    renderSessionList();
    await waitFor(() => expect(screen.getByText("New")).toBeInTheDocument());
    fireEvent.click(screen.getByText("New"));
    await waitFor(() => expect(api.createSession).toHaveBeenCalled());
  });

  it("displays untitled for sessions with empty title", async () => {
    api.listSessions.mockResolvedValue({
      sessions: [{ rkey: "x1", title: "", createdAt: "2024-01-01T00:00:00.000Z" }],
      cursor: null,
    });
    renderSessionList();
    await waitFor(() => expect(screen.getByText("(untitled)")).toBeInTheDocument());
  });
});
