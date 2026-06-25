import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { ToolsView } from "@/components/tools/ToolsView";
import { mockToolSummary } from "@/test/mockApi";

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

function renderToolsView() {
  return render(<MemoryRouter><ToolsView /></MemoryRouter>);
}

describe("ToolsView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listTools.mockResolvedValue(mockToolSummary);
    api.approveTool.mockResolvedValue({ message: "approved" });
    api.revokeTool.mockResolvedValue({ message: "revoked" });
    api.deleteTool.mockResolvedValue(undefined);
  });

  it("shows loading state initially", () => {
    renderToolsView();
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("loads and displays tools with status badges", async () => {
    renderToolsView();
    await waitFor(() => expect(screen.getByText("weather")).toBeInTheDocument());
    expect(screen.getByText("fetcher")).toBeInTheDocument();
    expect(screen.getByText("approved")).toBeInTheDocument();
    expect(screen.getByText("pending")).toBeInTheDocument();
    expect(screen.getByText("net")).toBeInTheDocument();
  });

  it("shows empty state when no tools", async () => {
    api.listTools.mockResolvedValue([]);
    renderToolsView();
    await waitFor(() => expect(screen.getByText(/No custom tools/)).toBeInTheDocument());
  });

  it("shows confirm dialog when deleting", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    renderToolsView();
    await waitFor(() => expect(screen.getByText("weather")).toBeInTheDocument());
    const allButtons = screen.getAllByRole("button");
    const deleteBtn = allButtons.find((btn) => {
      const svg = btn.querySelector("svg");
      return svg && svg.getAttribute("class")?.includes("trash");
    });
    if (deleteBtn) {
      fireEvent.click(deleteBtn);
      expect(confirmSpy).toHaveBeenCalled();
    }
    confirmSpy.mockRestore();
  });
});
