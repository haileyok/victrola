import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryView } from "@/components/memory/MemoryView";
import {
  mockMemoryList,
  mockMemoryListWithMore,
  mockMemorySearchResults,
} from "@/test/mockApi";

const { mockApi, ApiError } = vi.hoisted(() => ({
  mockApi: {
    getStatus: vi.fn(), listSessions: vi.fn(), createSession: vi.fn(),
    getSession: vi.fn(), deleteSession: vi.fn(), listMessages: vi.fn(),
    chat: vi.fn(), listTools: vi.fn(), getTool: vi.fn(), approveTool: vi.fn(),
    revokeTool: vi.fn(), deleteTool: vi.fn(), listSecrets: vi.fn(),
    setSecret: vi.fn(), deleteSecret: vi.fn(), listSchedules: vi.fn(),
    createSchedule: vi.fn(), enableSchedule: vi.fn(), disableSchedule: vi.fn(),
    deleteSchedule: vi.fn(), getSystemPrompt: vi.fn(),
    listMemory: vi.fn(), getMemory: vi.fn(), createMemory: vi.fn(),
    updateMemory: vi.fn(), deleteMemory: vi.fn(), searchMemory: vi.fn(),
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

describe("MemoryView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listMemory.mockResolvedValue(mockMemoryList);
    api.searchMemory.mockResolvedValue(mockMemorySearchResults);
    api.createMemory.mockResolvedValue(mockMemoryList.entries[0]);
    api.updateMemory.mockResolvedValue(mockMemoryList.entries[0]);
    api.deleteMemory.mockResolvedValue(undefined);
  });

  it("shows loading state initially", () => {
    render(<MemoryView />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("loads and displays entries with type badges and content previews", async () => {
    render(<MemoryView />);
    await waitFor(() => expect(screen.getByText(/You are Victrola/)).toBeInTheDocument());
    expect(screen.getByText(/Deploys use blue-green/)).toBeInTheDocument();
    // type badges appear in the entries
    expect(screen.getAllByText("self").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("factual").length).toBeGreaterThanOrEqual(1);
  });

  it("shows empty state when no entries", async () => {
    api.listMemory.mockResolvedValue({ entries: [], cursor: null });
    render(<MemoryView />);
    await waitFor(() => expect(screen.getByText(/No memory entries/)).toBeInTheDocument());
  });

  it("type filter changes call api.listMemory with selected type", async () => {
    render(<MemoryView />);
    await waitFor(() => expect(screen.getByText(/You are Victrola/)).toBeInTheDocument());
    // The filter pills are buttons; click the "factual" filter pill.
    // Filter pills render before entries, so the first "factual" is the pill.
    const factualEls = screen.getAllByText("factual");
    // Find the button element among matches (the pill is a <button>)
    const factualPill = factualEls.find((el) => el.tagName === "BUTTON")!;
    fireEvent.click(factualPill);
    await waitFor(() => {
      expect(api.listMemory).toHaveBeenCalledWith("factual");
    });
  });

  it("search button calls api.searchMemory and displays results", async () => {
    render(<MemoryView />);
    await waitFor(() => expect(screen.getByText(/You are Victrola/)).toBeInTheDocument());
    const input = screen.getByPlaceholderText("Search memory…");
    fireEvent.change(input, { target: { value: "deploy" } });
    fireEvent.click(screen.getByText("Search"));
    await waitFor(() => {
      expect(api.searchMemory).toHaveBeenCalledWith("deploy", undefined);
    });
    await waitFor(() => {
      expect(screen.getByText("score: 0.95")).toBeInTheDocument();
      expect(screen.getByText("both")).toBeInTheDocument();
    });
  });

  it("opens add dialog on button click", async () => {
    render(<MemoryView />);
    await waitFor(() => expect(screen.getByText("Add")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Add"));
    await waitFor(() => {
      expect(screen.getByText("New Memory Entry")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("Memory content…")).toBeInTheDocument();
    });
  });

  it("creates an entry via the dialog", async () => {
    render(<MemoryView />);
    await waitFor(() => expect(screen.getByText("Add")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Add"));
    await waitFor(() => expect(screen.getByText("New Memory Entry")).toBeInTheDocument());
    // Fill content + scope (episodic type needs a scope)
    fireEvent.change(screen.getByPlaceholderText("Memory content…"), {
      target: { value: "New fact here" },
    });
    fireEvent.change(screen.getByPlaceholderText("topic, session ID, or free-form"), {
      target: { value: "test-scope" },
    });
    fireEvent.click(screen.getByText("Create"));
    await waitFor(() => {
      expect(api.createMemory).toHaveBeenCalledWith(
        "episodic", "test-scope", "New fact here", undefined,
      );
    });
  });

  it("opens edit dialog for an existing entry", async () => {
    render(<MemoryView />);
    await waitFor(() => expect(screen.getByText(/You are Victrola/)).toBeInTheDocument());
    // Multiple entries have edit buttons; click the first
    const editBtns = screen.getAllByTitle("Edit");
    fireEvent.click(editBtns[0]);
    await waitFor(() => {
      expect(screen.getByText("Edit Memory Entry")).toBeInTheDocument();
    });
  });

  it("updates an entry via the edit dialog", async () => {
    render(<MemoryView />);
    await waitFor(() => expect(screen.getByText(/You are Victrola/)).toBeInTheDocument());
    const editBtns = screen.getAllByTitle("Edit");
    fireEvent.click(editBtns[0]);
    await waitFor(() => expect(screen.getByText("Edit Memory Entry")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("Memory content…"), {
      target: { value: "Updated content" },
    });
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => {
      expect(api.updateMemory).toHaveBeenCalled();
    });
  });

  it("deletes an entry after confirmation", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<MemoryView />);
    await waitFor(() => expect(screen.getByText(/You are Victrola/)).toBeInTheDocument());
    const deleteBtns = screen.getAllByTitle("Delete");
    fireEvent.click(deleteBtns[0]);
    await waitFor(() => {
      expect(api.deleteMemory).toHaveBeenCalled();
    });
  });

  it("load more button fetches next page", async () => {
    api.listMemory.mockResolvedValueOnce(mockMemoryListWithMore);
    render(<MemoryView />);
    await waitFor(() => expect(screen.getByText(/You are Victrola/)).toBeInTheDocument());
    await waitFor(() => {
      expect(screen.getByText("Load more")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Load more"));
    await waitFor(() => {
      expect(api.listMemory).toHaveBeenCalledWith(undefined, 50, 3);
    });
  });
});
