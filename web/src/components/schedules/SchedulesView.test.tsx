import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { SchedulesView } from "@/components/schedules/SchedulesView";
import { mockSchedules } from "@/test/mockApi";

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

describe("SchedulesView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listSchedules.mockResolvedValue(mockSchedules);
    api.createSchedule.mockResolvedValue(mockSchedules[0]);
    api.enableSchedule.mockResolvedValue({ ...mockSchedules[1], enabled: true });
    api.disableSchedule.mockResolvedValue({ ...mockSchedules[0], enabled: false });
    api.deleteSchedule.mockResolvedValue(undefined);
  });

  it("shows loading state initially", () => {
    render(<SchedulesView />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("loads and displays schedules with status badges", async () => {
    render(<SchedulesView />);
    await waitFor(() => expect(screen.getByText("daily_report")).toBeInTheDocument());
    expect(screen.getByText("weekly_cleanup")).toBeInTheDocument();
    expect(screen.getByText("enabled")).toBeInTheDocument();
    expect(screen.getByText("disabled")).toBeInTheDocument();
    expect(screen.getByText(/daily@9:00/)).toBeInTheDocument();
  });

  it("shows next run time for enabled schedules", async () => {
    render(<SchedulesView />);
    await waitFor(() => expect(screen.getByText(/next:/)).toBeInTheDocument());
  });

  it("shows empty state when no schedules", async () => {
    api.listSchedules.mockResolvedValue([]);
    render(<SchedulesView />);
    await waitFor(() => expect(screen.getByText(/No scheduled tasks/)).toBeInTheDocument());
  });

  it("opens create dialog on button click", async () => {
    render(<SchedulesView />);
    await waitFor(() => expect(screen.getByText("New")).toBeInTheDocument());
    fireEvent.click(screen.getByText("New"));
    await waitFor(() => {
      expect(screen.getByText("New Schedule")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("my_task")).toBeInTheDocument();
    });
  });

  it("shows error when fields are missing", async () => {
    render(<SchedulesView />);
    await waitFor(() => expect(screen.getByText("New")).toBeInTheDocument());
    fireEvent.click(screen.getByText("New"));
    await waitFor(() => expect(screen.getByText("Create")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Create"));
    expect(screen.getByText("All fields are required")).toBeInTheDocument();
  });

  it("creates a schedule with all fields filled", async () => {
    render(<SchedulesView />);
    await waitFor(() => expect(screen.getByText("New")).toBeInTheDocument());
    fireEvent.click(screen.getByText("New"));
    await waitFor(() => expect(screen.getByPlaceholderText("my_task")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("my_task"), { target: { value: "my_task" } });
    fireEvent.change(screen.getByPlaceholderText(/30m/), { target: { value: "1h" } });
    fireEvent.change(screen.getByPlaceholderText("Instruction for the agent…"), { target: { value: "Do something" } });
    fireEvent.click(screen.getByText("Create"));
    await waitFor(() => expect(api.createSchedule).toHaveBeenCalledWith("my_task", "1h", "Do something"));
  });
});
