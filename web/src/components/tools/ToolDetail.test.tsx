import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { ToolDetail } from "@/components/tools/ToolDetail";
import { mockToolDetail } from "@/test/mockApi";

const { mockApi, ApiError } = vi.hoisted(() => ({
  mockApi: {
    getStatus: vi.fn(), listSessions: vi.fn(), createSession: vi.fn(),
    getSession: vi.fn(), deleteSession: vi.fn(), listMessages: vi.fn(),
    chat: vi.fn(), listTools: vi.fn(), getTool: vi.fn(), approveTool: vi.fn(),
    revokeTool: vi.fn(), deleteTool: vi.fn(), testTool: vi.fn(),
    listSecrets: vi.fn(), setSecret: vi.fn(), deleteSecret: vi.fn(),
    listSchedules: vi.fn(), createSchedule: vi.fn(), enableSchedule: vi.fn(),
    disableSchedule: vi.fn(), deleteSchedule: vi.fn(), getSystemPrompt: vi.fn(),
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

function renderToolDetail(toolName: string = "weather") {
  return render(
    <MemoryRouter initialEntries={[`/tools/${toolName}`]}>
      <Routes>
        <Route path="/tools/:name" element={<ToolDetail />} />
        <Route path="/tools" element={<div>Tools List</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("ToolDetail", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getTool.mockResolvedValue(mockToolDetail);
    api.approveTool.mockResolvedValue({ message: "approved" });
    api.revokeTool.mockResolvedValue({ message: "revoked" });
    api.deleteTool.mockResolvedValue(undefined);
    api.setSecret.mockResolvedValue({ name: "API_KEY", masked_value: "********..." });
  });

  it("shows loading state initially", () => {
    renderToolDetail();
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("loads and displays tool details", async () => {
    renderToolDetail();
    await waitFor(() => expect(screen.getByText("weather")).toBeInTheDocument());
    expect(screen.getByText("pending")).toBeInTheDocument();
    expect(screen.getByText("Get the weather for a city")).toBeInTheDocument();
    expect(screen.getByText(/No \(sandboxed\)/)).toBeInTheDocument();
    expect(screen.getByText("API_KEY")).toBeInTheDocument();
    expect(screen.getByText("missing")).toBeInTheDocument();
    expect(screen.getByText("EXA_KEY")).toBeInTheDocument();
    expect(screen.getByText("set")).toBeInTheDocument();
  });

  it("shows code and parameters sections", async () => {
    renderToolDetail();
    await waitFor(() => expect(screen.getByText(/Parameters/)).toBeInTheDocument());
    expect(screen.getByText(/web_search/)).toBeInTheDocument();
  });

  it("approves tool when no missing secrets", async () => {
    api.getTool.mockResolvedValue({ ...mockToolDetail, secrets: [] });
    renderToolDetail();
    await waitFor(() => expect(screen.getByText("Approve")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Approve"));
    await waitFor(() => expect(api.approveTool).toHaveBeenCalledWith("weather"));
  });

  it("opens missing secret dialog when approval has missing secrets", async () => {
    api.approveTool.mockRejectedValueOnce(
      new ApiError(400, { message: "missing", missing_secrets: ["API_KEY"] }),
    );
    renderToolDetail();
    await waitFor(() => expect(screen.getByText("Approve")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Approve"));
    await waitFor(() => expect(screen.getByText("Set missing secret")).toBeInTheDocument());
    expect(screen.getByDisplayValue("API_KEY")).toBeDisabled();
  });

  it("saves missing secret then retries approval", async () => {
    api.approveTool
      .mockRejectedValueOnce(new ApiError(400, { message: "missing", missing_secrets: ["API_KEY"] }))
      .mockResolvedValueOnce({ message: "approved" });
    api.getTool
      .mockResolvedValueOnce(mockToolDetail)
      .mockResolvedValueOnce({ ...mockToolDetail, approved: true, secrets: [{ name: "API_KEY", status: "set" }] });

    renderToolDetail();
    await waitFor(() => expect(screen.getByText("Approve")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Approve"));
    await waitFor(() => expect(screen.getByText("Set missing secret")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("Enter value…"), { target: { value: "secret_value" } });
    fireEvent.click(screen.getByText("Save Secret"));
    await waitFor(() => expect(api.setSecret).toHaveBeenCalledWith("API_KEY", "secret_value"));
    await waitFor(() => expect(api.approveTool).toHaveBeenCalledTimes(2));
  });

  it("revokes an approved tool", async () => {
    api.getTool.mockResolvedValue({ ...mockToolDetail, approved: true });
    renderToolDetail();
    await waitFor(() => expect(screen.getByText("Revoke")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Revoke"));
    await waitFor(() => expect(api.revokeTool).toHaveBeenCalledWith("weather"));
  });

  it("deletes a tool after confirmation", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderToolDetail();
    await waitFor(() => expect(screen.getByText("Delete")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Delete"));
    await waitFor(() => expect(api.deleteTool).toHaveBeenCalledWith("weather"));
  });
});
