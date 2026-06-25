import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { SecretsView } from "@/components/secrets/SecretsView";
import { mockSecrets } from "@/test/mockApi";

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

describe("SecretsView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listSecrets.mockResolvedValue(mockSecrets);
    api.setSecret.mockResolvedValue({ name: "NEW_KEY", masked_value: "********..." });
    api.deleteSecret.mockResolvedValue(undefined);
  });

  it("shows loading state initially", () => {
    render(<SecretsView />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("loads and displays secrets with masked values", async () => {
    render(<SecretsView />);
    await waitFor(() => expect(screen.getByText("OPENAI_API_KEY")).toBeInTheDocument());
    expect(screen.getByText("DISCORD_TOKEN")).toBeInTheDocument();
    expect(screen.getByText("********...")).toBeInTheDocument();
    expect(screen.getByText("****")).toBeInTheDocument();
  });

  it("shows empty state when no secrets", async () => {
    api.listSecrets.mockResolvedValue([]);
    render(<SecretsView />);
    await waitFor(() => expect(screen.getByText(/No secrets configured/)).toBeInTheDocument());
  });

  it("opens add dialog on button click", async () => {
    render(<SecretsView />);
    await waitFor(() => expect(screen.getByText("Add")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Add"));
    await waitFor(() => {
      expect(screen.getByText("Add Secret")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("SECRET_NAME")).toBeInTheDocument();
    });
  });

  it("saves a secret via the dialog", async () => {
    render(<SecretsView />);
    await waitFor(() => expect(screen.getByText("Add")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Add"));
    await waitFor(() => expect(screen.getByPlaceholderText("SECRET_NAME")).toBeInTheDocument());
    fireEvent.change(screen.getByPlaceholderText("SECRET_NAME"), { target: { value: "NEW_KEY" } });
    fireEvent.change(screen.getByPlaceholderText("secret value"), { target: { value: "secret123" } });
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(api.setSecret).toHaveBeenCalledWith("NEW_KEY", "secret123"));
  });

  it("Save button is disabled until both fields are filled", async () => {
    render(<SecretsView />);
    await waitFor(() => expect(screen.getByText("Add")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Add"));
    await waitFor(() => expect(screen.getByText("Save")).toBeInTheDocument());
    expect(screen.getByText("Save")).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText("SECRET_NAME"), { target: { value: "KEY" } });
    expect(screen.getByText("Save")).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText("secret value"), { target: { value: "val" } });
    expect(screen.getByText("Save")).not.toBeDisabled();
  });
});
