import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { api, ApiError } from "@/lib/api";

describe("ApiError", () => {
  it("preserves status and detail from string detail", () => {
    const err = new ApiError(404, "not found");
    expect(err.status).toBe(404);
    expect(err.detail).toBe("not found");
    expect(err.message).toBe("not found");
    expect(err.name).toBe("ApiError");
  });

  it("extracts message from structured detail object", () => {
    const err = new ApiError(400, { message: "missing secrets", missing_secrets: ["KEY1"] });
    expect(err.status).toBe(400);
    expect(err.message).toBe("missing secrets");
    expect((err.detail as { missing_secrets: string[] }).missing_secrets).toEqual(["KEY1"]);
  });

  it("falls back to HTTP status when detail has no message", () => {
    const err = new ApiError(500, { unrelated: "field" });
    expect(err.message).toBe("HTTP 500");
  });
});

describe("api client", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("encodes session ID in URL paths", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      status: 204,
      json: async () => ({}),
    });

    await api.deleteSession("session/with/slashes");

    const calledUrl = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(calledUrl).toContain("session%2Fwith%2Fslashes");
  });

  it("encodes tool names in URL paths", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ name: "test", description: "", approved: false, requires_net: false, code: "", parameters: {}, secrets: [] }),
    });

    await api.getTool("tool with spaces");

    const calledUrl = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(calledUrl).toContain("tool%20with%20spaces");
  });

  it("encodes secret names in URL paths", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      status: 204,
      json: async () => ({}),
    });

    await api.deleteSecret("MY#SECRET?");

    const calledUrl = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(calledUrl).toContain("MY%23SECRET%3F");
  });

  it("encodes schedule names in URL paths", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ name: "", schedule: "", prompt: "", enabled: true, last_run: null, next_run: null }),
    });

    await api.enableSchedule("task/with?special#chars");

    const calledUrl = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(calledUrl).toContain("task%2Fwith%3Fspecial%23chars");
  });

  it("throws ApiError on non-OK response", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 403,
      json: async () => ({ detail: "forbidden" }),
    });

    await expect(api.listSessions()).rejects.toThrow(ApiError);
    await expect(api.listSessions()).rejects.toThrow("forbidden");
  });

  it("handles JSON parse failure in error response", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => { throw new Error("parse error"); },
    });

    await expect(api.getStatus()).rejects.toThrow(ApiError);
  });

  it("returns undefined for 204 responses", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      status: 204,
      json: async () => ({}),
    });

    const result = await api.deleteSession("abc");
    expect(result).toBeUndefined();
  });
});
