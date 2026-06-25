import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

// Auto-cleanup after each test
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
