import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatusBar } from "@/components/chat/StatusBar";

describe("StatusBar", () => {
  it("returns null when not thinking and no tool running", () => {
    const { container } = render(<StatusBar thinking={false} toolName={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows 'Thinking...' when thinking is true", () => {
    render(<StatusBar thinking={true} toolName={null} />);
    expect(screen.getByText("Thinking…")).toBeInTheDocument();
  });

  it("shows tool name when a tool is running", () => {
    render(<StatusBar thinking={false} toolName="execute_code" />);
    expect(screen.getByText(/Running: execute_code/)).toBeInTheDocument();
  });

  it("prioritizes tool name over thinking state", () => {
    render(<StatusBar thinking={true} toolName="web_search" />);
    expect(screen.getByText(/Running: web_search/)).toBeInTheDocument();
    expect(screen.queryByText("Thinking…")).not.toBeInTheDocument();
  });
});
