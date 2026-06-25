import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ToolActivity } from "@/components/chat/ToolActivity";

describe("ToolActivity", () => {
  it("renders tool name and running state", () => {
    render(<ToolActivity toolName="execute_code" done={false} />);
    expect(screen.getByText("execute_code")).toBeInTheDocument();
    expect(screen.getByText("running…")).toBeInTheDocument();
  });

  it("does not show 'running...' when done", () => {
    render(
      <ToolActivity toolName="web_search" done={true} success={true} result={{ found: true }} />,
    );
    expect(screen.queryByText("running…")).not.toBeInTheDocument();
  });

  it("auto-expands and shows result on failure", () => {
    render(
      <ToolActivity toolName="execute_code" done={true} success={false} result={{ error: "crashed" }} />,
    );
    expect(screen.getByText("execute_code")).toBeInTheDocument();
    expect(screen.getByText("Result")).toBeInTheDocument();
    expect(screen.getByText(/crashed/)).toBeInTheDocument();
  });

  it("is collapsed by default on success", () => {
    render(
      <ToolActivity toolName="execute_code" done={true} success={true} result="ok" />,
    );
    expect(screen.queryByText("Result")).not.toBeInTheDocument();
    expect(screen.queryByText("Code")).not.toBeInTheDocument();
  });

  it("expands on click to show code and result", () => {
    render(
      <ToolActivity toolName="execute_code" code="const x = 42;" done={true} success={true} result="42" />,
    );

    // Collapsed by default
    expect(screen.queryByText("Code")).not.toBeInTheDocument();

    // Click to expand
    fireEvent.click(screen.getByText("execute_code"));

    // Now code and result are visible
    expect(screen.getByText("Code")).toBeInTheDocument();
    expect(screen.getByText("Result")).toBeInTheDocument();
    expect(screen.getByText(/const x = 42/)).toBeInTheDocument();
  });
});
