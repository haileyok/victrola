import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TokenStats } from "@/components/chat/TokenStats";

describe("TokenStats", () => {
  it("returns null when calls is 0", () => {
    const { container } = render(
      <TokenStats ctx={0} tps={null} input={0} output={0} calls={0} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("displays all stats when calls > 0", () => {
    render(
      <TokenStats ctx={100000} tps={42.5} input={1500} output={300} calls={3} />,
    );
    expect(screen.getByText(/ctx:.*100,000/)).toBeInTheDocument();
    expect(screen.getByText(/tps:.*42.5/)).toBeInTheDocument();
    expect(screen.getByText(/in:.*1,500/)).toBeInTheDocument();
    expect(screen.getByText(/out:.*300/)).toBeInTheDocument();
    expect(screen.getByText(/calls:.*3/)).toBeInTheDocument();
  });

  it("hides tps when null", () => {
    render(
      <TokenStats ctx={0} tps={null} input={100} output={50} calls={1} />,
    );
    expect(screen.queryByText(/tps:/)).not.toBeInTheDocument();
  });
});
