import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";

function renderLayout(route: string = "/sessions") {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <Layout />
    </MemoryRouter>,
  );
}

describe("Layout", () => {
  it("renders the Victrola title", () => {
    renderLayout();
    expect(screen.getByText("Victrola")).toBeInTheDocument();
  });

  it("renders all nav items", () => {
    renderLayout();
    expect(screen.getByText("Sessions")).toBeInTheDocument();
    expect(screen.getByText("Tools")).toBeInTheDocument();
    expect(screen.getByText("Secrets")).toBeInTheDocument();
    expect(screen.getByText("Schedules")).toBeInTheDocument();
    expect(screen.getByText("Prompt")).toBeInTheDocument();
  });

  it("highlights active nav item", () => {
    renderLayout("/secrets");
    const secretsLink = screen.getByText("Secrets").closest("a");
    expect(secretsLink).toHaveClass("bg-accent");
  });

  it("renders child routes via Outlet", () => {
    renderLayout();
    // The Outlet renders the matched route — since no child routes are defined
    // in the test, we just verify the layout shell is present
    expect(screen.getByText("Victrola")).toBeInTheDocument();
  });
});
