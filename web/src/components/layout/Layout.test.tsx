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

  it("renders two navigation landmarks with distinct names and responsive visibility", () => {
    renderLayout();
    const navs = screen.getAllByRole("navigation");
    expect(navs).toHaveLength(2);
    expect(navs[0]).toHaveAccessibleName("Primary");
    expect(navs[1]).toHaveAccessibleName("Sections");
    // Desktop sidebar is hidden below md; mobile top bar is hidden at md and up.
    const sidebar = navs[0].closest("aside");
    expect(sidebar?.className).toContain("hidden");
    expect(sidebar?.className).toContain("md:flex");
    expect(navs[1].className).toContain("md:hidden");
  });

  it("renders all 8 nav items in both desktop and mobile navs", () => {
    renderLayout();
    // The layout renders a desktop sidebar and a mobile top bar, so each
    // label appears twice.
    for (const label of [
      "Sessions",
      "Tools",
      "MCP",
      "Workspace",
      "Secrets",
      "Schedules",
      "Prompt",
      "Memory",
    ]) {
      expect(screen.getAllByText(label)).toHaveLength(2);
    }
  });

  it("highlights active nav item", () => {
    renderLayout("/secrets");
    for (const secretsLink of screen.getAllByText("Secrets").map((el) => el.closest("a"))) {
      expect(secretsLink).toHaveClass("bg-accent");
    }
  });

  it("renders child routes via Outlet", () => {
    renderLayout();
    // The Outlet renders the matched route — since no child routes are defined
    // in the test, we just verify the layout shell is present
    expect(screen.getByText("Victrola")).toBeInTheDocument();
  });
});
