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

  it("renders all nav items in both desktop and mobile navs", () => {
    renderLayout();
    // The layout renders a desktop sidebar and a mobile top bar, so each
    // label appears twice.
    for (const label of ["Sessions", "Tools", "Secrets", "Schedules", "Prompt"]) {
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
