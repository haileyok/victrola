import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageBubble } from "@/components/chat/MessageBubble";

describe("MessageBubble", () => {
  it("renders user messages with 'You' label", () => {
    render(<MessageBubble role="user" content="Hello world" />);
    expect(screen.getByText("You")).toBeInTheDocument();
    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("renders assistant messages with 'Agent' label and markdown", () => {
    render(<MessageBubble role="assistant" content="**Bold** text" />);
    expect(screen.getByText("Agent")).toBeInTheDocument();
    // markdown renders bold as <strong>
    expect(screen.getByText("Bold")).toBeInTheDocument();
  });

  it("renders system messages with 'System' label", () => {
    render(<MessageBubble role="system" content="Something went wrong" />);
    expect(screen.getByText("System")).toBeInTheDocument();
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("renders markdown code blocks for assistant messages", () => {
    render(
      <MessageBubble
        role="assistant"
        content={"```python\nprint('hello')\n```"}
      />,
    );
    expect(screen.getByText(/print/)).toBeInTheDocument();
  });

  it("renders markdown links for assistant messages", () => {
    render(
      <MessageBubble
        role="assistant"
        content="[Click here](https://example.com)"
      />,
    );
    const link = screen.getByRole("link", { name: "Click here" });
    expect(link).toHaveAttribute("href", "https://example.com");
  });
});
