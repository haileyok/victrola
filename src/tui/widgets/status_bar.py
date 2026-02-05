from textual.widgets import Static


class AgentStatusBar(Static):
    """Shows agent activity status — hidden when idle."""

    DEFAULT_CSS = ""

    def on_mount(self) -> None:
        self.add_class("hidden")

    def show_thinking(self) -> None:
        self.update("Thinking...")
        self.remove_class("hidden")

    def show_tool(self, tool_name: str) -> None:
        self.update(f"Running: {tool_name}")
        self.remove_class("hidden")

    def hide(self) -> None:
        self.add_class("hidden")
        self.update("")
