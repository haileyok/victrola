from typing import Any

from rich.syntax import Syntax
from textual.containers import Vertical
from textual.widgets import Collapsible, Markdown, Static


class MessageBubble(Static):
    """A single chat message with role label and content."""

    def __init__(self, role: str, content: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._role = role
        self._content = content

    def compose(self):
        role_display = self._role.capitalize()
        yield Static(f"{role_display}:", classes="role-label")
        # Markdown rendering for assistant responses (lists, code blocks, etc.).
        # User/system content renders as plain text — arbitrary strings with
        # stray brackets, unclosed fences, etc. can otherwise crash or mangle
        # the Markdown/markup parser.
        if self._role == "assistant":
            yield Markdown(self._content)
        else:
            yield Static(self._content, markup=False)

    def on_mount(self) -> None:
        self.add_class(f"{self._role}-message")


# Max characters of result output to render in the expanded view.
MAX_RESULT_DISPLAY = 4000


def _format_result(result: dict[str, Any] | None) -> str:
    """Pretty-format a tool result for display, truncating if enormous."""
    if result is None:
        return "(no result)"

    # common `execute_code` shape: {success, output?, error?, debug?}
    parts: list[str] = []
    if "error" in result:
        parts.append(f"error: {result['error']}")
    if "output" in result:
        out = result["output"]
        if isinstance(out, (dict, list)):
            import json

            rendered = json.dumps(out, indent=2, default=str)
        else:
            rendered = str(out)
        parts.append(rendered)
    debug = result.get("debug") or []
    if debug:
        parts.append("--- debug ---")
        parts.extend(str(d) for d in debug)

    if not parts:
        # fall back to the raw dict
        import json

        parts.append(json.dumps(result, indent=2, default=str))

    text = "\n".join(parts)
    if len(text) > MAX_RESULT_DISPLAY:
        text = text[:MAX_RESULT_DISPLAY] + f"\n... (truncated, {len(text)} total chars)"
    return text


class ToolActivity(Vertical):
    """Collapsible block showing a tool call: code on top, result below.

    Starts collapsed with just a one-line header. Expands on click (Collapsible
    handles the toggle). Auto-expands on error.
    """

    DEFAULT_CLASSES = "tool-activity"

    def __init__(self, tool_name: str, code: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._code = code
        self._status = "running"
        self._result: dict[str, Any] | None = None
        self._collapsible: Collapsible | None = None
        self._result_widget: Static | None = None

    def _header_title(self) -> str:
        status_icon = {
            "running": "…",
            "success": "✓",
            "error": "✗",
        }.get(self._status, "")
        code_len = len(self._code)
        return f"▸ {self._tool_name}  ·  {code_len} chars  ·  {status_icon}"

    def compose(self):
        coll = Collapsible(title=self._header_title(), collapsed=True)
        self._collapsible = coll
        with coll:
            if self._code:
                yield Static(
                    Syntax(self._code, "typescript", theme="monokai", word_wrap=True),
                    classes="tool-code",
                )
            # markup=False: treat result text as plain text so URLs / JSON /
            # arbitrary content with `[...]` don't trip Rich's markup parser
            result_static = Static(
                "(running…)", classes="tool-result", markup=False
            )
            self._result_widget = result_static
            yield result_static

    def set_result(self, result: dict[str, Any], success: bool) -> None:
        """Update the widget with the tool's result after completion."""
        self._result = result
        self._status = "success" if success else "error"
        if self._collapsible is not None:
            self._collapsible.title = self._header_title()
        if self._result_widget is not None:
            self._result_widget.update(_format_result(result))
        # auto-expand on error so failures are immediately visible
        if not success and self._collapsible is not None:
            self._collapsible.collapsed = False
        # style the block by final status
        self.set_class(self._status == "error", "tool-activity-error")
        self.set_class(self._status == "success", "tool-activity-success")
