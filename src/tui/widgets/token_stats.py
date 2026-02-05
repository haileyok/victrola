from textual.widgets import Static


def _fmt(n: int) -> str:
    """Format token count with k/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


class TokenStatsBar(Static):
    """Displays token usage stats for the current session.

    Shows both the latest-call snapshot (ctx size, tps) and cumulative totals.
    """

    def __init__(self, context_limit: int = 200_000, **kwargs) -> None:
        super().__init__("", **kwargs)
        # cumulative (session-wide)
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_creation_tokens = 0
        self._llm_calls = 0
        # latest-call snapshot
        self._context_limit = context_limit
        self._latest_ctx: int | None = None
        self._latest_tps: float | None = None
        # render placeholder so the bar is visible before the first LLM call
        self._render_stats()

    def add_usage(self, usage: dict[str, int], elapsed_s: float | None = None) -> None:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_create = usage.get("cache_creation_input_tokens", 0)

        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cache_read_tokens += cache_read
        self._cache_creation_tokens += cache_create
        self._llm_calls += 1

        # effective context size = everything the model had to look at
        self._latest_ctx = input_tokens + cache_read + cache_create

        if elapsed_s and elapsed_s > 0 and output_tokens > 0:
            self._latest_tps = output_tokens / elapsed_s

        self._render_stats()

    def _render_stats(self) -> None:
        parts: list[str] = []
        if self._latest_ctx is not None:
            pct = (
                f" ({self._latest_ctx / self._context_limit:.0%})"
                if self._context_limit
                else ""
            )
            parts.append(
                f"ctx {_fmt(self._latest_ctx)}/{_fmt(self._context_limit)}{pct}"
            )
        else:
            parts.append(f"ctx —/{_fmt(self._context_limit)}")
        if self._latest_tps is not None:
            parts.append(f"tps {self._latest_tps:.1f}")
        else:
            parts.append("tps —")
        parts.append(f"in {_fmt(self._input_tokens)}")
        parts.append(f"out {_fmt(self._output_tokens)}")
        if self._cache_read_tokens:
            parts.append(f"cache {_fmt(self._cache_read_tokens)}")
        parts.append(f"calls {self._llm_calls}")
        self.update(" · ".join(parts))
