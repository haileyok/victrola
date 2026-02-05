import logging
from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

LENGTH_GUIDANCE = {
    "short": "Respond in 2-3 sentences maximum.",
    "medium": "Respond in 1-2 short paragraphs.",
    "long": "Respond in up to 4 paragraphs with key details preserved.",
}


@TOOL_REGISTRY.tool(
    name="summarize.summarize",
    description="Summarize text using a sub-agent LLM. Useful for condensing long tool outputs, articles, or thread contents before presenting to the user.",
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="Text to summarize",
        ),
        ToolParameter(
            name="instructions",
            type="string",
            description="Specific summarization instructions (e.g. 'focus on technical claims')",
            required=False,
        ),
        ToolParameter(
            name="max_length",
            type="string",
            description='"short", "medium", or "long"',
            required=False,
            default="medium",
        ),
    ],
)
async def summarize(
    ctx: ToolContext,
    text: str,
    instructions: str | None = None,
    max_length: str = "medium",
) -> dict[str, Any]:
    if not text:
        return {"error": "text is required"}

    llm = ctx.llm_client
    length_guide = LENGTH_GUIDANCE.get(max_length, LENGTH_GUIDANCE["medium"])

    system = "You are a precise summarization assistant. Preserve key facts, names, and numbers. Do not add information not present in the source text."

    prompt_parts = [f"Summarize the following text. {length_guide}"]
    if instructions:
        prompt_parts.append(f"Additional instructions: {instructions}")
    prompt_parts.append(f"\n---\n{text}\n---")

    prompt = "\n\n".join(prompt_parts)

    # truncate input if extremely long (sub-agent has limited context)
    if len(prompt) > 100_000:
        prompt = prompt[:100_000] + "\n... (truncated)"

    try:
        summary = await llm.complete(prompt, system=system)
        return {"summary": summary}
    except Exception as e:
        logger.exception("summarize failed")
        return {"error": str(e)}
