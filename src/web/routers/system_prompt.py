"""System prompt view endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from src.agent.agent import Agent
from src.web.dependencies import get_agent
from src.web.schemas import SystemPromptResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/system-prompt", response_model=SystemPromptResponse)
async def get_system_prompt(
    agent: Agent = Depends(get_agent),
) -> SystemPromptResponse:
    text: str
    provider = agent.system_prompt_provider
    if provider is not None:
        try:
            text = await provider()
        except Exception as e:
            logger.exception("Failed to fetch current system prompt")
            text = f"(error fetching prompt: {e})"
    else:
        text = agent.system_prompt or "(no prompt set)"

    return SystemPromptResponse(
        text=text,
        char_count=len(text),
        token_estimate=len(text) // 4,
    )
