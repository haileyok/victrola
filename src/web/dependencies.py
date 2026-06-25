"""Shared state accessors for FastAPI route dependencies."""

from __future__ import annotations

from fastapi import Request

from src.agent.agent import Agent
from src.agent.conversation import ConversationManager
from src.tools.executor import ToolExecutor


def get_executor(request: Request) -> ToolExecutor:
    return request.app.state.executor


def get_agent(request: Request) -> Agent:
    return request.app.state.agent


def get_conversation_manager(request: Request) -> ConversationManager:
    return request.app.state.conversation_manager
