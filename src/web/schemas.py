"""Pydantic request/response models for the web API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


# -- request models --


class CreateSessionRequest(BaseModel):
    title: str = ""


class ChatRequest(BaseModel):
    message: str
    images: list[dict[str, str]] | None = None


class SetSecretRequest(BaseModel):
    name: str
    value: str


class CreateScheduleRequest(BaseModel):
    name: str
    schedule: str
    prompt: str


# -- response models --


class SessionResponse(BaseModel):
    rkey: str
    title: str = ""
    createdAt: str = ""


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    cursor: str | None = None


class MessageResponse(BaseModel):
    id: int
    sessionId: str | None = None
    sender: str
    content: str
    createdAt: str


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]
    cursor: str | None = None


class StatusResponse(BaseModel):
    model: str
    discord: bool
    schedules: int
    secrets: int
    custom_tools_approved: int
    custom_tools_pending: int
    mcp_servers: int = 0
    mcp_tools_approved: int = 0


class ToolSummary(BaseModel):
    name: str
    description: str
    approved: bool
    requires_net: bool
    secrets: list[str]


class ToolDetailResponse(BaseModel):
    name: str
    description: str
    approved: bool
    requires_net: bool
    code: str
    parameters: dict
    secrets: list[dict[str, str]]  # [{"name": ..., "status": "set"|"missing"}]


class SecretResponse(BaseModel):
    name: str
    masked_value: str


class ScheduleResponse(BaseModel):
    name: str
    schedule: str
    prompt: str
    enabled: bool
    last_run: str | None = None
    next_run: str | None = None


class SystemPromptResponse(BaseModel):
    text: str
    char_count: int
    token_estimate: int


# -- MCP models --


class CreateMCPServerRequest(BaseModel):
    name: str
    transport: Literal["sse", "stdio"]
    url: str | None = None
    command: str | None = None
    args: list[str] = []
    auth_token_secret: str | None = None
    env_secrets: list[str] = []
    enabled: bool = True


class MCPServerSummary(BaseModel):
    name: str
    transport: str
    enabled: bool
    connected: bool
    tools_total: int
    tools_approved: int


class MCPToolSummary(BaseModel):
    name: str
    description: str
    approved: bool


class MCPServerDetail(BaseModel):
    name: str
    transport: str
    url: str | None
    command: str | None
    args: list[str]
    auth_token_secret: str | None
    auth_token_status: str  # "set" | "missing" | "none"
    env_secrets: list[dict[str, str]]  # [{"name": ..., "status": "set"|"missing"}]
    enabled: bool
    connected: bool
    tools: list[MCPToolSummary]
