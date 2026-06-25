"""Tests for PR 19: Extract public API facade — remove private attribute access."""

import pytest
import subprocess


def test_executor_has_public_properties():
    """ToolExecutor should expose public properties."""
    from src.tools.executor import ToolExecutor
    from src.tools.registry import ToolRegistry, ToolContext

    ctx = ToolContext()
    executor = ToolExecutor(registry=ToolRegistry(), ctx=ctx)

    assert hasattr(executor, "store")
    assert hasattr(executor, "secret_manager")
    assert hasattr(executor, "custom_tool_manager")
    assert hasattr(executor, "scheduler")
    assert hasattr(executor, "http_client")
    assert hasattr(executor, "llm_client")
    assert hasattr(executor, "exa_client")
    assert hasattr(executor, "ctx")


def test_agent_has_public_properties():
    """Agent should expose public properties."""
    from src.agent.agent import Agent

    agent = Agent(model_api="anthropic", model_name="x", model_api_key="key")

    assert hasattr(agent, "client")
    assert hasattr(agent, "system_prompt")
    assert hasattr(agent, "system_prompt_provider")
    assert hasattr(agent, "refresh_system_prompt")


def test_no_private_access_in_web():
    """No web module should access underscore-prefixed executor/agent attributes."""
    result = subprocess.run(
        ["grep", "-rn", r"executor\._ctx\._\|executor\._secret_manager\|executor\._scheduler\b\|executor\._custom_tool_manager\|agent\._system_prompt\b\|agent\._system_prompt_provider\|agent\._client\b",
         "src/web/"],
        capture_output=True,
        text=True,
    )
    violations = [line for line in result.stdout.strip().split("\n") if line]
    assert not violations, f"Found private attribute access in web:\n{result.stdout}"


def test_no_private_access_in_discord():
    """No Discord bot file should access underscore-prefixed executor/agent attributes."""
    result = subprocess.run(
        ["grep", "-rn", r"executor\._ctx\._\|executor\._secret_manager\|executor\._scheduler\b\|executor\._custom_tool_manager\|executor\._conversation\|executor\._chat_lock",
         "src/discord_bot/"],
        capture_output=True,
        text=True,
    )
    violations = [line for line in result.stdout.strip().split("\n") if line]
    assert not violations, f"Found private attribute access in Discord:\n{result.stdout}"


def test_no_private_access_in_main_py():
    """main.py should not access underscore-prefixed executor attributes."""
    result = subprocess.run(
        ["grep", "-n", r"executor\._scheduler\|executor\._secret_manager\|executor\._custom_tool_manager\|executor\._ctx",
         "main.py"],
        capture_output=True,
        text=True,
    )
    violations = [line for line in result.stdout.strip().split("\n") if line]
    assert not violations, f"Found private attribute access in main.py:\n{result.stdout}"
