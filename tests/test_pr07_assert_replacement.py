"""Tests for PR 7: Replace assert with explicit validation."""

import pytest

from src.agent.agent import Agent


def test_agent_raises_on_empty_api_key():
    """Agent with empty model_api_key should raise ValueError."""
    with pytest.raises(ValueError, match="model_api_key"):
        Agent(model_api="anthropic", model_name="x", model_api_key="")


def test_agent_raises_on_none_api_key():
    """Agent with None model_api_key should raise ValueError."""
    with pytest.raises(ValueError, match="model_api_key"):
        Agent(model_api="anthropic", model_name="x", model_api_key=None)


def test_agent_openapi_raises_without_endpoint():
    """Agent with openapi but no endpoint should raise ValueError."""
    with pytest.raises(ValueError, match="model_endpoint"):
        Agent(model_api="openapi", model_name="x", model_api_key="key")


def test_no_asserts_in_target_files():
    """No assert statements should remain in the target files."""
    import subprocess
    files = [
        "src/agent/agent.py",
        "src/agent/conversation.py",
        "src/store/store.py",
        "src/tools/custom.py",
        "src/tools/definitions/memory.py",
        "src/tools/executor.py",
        "src/discord_bot/bot.py",
        "src/scheduler/schedule.py",
    ]
    for f in files:
        result = subprocess.run(
            ["grep", "-n", "^[[:space:]]*assert ", f],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, f"Found assert in {f}:\n{result.stdout}"
