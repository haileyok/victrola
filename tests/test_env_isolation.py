"""Tests for execute_code env isolation (Bug 07)."""

import os
from unittest.mock import patch

from src.tools.executor import ToolExecutor


def test_minimal_env_excludes_arbitrary_vars():
    """_minimal_env should include only PATH and HOME — no secrets."""
    sentinel = "VICTROLA_TEST_SECRET_BUG07"
    with patch.dict(os.environ, {sentinel: "leaked", "PATH": "/usr/bin", "HOME": "/tmp"}):
        env = ToolExecutor._minimal_env()

    assert sentinel not in env, (
        f"_minimal_env leaked parent env var {sentinel}"
    )
    assert env.get("PATH") == "/usr/bin"
    assert env.get("HOME") == "/tmp"


def test_minimal_env_with_extras():
    """_minimal_env should layer on explicitly-provided vars only."""
    with patch.dict(os.environ, {"PATH": "/usr/bin", "HOME": "/tmp", "SECRET": "x"}):
        env = ToolExecutor._minimal_env({"GRANTED_KEY": "value"})

    assert "SECRET" not in env
    assert env["GRANTED_KEY"] == "value"
    assert "PATH" in env
