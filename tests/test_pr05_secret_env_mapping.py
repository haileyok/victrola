"""Tests for PR 5: Fix secret name→env-var mapping (stop .upper())."""

import pytest
from unittest.mock import MagicMock

from src.tools.custom import CustomTool, CustomToolManager


def test_build_env_uses_exact_secret_name():
    """_build_env should inject secrets using their exact configured name (no .upper())."""
    store = MagicMock()
    executor = MagicMock()
    secret_manager = MagicMock()
    secret_manager.get_secret = MagicMock(return_value="secret_value")

    manager = CustomToolManager(store=store, executor=executor, secret_manager=secret_manager)

    tool = CustomTool(
        name="test_tool",
        description="test",
        parameters={},
        code="output('hello')",
        approved=True,
        secrets=["my_api_key"],
    )

    env = manager._build_env(tool)
    assert "my_api_key" in env
    assert env["my_api_key"] == "secret_value"
    # Should NOT have the uppercased version
    assert "MY_API_KEY" not in env


def test_build_env_preserves_case():
    """_build_env should preserve the original case of secret names."""
    store = MagicMock()
    executor = MagicMock()
    secret_manager = MagicMock()
    secret_manager.get_secret = MagicMock(return_value="val123")

    manager = CustomToolManager(store=store, executor=executor, secret_manager=secret_manager)

    tool = CustomTool(
        name="test_tool",
        description="test",
        parameters={},
        code="output('hello')",
        approved=True,
        secrets=["MixedCase_Name"],
    )

    env = manager._build_env(tool)
    assert "MixedCase_Name" in env
    assert env["MixedCase_Name"] == "val123"


def test_build_env_no_secrets():
    """_build_env should return empty dict when tool has no secrets."""
    store = MagicMock()
    executor = MagicMock()

    manager = CustomToolManager(store=store, executor=executor, secret_manager=None)

    tool = CustomTool(
        name="test_tool",
        description="test",
        parameters={},
        code="output('hello')",
        approved=True,
        secrets=[],
    )

    env = manager._build_env(tool)
    assert env == {}


def test_build_env_skips_missing_secrets():
    """_build_env should skip secrets that are not set (None/empty)."""
    store = MagicMock()
    executor = MagicMock()
    secret_manager = MagicMock()
    secret_manager.get_secret = MagicMock(return_value=None)

    manager = CustomToolManager(store=store, executor=executor, secret_manager=secret_manager)

    tool = CustomTool(
        name="test_tool",
        description="test",
        parameters={},
        code="output('hello')",
        approved=True,
        secrets=["missing_secret"],
    )

    env = manager._build_env(tool)
    assert "missing_secret" not in env
    assert env == {}
