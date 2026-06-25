"""Tests for PR 8: Validate sub-agent key/provider consistency."""

import pytest
import logging
from src.agent.config_utils import resolve_sub_agent_key as _resolve_sub_agent_key


def test_same_provider_falls_back_to_main_key():
    """When providers match and sub key is empty, fall back to main key."""
    result = _resolve_sub_agent_key(
        model_api="anthropic",
        model_api_key="main-key",
        sub_model_api="anthropic",
        sub_model_api_key=None,
    )
    assert result == "main-key"


def test_same_provider_falls_back_on_empty_string():
    """Empty string sub key should also fall back to main key."""
    result = _resolve_sub_agent_key(
        model_api="anthropic",
        model_api_key="main-key",
        sub_model_api="anthropic",
        sub_model_api_key="",
    )
    assert result == "main-key"


def test_different_provider_no_sub_key_returns_none(caplog):
    """When providers differ and sub key is empty, return None with a warning."""
    with caplog.at_level(logging.WARNING):
        result = _resolve_sub_agent_key(
            model_api="anthropic",
            model_api_key="main-key",
            sub_model_api="openai",
            sub_model_api_key=None,
        )
    assert result is None
    assert any("differs from main provider" in r.message for r in caplog.records)


def test_different_provider_with_sub_key_returns_it():
    """When sub key is set, return it regardless of provider mismatch."""
    result = _resolve_sub_agent_key(
        model_api="anthropic",
        model_api_key="main-key",
        sub_model_api="openai",
        sub_model_api_key="sub-key",
    )
    assert result == "sub-key"


def test_same_provider_with_sub_key_returns_it():
    """When sub key is set and providers match, return the sub key."""
    result = _resolve_sub_agent_key(
        model_api="anthropic",
        model_api_key="main-key",
        sub_model_api="anthropic",
        sub_model_api_key="sub-key",
    )
    assert result == "sub-key"


def test_different_provider_empty_string_returns_none(caplog):
    """Empty string sub key with different provider should return None."""
    with caplog.at_level(logging.WARNING):
        result = _resolve_sub_agent_key(
            model_api="anthropic",
            model_api_key="main-key",
            sub_model_api="openapi",
            sub_model_api_key="",
        )
    assert result is None
