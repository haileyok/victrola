"""Custom auth header (instead of the API key) for openapi model providers."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from src.agent.agent import Agent, OpenAICompatibleClient
from src.agent.llm import SubAgentLLM
from src.config import CONFIG, Config

CHAT_RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


def _recording_transport(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=CHAT_RESPONSE)

    return httpx.MockTransport(handler)


# --- main model client ---------------------------------------------------------


async def test_client_sends_custom_header_instead_of_bearer():
    seen: list[httpx.Request] = []
    client = OpenAICompatibleClient(
        api_key=None, model_name="m", endpoint="https://gw.test/v1", auth_header={"x-gateway-key": "secret"}
    )
    client._http = httpx.AsyncClient(transport=_recording_transport(seen))
    await client.complete([{"role": "user", "content": "hello"}], system="s")
    req = seen[0]
    assert str(req.url) == "https://gw.test/v1/chat/completions"
    assert req.headers["x-gateway-key"] == "secret"
    assert "authorization" not in req.headers
    await client.aclose()


async def test_client_without_header_still_uses_bearer_key():
    seen: list[httpx.Request] = []
    client = OpenAICompatibleClient(api_key="k", model_name="m", endpoint="https://api.test/v1")
    client._http = httpx.AsyncClient(transport=_recording_transport(seen))
    await client.complete([{"role": "user", "content": "hello"}], system="s")
    assert seen[0].headers["authorization"] == "Bearer k"
    await client.aclose()


def test_client_requires_key_or_header():
    with pytest.raises(ValueError):
        OpenAICompatibleClient(api_key=None, model_name="m", endpoint="https://x")


# --- Agent ------------------------------------------------------------------------


def test_agent_openapi_accepts_header_without_key():
    agent = Agent(
        model_api="openapi",
        model_name="m",
        model_api_key=None,
        model_endpoint="https://gw.test/v1",
        model_auth_header={"x-gateway-key": "secret"},
    )
    assert agent.client._auth_headers == {"x-gateway-key": "secret"}


def test_agent_openapi_still_requires_key_or_header():
    with pytest.raises(ValueError, match="model_api_key"):
        Agent(model_api="openapi", model_name="m", model_api_key=None, model_endpoint="https://x")


@pytest.mark.parametrize("api", ["anthropic", "openai", "umans"])
def test_agent_rejects_header_for_other_providers(api):
    with pytest.raises(ValueError, match="only supported for openapi"):
        Agent(model_api=api, model_name="m", model_api_key="k", model_auth_header={"x": "y"})


# --- sub-agent client -------------------------------------------------------------


async def test_sub_agent_sends_custom_header(monkeypatch):
    seen: list[httpx.Request] = []
    real = httpx.AsyncClient
    monkeypatch.setattr(
        "src.agent.llm.httpx.AsyncClient",
        lambda **kw: real(transport=_recording_transport(seen), **kw),
    )
    llm = SubAgentLLM(
        api="openapi", model="m", api_key=None, endpoint="https://gw.test/v1",
        auth_header={"x-gateway-key": "secret"},
    )
    assert await llm.complete("hello") == "hi"
    assert seen[0].headers["x-gateway-key"] == "secret"
    assert "authorization" not in seen[0].headers


def test_sub_agent_rejects_header_for_other_providers():
    with pytest.raises(ValueError, match="only supported for openapi"):
        SubAgentLLM(api="anthropic", model="m", api_key="k", auth_header={"x": "y"})


# --- config validation -----------------------------------------------------------


def test_config_header_pair_and_helper():
    c = Config(_env_file=None, model_auth_header_name="x-gateway-key", model_auth_header_value="v")
    assert c.model_auth_header() == {"x-gateway-key": "v"}
    assert c.sub_model_auth_header() is None


@pytest.mark.parametrize(
    "fields",
    [
        {"model_auth_header_name": "x-gateway-key"},  # value missing
        {"model_auth_header_value": "v"},  # name missing
        {"model_auth_header_name": "bad header", "model_auth_header_value": "v"},
        {"model_auth_header_name": "x:y", "model_auth_header_value": "v"},
        {"model_auth_header_name": "x", "model_auth_header_value": "v\r\nInjected: 1"},
        {"sub_model_auth_header_name": "x"},
    ],
)
def test_config_rejects_bad_header_settings(fields):
    with pytest.raises(ValidationError):
        Config(_env_file=None, **fields)


# --- build_services wiring --------------------------------------------------------


def _set(monkeypatch, **fields):
    for k, v in fields.items():
        monkeypatch.setattr(CONFIG, k, v)


def test_build_services_uses_header_for_main_and_sub(monkeypatch):
    import main

    _set(
        monkeypatch,
        model_api="openapi", model_endpoint="https://gw.test/v1", model_api_key="",
        model_auth_header_name="x-gateway-key", model_auth_header_value="secret",
        sub_model_api="openapi", sub_model_endpoint="https://gw.test/v1", sub_model_api_key="",
        sub_model_auth_header_name="", sub_model_auth_header_value="",
        exa_api_key="",
    )
    executor, agent = main.build_services(None, None, None, None)
    assert agent.client._auth_headers == {"x-gateway-key": "secret"}
    sub = executor.ctx.llm_client
    assert sub is not None and sub._auth_header == {"x-gateway-key": "secret"}  # reused


def test_build_services_sub_header_overrides(monkeypatch):
    import main

    _set(
        monkeypatch,
        model_api="openapi", model_endpoint="https://gw.test/v1", model_api_key="",
        model_auth_header_name="x-gateway-key", model_auth_header_value="main",
        sub_model_api="openapi", sub_model_endpoint="https://other.test/v1", sub_model_api_key="",
        sub_model_auth_header_name="x-other", sub_model_auth_header_value="sub",
        exa_api_key="",
    )
    executor, _ = main.build_services(None, None, None, None)
    assert executor.ctx.llm_client._auth_header == {"x-other": "sub"}


def test_build_services_ignores_header_for_non_openapi(monkeypatch, caplog):
    import main

    _set(
        monkeypatch,
        model_api="anthropic", model_api_key="k",
        model_auth_header_name="x-gateway-key", model_auth_header_value="secret",
        sub_model_api="anthropic", sub_model_api_key="",
        sub_model_auth_header_name="", sub_model_auth_header_value="",
        exa_api_key="",
    )
    with caplog.at_level("WARNING"):
        executor, agent = main.build_services(None, None, None, None)
    assert "only supported with MODEL_API=openapi" in caplog.text
    assert executor.ctx.llm_client._auth_header is None
