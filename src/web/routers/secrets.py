"""Secrets CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.tools.executor import ToolExecutor
from src.web.dependencies import get_executor
from src.web.schemas import SecretResponse, SetSecretRequest

router = APIRouter()


def _get_sm(executor: ToolExecutor):
    sm = executor.secret_manager
    if sm is None:
        raise HTTPException(500, "Secret manager not initialized")
    return sm


@router.get("/secrets", response_model=list[SecretResponse])
def list_secrets(
    executor: ToolExecutor = Depends(get_executor),
) -> list[SecretResponse]:
    sm = _get_sm(executor)
    result: list[SecretResponse] = []
    for name in sm.list_secret_names():
        value = sm.get_secret(name) or ""
        if len(value) > 8:
            masked = "*" * 8 + "..."
        else:
            masked = "*" * len(value)
        result.append(SecretResponse(name=name, masked_value=masked))
    return result


@router.post("/secrets", response_model=SecretResponse, status_code=201)
async def set_secret(
    body: SetSecretRequest,
    executor: ToolExecutor = Depends(get_executor),
) -> SecretResponse:
    sm = _get_sm(executor)
    await sm.set_secret(body.name, body.value)
    value = body.value
    if len(value) > 8:
        masked = "*" * 8 + "..."
    else:
        masked = "*" * len(value)
    return SecretResponse(name=body.name, masked_value=masked)


@router.delete("/secrets/{name}", status_code=204)
async def delete_secret(
    name: str,
    executor: ToolExecutor = Depends(get_executor),
) -> None:
    sm = _get_sm(executor)
    result = await sm.delete_secret(name)
    if "not found" in result:
        raise HTTPException(404, f"Secret '{name}' not found")
