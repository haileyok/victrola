"""Workspace file browsing, reading, and deletion endpoints."""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from src.config import CONFIG
from src.tools.executor import ToolExecutor
from src.web.dependencies import get_executor

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_READ_SIZE = 256 * 1024  # 256KB cap for file content responses


def _resolve_workspace_path(path: str) -> tuple[Path, Path]:
    """Resolve a relative path against the workspace root.

    Returns ``(workspace_root, resolved_target)``. Raises 403 if the
    resolved path escapes the workspace.
    """
    workspace = Path(CONFIG.workspace_dir).resolve()
    target = (workspace / path).resolve()

    try:
        target.relative_to(workspace)
    except ValueError:
        raise HTTPException(403, "Path outside workspace")

    return workspace, target


def _compute_workspace_size(workspace: Path) -> int:
    """Walk the workspace tree and sum file sizes."""
    return sum(f.stat().st_size for f in workspace.rglob("*") if f.is_file())


@router.get("/workspace")
async def list_workspace(
    path: str = Query(default=""),
    executor: ToolExecutor = Depends(get_executor),
):
    """List files and directories at a given path within the workspace."""
    workspace, target = _resolve_workspace_path(path)

    if not target.exists():
        raise HTTPException(404, "Path not found")

    if target.is_file():
        stat = target.stat()
        return {
            "type": "file",
            "name": target.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }

    # List directory contents
    entries = []
    for entry in sorted(target.iterdir()):
        try:
            stat = entry.stat()
            entries.append(
                {
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "size": stat.st_size if entry.is_file() else None,
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        except OSError:
            # broken symlink or permission issue — skip
            continue

    # Compute total workspace size in a threadpool to avoid blocking the event loop.
    total_size = await asyncio.to_thread(_compute_workspace_size, workspace)

    # Log a warning if workspace exceeds the soft limit
    max_size = CONFIG.workspace_max_size_mb * 1024 * 1024
    if total_size > max_size:
        logger.warning(
            "Workspace size %d bytes exceeds limit of %d bytes (%d MB)",
            total_size,
            max_size,
            CONFIG.workspace_max_size_mb,
        )

    return {
        "path": path,
        "entries": entries,
        "total_size_bytes": total_size,
        "max_size_bytes": max_size,
    }


@router.get("/workspace/file")
async def read_workspace_file(
    path: str = Query(...),
    executor: ToolExecutor = Depends(get_executor),
):
    """Read a file's contents from the workspace."""
    _, target = _resolve_workspace_path(path)

    if not target.is_file():
        raise HTTPException(404, "File not found")

    stat = target.stat()

    content = target.read_text(errors="replace")
    if len(content) > MAX_READ_SIZE:
        content = content[:MAX_READ_SIZE] + "\n... (truncated)"

    return {
        "path": path,
        "content": content,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


@router.delete("/workspace/file")
async def delete_workspace_file(
    path: str = Query(...),
    executor: ToolExecutor = Depends(get_executor),
):
    """Delete a file or directory from the workspace."""
    workspace, target = _resolve_workspace_path(path)

    # Reject deleting the workspace root itself
    if target == workspace:
        raise HTTPException(400, "Cannot delete workspace root")

    if not target.exists():
        raise HTTPException(404, "File not found")

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()

    return {"deleted": path}


@router.post("/workspace/dir")
async def create_workspace_dir(
    path: str = Query(...),
    executor: ToolExecutor = Depends(get_executor),
):
    """Create a directory in the workspace."""
    _, target = _resolve_workspace_path(path)

    target.mkdir(parents=True, exist_ok=True)
    return {"created": path}
