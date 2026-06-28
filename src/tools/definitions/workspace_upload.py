"""Upload a workspace file to a registered MCP server (e.g. the Supernote cloud).

The agent runs in a Deno sandbox whose tool-call bridge caps arguments at ~64 KiB,
so it can't pass a base64'd PDF to an MCP `upload_file` tool inline. This
host-side tool takes a workspace **path** instead: it reads the file in Python
(safely scoped to the workspace), base64-encodes it host-side, and calls the MCP
server's `upload_file(filename, content_base64, dest)` over HTTP — nothing large
crosses the Deno bridge.

Pairs with `web.save_url_as_pdf` (which saves a PDF into the workspace) to send a
rendered page to the device.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any

from src.config import CONFIG
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# Fixed destination MCP server. NOT agent-controllable: letting the model choose
# the server would let a prompt-injected agent route the file's bytes to any
# other connected MCP (exfiltration).
_UPLOAD_SERVER = "supernote"


def _read_workspace_file(rel_path: str) -> tuple[bytes, str]:
    """Read a workspace-relative file safely; return (bytes, own_filename).

    Refuses anything that would read outside the workspace: path traversal,
    symlink components, the final component being a symlink (`O_NOFOLLOW`), and
    multi-linked files (a hardlink could share an inode with an outside file).
    """
    if not rel_path or not rel_path.strip():
        raise ValueError("path is required")
    workspace = Path(CONFIG.workspace_dir).resolve()
    target = Path(os.path.normpath(workspace / rel_path))
    if target != workspace and workspace not in target.parents:
        raise ValueError("path is outside the workspace")
    cur = workspace
    for part in target.relative_to(workspace).parts:
        cur = cur / part
        if cur.is_symlink():
            raise ValueError("symlinks are not permitted in the workspace path")
    try:
        # O_NONBLOCK so opening a FIFO/device doesn't block waiting for a writer;
        # O_NOFOLLOW so a final-component symlink can't redirect the read.
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError(f"cannot open {rel_path!r}: {exc}")
    # Validate the fd before fdopen (fdopen on a dir raises IsADirectoryError).
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("path is not a regular file")
        if st.st_nlink > 1:
            raise ValueError("refusing a multi-linked file (possible hardlink outside the workspace)")
        if st.st_size > MAX_UPLOAD_BYTES:
            raise ValueError(f"file exceeds {MAX_UPLOAD_BYTES} bytes")
        with os.fdopen(fd, "rb") as f:  # takes ownership of fd, closes on exit
            # Capped read: enforce the limit on the bytes actually read, in case
            # the file grew after fstat.
            data = f.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise ValueError(f"file exceeds {MAX_UPLOAD_BYTES} bytes")
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return data, target.name


def _safe_name(name: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/")).strip()
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return base or "file"


@TOOL_REGISTRY.tool(
    name="files.upload_workspace_file",
    description=(
        "Upload a file from the workspace to the Supernote cloud (it syncs to "
        "the device). Pass the workspace-relative path; the file is read and "
        "base64-encoded host-side and sent over HTTP, so it works for large files "
        "that can't be passed inline. Use this to send a workspace PDF (e.g. one "
        "from web.save_url_as_pdf) to the device."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Workspace-relative path of the file to upload, e.g. 'article.pdf'.",
        ),
        ToolParameter(
            name="dest",
            type="string",
            description="Destination folder (Document or Note, or a subpath like Note/Research). Defaults to Document.",
            required=False,
            default="Document",
        ),
        ToolParameter(
            name="filename",
            type="string",
            description="Name to store it as; defaults to the file's own name.",
            required=False,
            default=None,
        ),
    ],
)
async def upload_workspace_file(
    ctx: ToolContext,
    path: str,
    dest: str = "Document",
    filename: str | None = None,
) -> dict[str, Any]:
    try:
        data, own_name = _read_workspace_file(path)
    except ValueError as exc:
        return {"error": str(exc)}

    if ctx.mcp_manager is None:
        return {"error": "MCP manager is not available"}

    name = _safe_name(filename or own_name)
    b64 = base64.b64encode(data).decode("ascii")
    result = await ctx.mcp_manager.call_tool(
        _UPLOAD_SERVER, "upload_file", {"filename": name, "content_base64": b64, "dest": dest}
    )
    if isinstance(result, dict) and result.get("error"):
        return {"error": f"upload via '{_UPLOAD_SERVER}' failed: {result['error']}"}

    logger.info(
        "upload_workspace_file: %s -> %s/%s (%d bytes)",
        path, dest, name, len(data),
    )
    return {
        "uploaded": True,
        "filename": name,
        "dest": dest,
        "size_bytes": len(data),
        "result": result,
    }
