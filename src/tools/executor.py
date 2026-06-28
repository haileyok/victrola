import asyncio
import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from src.tools.registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

DENO_DIR = Path(__file__).parent / "deno"

# security limits for deno execution
MAX_CODE_SIZE = 50_000  # max input code size in characters
MAX_TOOL_CALLS = 25  # max number of tool calls per execution
MAX_OUTPUT_SIZE = 1_000_000  # max total output size in bytes
MAX_EXECUTION_TIME = 60.0  # total wall-clock timeout in seconds
DENO_MEMORY_LIMIT_MB = 256  # v8 heap limit


def _resolve_workspace() -> str:
    """Resolve the workspace directory to an absolute path.

    Rejects paths containing commas — Deno's --allow-read/--allow-write flags
    use commas as path separators, so a comma in the workspace path would
    silently grant access to extra directories.
    """
    from src.config import CONFIG

    workspace = str(Path(CONFIG.workspace_dir).resolve())
    if "," in workspace:
        raise ValueError(
            f"workspace_dir must not contain commas (Deno treats commas as "
            f"path separators in permission flags): {workspace!r}"
        )
    return workspace


class WorkspaceError(Exception):
    """The workspace is in an unsafe state to grant to Deno.

    Raised when the workspace contains a symlink (which could escape the
    scoped write permission) or is already over its size quota.
    """


def _scan_workspace_or_raise(workspace: str) -> None:
    """Pre-execution guard run before Deno is granted workspace access.

    Deno authorizes a read/write *through* a link based on the link's location,
    not its target, so a link inside the workspace pointing/refers outside turns
    ``--allow-write=<workspace>`` into a write-outside primitive. The agent
    cannot create escaping links (Deno denies symlink creation and won't link or
    rename across the permission boundary), but the workspace is persistent and
    an operator, a restored backup, another local process, or a future upload
    path could plant one — so we refuse to run rather than follow it.

    1. Reject ANY symlink in the workspace tree.
    2. Reject any hardlink whose inode also has a link OUTSIDE the workspace
       (writing it would modify out-of-workspace data). Hardlinks that live
       entirely inside the workspace are allowed.
    3. Refuse to START a run while the workspace already exceeds the configured
       size quota. This is a next-run backpressure check, not a per-run cap: a
       single permitted run can still write past the limit.

    A single ``scandir`` traversal collects everything; the symlink check
    short-circuits. Cost is O(workspace entries) per execution — negligible for
    the expected small workspaces.

    Threat model: the agent runs in Deno (``--deny-run/ffi/sys``, no net on the
    main path) and cannot create symlinks, escaping hardlinks, or special files,
    so it cannot defeat this guard. Two residuals are out of scope because they
    require an *active local attacker* racing the filesystem (not the agent):
    an intermediate path component swapped to a symlink between this scan and
    Deno's open, and the same race on the web delete path. A single permitted
    run can also exceed the quota before the next run is refused (availability,
    not an escape).
    """
    from src.config import CONFIG

    max_bytes = CONFIG.workspace_max_size_mb * 1024 * 1024
    total = 0
    # inode (st_dev, st_ino) -> [fs-wide link count, links seen inside workspace]
    multilink: dict[tuple[int, int], list[int]] = {}
    stack = [workspace]
    while stack:
        current = stack.pop()
        try:
            it = os.scandir(current)
        except FileNotFoundError:
            continue
        except PermissionError:
            # The agent has write scope and can chmod its OWN workspace dirs
            # unreadable (e.g. Deno.chmod(dir, 0)), which would otherwise
            # permanently fail the guard and brick all future runs. Since we own
            # the dir, restore owner access and retry once. If we don't own it
            # (chmod fails), fall through to fail-closed — a planted dir owned by
            # another user can't be forced open to hide a symlink/hardlink.
            try:
                os.chmod(current, os.stat(current).st_mode | 0o700)
                it = os.scandir(current)
            except OSError as exc:
                raise WorkspaceError(
                    f"cannot scan the workspace ({exc}); refusing to run code "
                    "that writes to the workspace."
                )
        except OSError as exc:
            # Fail closed: if we can't enumerate a directory we can't prove it's
            # symlink/hardlink free, so don't grant write access.
            raise WorkspaceError(
                f"cannot scan the workspace ({exc}); refusing to run code that "
                "writes to the workspace."
            )
        with it:
            for entry in it:
                if entry.is_symlink():
                    raise WorkspaceError(
                        f"workspace contains a symlink ({entry.path!r}); "
                        "symlinks can escape the sandbox's scoped write "
                        "permission and are not permitted. Remove it (e.g. via "
                        "the workspace file browser) and retry."
                    )
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    # FIFOs/sockets/device nodes are not supported workspace
                    # artifacts and could be a planted DoS/escape vector.
                    raise WorkspaceError(
                        f"workspace contains a special file ({entry.path!r}); "
                        "only regular files and directories are supported. "
                        "Remove it and retry."
                    )
                if st.st_nlink > 1:
                    rec = multilink.get((st.st_dev, st.st_ino))
                    if rec is None:
                        multilink[(st.st_dev, st.st_ino)] = [st.st_nlink, 1]
                    else:
                        rec[1] += 1
                total += st.st_size

    for fs_links, inside_links in multilink.values():
        if fs_links > inside_links:
            raise WorkspaceError(
                "workspace contains a hardlink to a file outside the workspace; "
                "writes through it would escape the sandbox's scoped write "
                "permission. Remove it (e.g. via the workspace file browser) "
                "and retry."
            )

    if max_bytes and total > max_bytes:
        raise WorkspaceError(
            f"workspace is over its size limit ({total} bytes > {max_bytes} "
            f"bytes / {CONFIG.workspace_max_size_mb} MB). Free space before "
            "running code that writes to the workspace."
        )


class ToolExecutor:
    """executor that runs Typescript code in a deno subprocess"""

    def __init__(self, registry: ToolRegistry, ctx: ToolContext) -> None:
        self._registry = registry
        self._ctx = ctx
        self._tool_definition: dict[str, Any] | None = None
        self._custom_tool_manager: Any | None = None
        self._mcp_manager: Any | None = None
        self._secret_manager: Any | None = None
        self._scheduler: Any | None = None

    # -- public read-only properties for web/Discord/main.py --

    @property
    def store(self) -> Any:
        return self._ctx._store

    @property
    def secret_manager(self) -> Any | None:
        return self._secret_manager

    @property
    def custom_tool_manager(self) -> Any | None:
        return self._custom_tool_manager

    @property
    def mcp_manager(self) -> Any | None:
        return self._mcp_manager

    @property
    def scheduler(self) -> Any | None:
        return self._scheduler

    @property
    def http_client(self) -> Any:
        return self._ctx._http_client

    @property
    def llm_client(self) -> Any:
        return self._ctx._llm_client

    @property
    def exa_client(self) -> Any:
        return self._ctx._exa_client

    @property
    def ctx(self) -> ToolContext:
        return self._ctx

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def initialize(self) -> None:
        from src.config import CONFIG

        data_dir = Path(CONFIG.data_dir)

        # initialize local SQLite store (notes, tracking, chat sessions)
        from src.store.store import Store

        store = Store(path=data_dir / "store.db")
        await store.initialize()
        self._ctx._store = store

        # initialize secrets (local file storage)
        try:
            from src.tools.secrets import SecretManager

            self._secret_manager = SecretManager(path=data_dir / "secrets.json")
            await self._secret_manager.load_secrets()
            self._ctx._secret_manager = self._secret_manager
        except Exception:
            logger.warning("Failed to initialize secrets", exc_info=True)

        # initialize scheduler (DocumentStore-backed)
        try:
            from src.scheduler.scheduler import Scheduler

            self._scheduler = Scheduler(
                store=store.documents,
                legacy_path=data_dir / "schedules.json",
            )
            await self._scheduler.load_tasks()
            self._ctx._scheduler = self._scheduler
        except Exception:
            logger.warning("Failed to initialize scheduler", exc_info=True)

        # initialize custom tool manager (local store)
        try:
            from src.tools.custom import CustomToolManager

            self._custom_tool_manager = CustomToolManager(
                store=store,
                executor=self,
                secret_manager=self._secret_manager,
            )
            await self._custom_tool_manager.load_tools()
            self._ctx._custom_tool_manager = self._custom_tool_manager
        except Exception:
            logger.warning("Failed to initialize custom tool manager", exc_info=True)

        # initialize MCP manager (connect to external MCP servers)
        try:
            from src.tools.mcp import MCPManager

            self._mcp_manager = MCPManager(
                store=store,
                secret_manager=self._secret_manager,
                registry=self._registry,
            )
            await self._mcp_manager.load_servers()
            await self._mcp_manager.connect_all()
            self._ctx._mcp_manager = self._mcp_manager
        except Exception:
            logger.warning("Failed to initialize MCP manager", exc_info=True)

        # initialize embedding client (Ollama) — non-fatal if unavailable
        embedding_client = None
        try:
            from src.memory.embeddings import EmbeddingClient

            embedding_client = EmbeddingClient(
                endpoint=CONFIG.embedding_endpoint,
                model=CONFIG.embedding_model,
                dimensions=CONFIG.embedding_dimensions,
            )
            self._ctx._embedding_client = embedding_client
        except Exception:
            logger.warning("Failed to initialize embedding client", exc_info=True)

        # wire embedding client into MemoryStore for auto-embedding on write
        if store.memory is not None and embedding_client is not None:
            store.memory.set_embedding_client(
                embedding_client, dimensions=CONFIG.embedding_dimensions
            )

        # initialize search engine (hybrid FTS5 + vector cosine)
        try:
            from src.memory.search import SearchEngine

            search_engine = SearchEngine(
                store=store.memory,
                embedding_client=embedding_client,
            )
            self._ctx._search_engine = search_engine
        except Exception:
            logger.warning("Failed to initialize search engine", exc_info=True)

    async def aclose(self) -> None:
        """Close resources held by the executor (embedding client HTTP connection)."""
        if self._mcp_manager is not None:
            try:
                await self._mcp_manager.disconnect_all()
            except Exception:
                logger.warning("Error closing MCP connections", exc_info=True)

        if self._ctx._embedding_client is not None:
            try:
                await self._ctx._embedding_client.close()
            except Exception:
                logger.warning("Error closing embedding client", exc_info=True)

    async def execute_code(self, code: str) -> dict[str, Any]:
        """
        execute Typescript code in a deno subprocess.

        code has access to tools defined in the registry via the generated typescript
        stubs. calls are bridged to python via stdin/out
        """

        if len(code) > MAX_CODE_SIZE:
            return {
                "success": False,
                "error": f"code too large ({len(code)} chars, max {MAX_CODE_SIZE})",
                "debug": [],
            }

        stub_path = self._write_stub_file()
        stub_name = os.path.basename(stub_path)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ts", delete=False, dir=DENO_DIR
        ) as f:
            workspace_json = json.dumps(_resolve_workspace())
            # start by adding all the imports that we need...
            full_code = f"""
import {{ output, debug }} from "./runtime.ts";
import * as tools from "./{stub_name}";
export {{ tools }};

const WORKSPACE = {workspace_json};

{code}
"""
            f.write(full_code)
            temp_path = f.name

        try:
            return await self._run_deno(temp_path)
        except WorkspaceError as e:
            return {"success": False, "error": str(e), "debug": []}
        finally:
            os.unlink(temp_path)
            os.unlink(stub_path)

    async def execute_custom_tool_code(
        self,
        code: str,
        params: dict[str, Any],
        env: dict[str, str] | None = None,
        allow_net: bool = False,
    ) -> dict[str, Any]:
        """Execute custom tool TypeScript code with injected params and relaxed permissions."""

        if len(code) > MAX_CODE_SIZE:
            return {
                "success": False,
                "error": f"code too large ({len(code)} chars, max {MAX_CODE_SIZE})",
                "debug": [],
            }

        stub_path = self._write_stub_file()
        stub_name = os.path.basename(stub_path)

        params_json = json.dumps(params)
        workspace_json = json.dumps(_resolve_workspace())
        full_code = f"""
import {{ output, debug }} from "./runtime.ts";
import * as tools from "./{stub_name}";
export {{ tools }};

const params = {params_json};
const WORKSPACE = {workspace_json};

// --- tool code ---
{code}
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ts", delete=False, dir=DENO_DIR
        ) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            return await self._run_deno_with_permissions(
                temp_path,
                allow_net=allow_net,
                env=env or {},
            )
        except WorkspaceError as e:
            return {"success": False, "error": str(e), "debug": []}
        finally:
            os.unlink(temp_path)
            os.unlink(stub_path)

    async def execute_condition_code(
        self,
        code: str,
        env: dict[str, str] | None = None,
        allow_net: bool = False,
    ) -> dict[str, Any]:
        """Execute trigger condition TypeScript code WITHOUT the tools bridge.

        Condition scripts only have access to ``output()`` and ``debug()`` —
        no backend tools. This prevents a condition predicate from performing
        side effects via the tool registry.
        """

        if len(code) > MAX_CODE_SIZE:
            return {
                "success": False,
                "error": f"code too large ({len(code)} chars, max {MAX_CODE_SIZE})",
                "debug": [],
            }

        full_code = f"""
import {{ output, debug }} from "./runtime.ts";

// --- condition code ---
{code}
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ts", delete=False, dir=DENO_DIR
        ) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            return await self._run_deno_with_permissions(
                temp_path,
                allow_net=allow_net,
                env=env or {},
                allow_tool_calls=False,
                allow_workspace=False,
            )
        finally:
            os.unlink(temp_path)

    def _write_stub_file(self) -> str:
        """Write generated tool stubs to a unique file in DENO_DIR.

        Each execution gets its own stub file so concurrent executions can't
        observe a half-written file or a stub set written by another run. The
        executor is shared across web, Discord, Signal, and the scheduler, so
        executions overlap. Returns the stub path; the caller must unlink it.
        """
        tools_ts = self._registry.generate_typescript_types()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ts", prefix="tools_", delete=False, dir=DENO_DIR
        ) as f:
            f.write(tools_ts)
            return f.name

    @staticmethod
    def _kill_process(process: asyncio.subprocess.Process) -> None:
        """kill a subprocess, ignoring errors if it's already dead"""
        try:
            process.kill()
        except ProcessLookupError:
            pass

    @staticmethod
    def _minimal_env(extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build a minimal environment for Deno subprocesses.

        Only PATH (for binary resolution) and HOME (for Deno's cache dir)
        are inherited from the parent. Arbitrary parent env vars — including
        API keys and secrets — are never passed through unless explicitly
        granted via *extra*.
        """
        env: dict[str, str] = {
            k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ
        }
        if extra:
            env.update(extra)
        return env

    async def _run_deno(self, script_path: str) -> dict[str, Any]:
        """run the input script in a deno subprocess"""

        # spawn a subprocess that executes deno with minimal permissions. explicit deny flags
        # ensure these can't be escalated via dynamic imports or permission prompts.
        deno_read_path = str(DENO_DIR)
        if "," in deno_read_path:
            raise ValueError(
                f"DENO_DIR must not contain commas (Deno permission separator): {deno_read_path!r}"
            )
        workspace = _resolve_workspace()
        # Refuse to run if the workspace contains a symlink (write-scope escape)
        # or is over quota, before granting Deno --allow-write to it.
        _scan_workspace_or_raise(workspace)
        read_paths = [deno_read_path, workspace]

        process = await asyncio.create_subprocess_exec(
            "deno",
            "run",
            f"--allow-read={','.join(read_paths)}",
            f"--allow-write={workspace}",
            "--deny-net",
            "--deny-run",
            "--deny-env",
            "--deny-ffi",
            "--deny-sys",
            "--no-prompt",
            "--no-remote",
            "--no-npm",
            f"--v8-flags=--max-old-space-size={DENO_MEMORY_LIMIT_MB}",
            script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._minimal_env(),
        )

        return await self._process_deno_output(process)

    async def _run_deno_with_permissions(
        self,
        script_path: str,
        allow_net: bool = False,
        env: dict[str, str] | None = None,
        allow_tool_calls: bool = True,
        allow_workspace: bool = True,
    ) -> dict[str, Any]:
        """Run deno with configurable permissions for custom tools.

        When ``allow_workspace`` is True (default, custom tools path), scoped
        workspace read/write access is granted. When False (condition code
        path), writes remain fully denied — condition scripts are side-effect
        free by design.
        """

        deno_read_path = str(DENO_DIR)
        if "," in deno_read_path:
            raise ValueError(
                f"DENO_DIR must not contain commas (Deno permission separator): {deno_read_path!r}"
            )

        if allow_workspace:
            workspace = _resolve_workspace()
            # Same pre-execution guard as _run_deno before granting write scope.
            _scan_workspace_or_raise(workspace)
            args = [
                "deno",
                "run",
                f"--allow-read={deno_read_path},{workspace}",
                f"--allow-write={workspace}",
                "--deny-run",
                "--deny-ffi",
                "--deny-sys",
                "--no-prompt",
            ]
        else:
            args = [
                "deno",
                "run",
                f"--allow-read={deno_read_path}",
                "--deny-write",
                "--deny-run",
                "--deny-ffi",
                "--deny-sys",
                "--no-prompt",
            ]

        if allow_net:
            # Allow outbound fetch() but NOT remote module loading —
            # approved code is reviewed as-is, and remote imports could
            # change behavior after approval.
            args.append("--allow-net")
            args.append("--no-remote")
            args.append("--no-npm")
        else:
            args.append("--deny-net")
            args.append("--no-remote")
            args.append("--no-npm")

        # allow specific env vars if provided
        if env:
            env_list = ",".join(env.keys())
            args.append(f"--allow-env={env_list}")
        else:
            args.append("--deny-env")

        args.append(f"--v8-flags=--max-old-space-size={DENO_MEMORY_LIMIT_MB}")
        args.append(script_path)

        # Don't inherit the parent env wholesale — it can contain MODEL_API_KEY
        # and other secrets the tool was never granted.
        proc_env = self._minimal_env(env)

        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )

        return await self._process_deno_output(process, allow_tool_calls=allow_tool_calls)

    async def _process_deno_output(
        self, process: asyncio.subprocess.Process, allow_tool_calls: bool = True
    ) -> dict[str, Any]:
        """Shared logic for processing deno subprocess output.

        When ``allow_tool_calls`` is False, any ``__tool_call__`` message on
        stdout is treated as an error — the process is killed and the result
        is marked as failed. This prevents code that shouldn't have tool
        access (e.g. trigger condition scripts) from forging tool-call
        protocol messages to invoke backend tools.
        """

        if process.stdin is None:
            raise RuntimeError("Process stdin is not available")
        if process.stdout is None:
            raise RuntimeError("Process stdout is not available")
        if process.stderr is None:
            raise RuntimeError("Process stderr is not available")

        # Maximum stderr bytes to capture (prevents memory exhaustion)
        MAX_STDERR_SIZE = 64 * 1024

        # Drain stderr concurrently to avoid a pipe-buffer deadlock: if the
        # process fills the OS stderr pipe buffer (~64KB) while nobody is
        # reading it, the process blocks on stderr write, never produces
        # stdout, and the readline loop below deadlocks until the read
        # timeout.
        async def _drain_stderr() -> bytes:
            chunks: list[bytes] = []
            total = 0
            while True:
                try:
                    chunk = await process.stderr.read(MAX_STDERR_SIZE)
                except Exception:
                    break
                if not chunk:
                    break
                total += len(chunk)
                if total <= MAX_STDERR_SIZE:
                    chunks.append(chunk)
                else:
                    excess = total - MAX_STDERR_SIZE
                    keep = len(chunk) - excess
                    if keep > 0:
                        chunks.append(chunk[:keep])
                    chunks.append(b"\n... (stderr truncated)")
                    break
            return b"".join(chunks)

        stderr_task = asyncio.create_task(_drain_stderr())

        outputs: list[Any] = []
        debug_messages: list[str] = []
        error: str | None = None
        tool_call_count = 0
        total_output_bytes = 0
        deadline = asyncio.get_running_loop().time() + MAX_EXECUTION_TIME

        try:
            while True:
                # calculate remaining time against the total execution deadline
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    self._kill_process(process)
                    error = f"execution timed out after {MAX_EXECUTION_TIME:.0f} seconds (total)"
                    break

                # read next line with the lesser of 30s or remaining wall-clock time
                read_timeout = min(30.0, remaining)
                line = await asyncio.wait_for(
                    process.stdout.readline(), timeout=read_timeout
                )

                # if there are no more lines we're finished...
                if not line:
                    # Empty stdout before any tool call. If the process
                    # exited with an error, treat it as a crash.
                    # Otherwise it's a clean exit (empty output).
                    if process.returncode is not None and process.returncode != 0 and tool_call_count == 0:
                        error = f"deno process exited early with code {process.returncode}"
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue

                # track total output size to prevent stdout flooding
                total_output_bytes += len(line)
                if total_output_bytes > MAX_OUTPUT_SIZE:
                    self._kill_process(process)
                    error = f"output exceeded {MAX_OUTPUT_SIZE} bytes, killed"
                    break

                try:
                    message = json.loads(line_str)
                except json.JSONDecodeError:
                    debug_messages.append(line_str)
                    continue

                # whenever we encounter a tool call, we then need to execute that tool and give
                # it the response
                if "__tool_call__" in message:
                    if not allow_tool_calls:
                        self._kill_process(process)
                        error = "tool calls are not allowed in this execution mode"
                        break

                    tool_call_count += 1
                    if tool_call_count > MAX_TOOL_CALLS:
                        self._kill_process(process)
                        error = f"exceeded maximum of {MAX_TOOL_CALLS} tool calls"
                        break

                    tool_name = message["tool"]
                    params = message["params"]
                    logger.info(f"Tool call: {tool_name} with params: {params}")

                    try:
                        result = await self._registry.execute(
                            self._ctx, tool_name, params
                        )
                        response = json.dumps({"__tool_result__": result}, default=str)
                    except Exception as e:
                        logger.exception(f"Tool error: {tool_name}")
                        response = json.dumps({"__tool_error__": str(e)})

                    try:
                        process.stdin.write((response + "\n").encode())
                        await process.stdin.drain()
                    except (ConnectionResetError, BrokenPipeError):
                        error = f"deno process exited while sending tool result for {tool_name}"
                        break

                elif "__output__" in message:
                    outputs.append(message["__output__"])

                elif "__debug__" in message:
                    debug_messages.append(message["__debug__"])

                else:
                    debug_messages.append(line_str)

        # make sure that we kill deno subprocess if the execution times out
        except asyncio.TimeoutError:
            self._kill_process(process)
            error = "execution timed out"
        # also kill it for any other exceptions we encounter
        except Exception as e:
            self._kill_process(process)
            error = str(e)

        await process.wait()

        # Await the concurrent stderr drain (the process has exited, so the
        # stderr pipe will hit EOF quickly).
        try:
            stderr_content = await asyncio.wait_for(stderr_task, timeout=5.0)
        except Exception:
            stderr_task.cancel()
            try:
                await stderr_task
            except Exception:
                pass
            stderr_content = b""

        if stderr_content:
            stderr_str = stderr_content.decode().strip()
            if stderr_str:
                if error:
                    error += f"\n\nStderr:\n{stderr_str}"
                else:
                    error = stderr_str

        # After wait(), returncode is guaranteed to be an int.
        # Treat any non-zero exit as an error, even if error is already set.
        if process.returncode != 0 and error is None:
            error = f"deno process exited with code {process.returncode}"

        success = process.returncode == 0 and error is None

        result: dict[str, Any] = {
            "success": success,
            "debug": debug_messages,
        }

        if outputs:
            result["output"] = outputs[-1] if len(outputs) == 1 else outputs

        if error:
            result["error"] = error

        return result

    def get_execute_code_tool_definition(self) -> dict[str, Any]:
        """get tool definition for execute_code, including all the docs for available backend tools"""

        if self._tool_definition is not None:
            return self._tool_definition

        description = """Run TypeScript in a sandboxed Deno runtime. This is the ONLY way to invoke any tool — every operation (managing memory, searching the web, sending Discord alerts, creating schedules, etc.) must go through TypeScript code you submit to this function.

The code has access to a global `tools` namespace (e.g. `tools.memory.get(...)`, `tools.web.web_search(...)`). See the full catalog in the system prompt. Use `output(value)` to return results and `debug(...args)` to log.

Always wrap tool calls in `await` and batch independent calls with `Promise.allSettled([...])` inside a single code submission.

Example:
```typescript
const result = await tools.memory.get({ scope: "operator" });
const search = await tools.memory.search({ query: "deploy steps" });
output({ result, search });
```"""

        self._tool_definition = {
            "name": "execute_code",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Typescript code to execute. Has access to `tools` namespace and `output()` function.",
                    }
                },
                "required": ["code"],
            },
        }
        return self._tool_definition
