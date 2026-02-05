import asyncio
import json
import logging
import os
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


class ToolExecutor:
    """executor that runs Typescript code in a deno subprocess"""

    def __init__(self, registry: ToolRegistry, ctx: ToolContext) -> None:
        self._registry = registry
        self._ctx = ctx
        self._tool_definition: dict[str, Any] | None = None
        self._custom_tool_manager: Any | None = None
        self._secret_manager: Any | None = None
        self._scheduler: Any | None = None

    async def initialize(self) -> None:
        from src.config import CONFIG

        data_dir = Path(CONFIG.data_dir)

        # initialize local SQLite store (notes, tracking, chat sessions)
        from src.store.store import Store

        store = Store(path=data_dir / "store.db")
        await store.initialize()
        self._ctx._store = store

        # initialize secrets and scheduler (local file storage)
        try:
            from src.tools.secrets import SecretManager

            self._secret_manager = SecretManager(path=data_dir / "secrets.json")
            await self._secret_manager.load_secrets()
            self._ctx._secret_manager = self._secret_manager

            from src.scheduler.scheduler import Scheduler

            self._scheduler = Scheduler(path=data_dir / "schedules.json")
            await self._scheduler.load_tasks()
            self._ctx._scheduler = self._scheduler
        except Exception:
            logger.warning("Failed to initialize secrets/scheduler", exc_info=True)

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

        self._write_generated_tools()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ts", delete=False, dir=DENO_DIR
        ) as f:
            # start by adding all the imports that we need...
            full_code = f"""
import {{ output, debug }} from "./runtime.ts";
import * as tools from "./tools.ts";
export {{ tools }};

{code}
"""
            f.write(full_code)
            temp_path = f.name

        try:
            return await self._run_deno(temp_path)
        finally:
            os.unlink(temp_path)

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

        self._write_generated_tools()

        params_json = json.dumps(params)
        full_code = f"""
import {{ output, debug }} from "./runtime.ts";
import * as tools from "./tools.ts";
export {{ tools }};

const params = {params_json};

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
        finally:
            os.unlink(temp_path)

    def _write_generated_tools(self) -> None:
        """generate tool stubs and write them to the deno directory"""

        tools_ts = self._registry.generate_typescript_types()
        tools_path = DENO_DIR / "tools.ts"
        tools_path.write_text(tools_ts)

    @staticmethod
    def _kill_process(process: asyncio.subprocess.Process) -> None:
        """kill a subprocess, ignoring errors if it's already dead"""
        try:
            process.kill()
        except ProcessLookupError:
            pass

    async def _run_deno(self, script_path: str) -> dict[str, Any]:
        """run the input script in a deno subprocess"""

        # spawn a subprocess that executes deno with minimal permissions. explicit deny flags
        # ensure these can't be escalated via dynamic imports or permission prompts.
        deno_read_path = str(DENO_DIR)
        process = await asyncio.create_subprocess_exec(
            "deno",
            "run",
            f"--allow-read={deno_read_path}",
            "--deny-write",
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
        )

        return await self._process_deno_output(process)

    async def _run_deno_with_permissions(
        self,
        script_path: str,
        allow_net: bool = False,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run deno with configurable permissions for custom tools."""

        deno_read_path = str(DENO_DIR)

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
            args.append("--allow-net")
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
        # and other secrets the tool was never granted. Keep only the minimum
        # Deno needs (PATH for binary resolution, HOME for its cache dir) and
        # then layer on the explicitly-declared secrets.
        proc_env: dict[str, str] = {
            k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ
        }
        if env:
            proc_env.update(env)

        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )

        return await self._process_deno_output(process)

    async def _process_deno_output(
        self, process: asyncio.subprocess.Process
    ) -> dict[str, Any]:
        """Shared logic for processing deno subprocess output."""

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        outputs: list[Any] = []
        debug_messages: list[str] = []
        error: str | None = None
        tool_call_count = 0
        total_output_bytes = 0
        deadline = asyncio.get_event_loop().time() + MAX_EXECUTION_TIME

        try:
            while True:
                # calculate remaining time against the total execution deadline
                remaining = deadline - asyncio.get_event_loop().time()
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

        stderr_content = await process.stderr.read()
        if stderr_content:
            stderr_str = stderr_content.decode().strip()
            if stderr_str:
                if error:
                    error += f"\n\nStderr:\n{stderr_str}"
                else:
                    error = stderr_str

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

        description = """Run TypeScript in a sandboxed Deno runtime. This is the ONLY way to invoke any tool — every operation (reading notes, searching the web, sending Discord alerts, creating schedules, etc.) must go through TypeScript code you submit to this function.

The code has access to a global `tools` namespace (e.g. `tools.notes.note_get(...)`, `tools.web.web_search(...)`). See the full catalog in the system prompt. Use `output(value)` to return results and `debug(...args)` to log.

Always wrap tool calls in `await` and batch independent calls with `Promise.allSettled([...])` inside a single code submission.

Example:
```typescript
const note = await tools.notes.note_get({ rkeys: ["operator"] });
const results = await tools.web.web_search({ query: "victrola agent", num_results: 3 });
output({ note, results });
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
