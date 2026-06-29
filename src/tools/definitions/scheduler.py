import json
import logging

from src.config import resolve_operator_tz
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)

MAX_CODE_SIZE = 50_000


@TOOL_REGISTRY.tool(
    name="scheduler.list_schedules",
    description="List all scheduled tasks with their status and next run time.",
    parameters=[],
)
async def list_schedules(ctx: ToolContext) -> str:
    sched = ctx.scheduler
    if sched is None:
        return "Error: Scheduler not available."

    from datetime import datetime, timezone

    tasks = sched.list_tasks()
    if not tasks:
        return "No scheduled tasks found."

    now = datetime.now(timezone.utc)
    lines = [f"Found {len(tasks)} schedule(s):\n"]
    for task in tasks:
        status = "enabled" if task.enabled else "disabled"
        try:
            desc = str(task.config)
            if task.last_run:
                last = datetime.fromisoformat(task.last_run)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                next_run = task.config.next_run(last)
                next_str = next_run.astimezone(resolve_operator_tz()).strftime(
                    "%Y-%m-%d %H:%M %Z"
                )
            else:
                next_str = "pending"
        except Exception:
            desc = task.schedule
            next_str = "unknown"

        trigger_badge = ""
        if task.condition_code is not None:
            trigger_badge = " [trigger"
            if task.approved:
                trigger_badge += ":approved]"
            else:
                trigger_badge += ":pending]"

        lines.append(f"- **{task.name}** [{status}]{trigger_badge} ({desc})")
        lines.append(f"  prompt: {task.prompt[:80]}")
        if task.condition_code is not None:
            preview = task.condition_code[:80]
            if len(task.condition_code) > 80:
                preview += "…"
            lines.append(f"  condition: {preview}")
        if task.enabled:
            lines.append(f"  next run: {next_str}")

    return "\n".join(lines)


@TOOL_REGISTRY.tool(
    name="scheduler.get_schedule",
    description="Get full details of a scheduled task.",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name of the schedule to retrieve.",
        ),
    ],
)
async def get_schedule(ctx: ToolContext, name: str) -> str:
    sched = ctx.scheduler
    if sched is None:
        return "Error: Scheduler not available."

    task = sched.get_task(name)
    if task is None:
        return f"Schedule '{name}' not found."

    d = task.to_dict()
    # Truncate condition_code to avoid bloating the agent's context
    if d.get("condition_code"):
        d["condition_code"] = d["condition_code"][:200]
        if len(task.condition_code) > 200:
            d["condition_code"] += "…"
    return json.dumps(d, indent=2)


SCHEDULE_SYNTAX_HELP = """Supported schedule expressions:
- Duration: `30m`, `2h`, `1h30m`, `90s` — fires every interval (min 1 minute)
- Keywords: `hourly`, `daily`, `weekly`
- Daily at time: `daily@9:00`, `daily@14:30` — 24h clock, in the operator's local timezone
- Weekly on day: `weekly@monday`, `weekly@fri`
- Weekly on day at time: `weekly@monday@9:00`
- Cron: `cron:0 9 * * *` (if croniter installed)

Absolute times (daily@, weekly@…@, cron) are interpreted in the operator's local
timezone — just use the wall-clock time they asked for. Durations are relative."""


@TOOL_REGISTRY.tool(
    name="scheduler.create_schedule",
    description=f"""Create a new scheduled task that will fire a prompt on a recurring schedule. When the schedule fires, the agent runs the prompt in a fresh conversation (no prior history) with full access to tools and notes, and the run is saved as its own chat session.

Use this to give yourself recurring work — daily summaries, periodic checks, reminders. The prompt should be self-contained since scheduled runs start with no conversation context.

Optionally, you can attach a **condition script** (TypeScript) that runs on schedule *before* waking you. The script calls `output({{ wake: true }})` to wake you, or `output({{ wake: false }})` to skip this cycle. This avoids wasting a turn when there's nothing to act on. Condition code requires operator approval before it will fire.

{SCHEDULE_SYNTAX_HELP}""",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Unique name for this schedule (alphanumeric, dashes, underscores).",
        ),
        ToolParameter(
            name="schedule",
            type="string",
            description="Schedule expression. See description for supported formats.",
        ),
        ToolParameter(
            name="prompt",
            type="string",
            description="The prompt to send yourself when the schedule fires. Must be self-contained (scheduled runs start with no conversation history).",
        ),
        ToolParameter(
            name="condition_code",
            type="string",
            required=False,
            description="Optional TypeScript condition script. If provided, the script runs on schedule before waking you. Call output({ wake: true }) to wake, output({ wake: false }) to skip. Requires operator approval before it will fire. Max 50,000 chars.",
        ),
        ToolParameter(
            name="requires_net",
            type="boolean",
            required=False,
            description="Set true if the condition code needs network access (fetch, HTTP). Defaults to false.",
        ),
        ToolParameter(
            name="secrets",
            type="array",
            required=False,
            description="Optional list of secret names to inject as env vars into the condition script.",
        ),
    ],
)
async def create_schedule(
    ctx: ToolContext,
    name: str,
    schedule: str,
    prompt: str,
    condition_code: str | None = None,
    requires_net: bool = False,
    secrets: list[str] | None = None,
) -> str:
    sched = ctx.scheduler
    if sched is None:
        return "Error: Scheduler not available."

    if not name or not schedule or not prompt:
        return "Error: name, schedule, and prompt are all required."

    if sched.get_task(name) is not None:
        return f"Error: schedule '{name}' already exists. Use update_schedule to modify it, or delete_schedule first."

    if condition_code is not None:
        if not condition_code.strip():
            return "Error: condition_code must not be empty."
        if len(condition_code) > MAX_CODE_SIZE:
            return f"Error: condition_code too large ({len(condition_code)} chars, max {MAX_CODE_SIZE})."

    # Coerce/validate untyped tool-bridge params
    requires_net = bool(requires_net)
    if secrets is not None:
        if not isinstance(secrets, list) or not all(
            isinstance(s, str) for s in secrets
        ):
            return "Error: secrets must be a list of strings."
        secrets = list(dict.fromkeys(secrets))  # dedupe, preserve order
    else:
        secrets = []

    from src.scheduler.scheduler import ScheduledTask

    try:
        task = ScheduledTask(
            name=name,
            schedule=schedule,
            prompt=prompt,
            condition_code=condition_code,
            requires_net=requires_net,
            secrets=secrets or [],
            approved=False,
        )
        return await sched.create_task(task)
    except ValueError as e:
        return f"Error: invalid schedule expression — {e}"
    except Exception as e:
        logger.exception("Failed to create schedule")
        return f"Error: {e}"


@TOOL_REGISTRY.tool(
    name="scheduler.update_schedule",
    description=f"""Update an existing scheduled task. Any unspecified field is left unchanged.

If you change the condition_code or broaden requires_net (false→true), approval is reset — the operator must re-approve before the task will fire.

{SCHEDULE_SYNTAX_HELP}""",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name of the schedule to update.",
        ),
        ToolParameter(
            name="schedule",
            type="string",
            required=False,
            description="New schedule expression.",
        ),
        ToolParameter(
            name="prompt",
            type="string",
            required=False,
            description="New prompt to fire on this schedule.",
        ),
        ToolParameter(
            name="enabled",
            type="boolean",
            required=False,
            description="Enable or disable the schedule.",
        ),
        ToolParameter(
            name="condition_code",
            type="string",
            required=False,
            description="New TypeScript condition script. Changing this resets operator approval.",
        ),
        ToolParameter(
            name="requires_net",
            type="boolean",
            required=False,
            description="Whether the condition code needs network access.",
        ),
        ToolParameter(
            name="secrets",
            type="array",
            required=False,
            description="List of secret names to inject as env vars into the condition script.",
        ),
    ],
)
async def update_schedule(
    ctx: ToolContext,
    name: str,
    schedule: str | None = None,
    prompt: str | None = None,
    enabled: bool | None = None,
    condition_code: str | None = None,
    requires_net: bool | None = None,
    secrets: list[str] | None = None,
) -> str:
    sched = ctx.scheduler
    if sched is None:
        return "Error: Scheduler not available."

    if sched.get_task(name) is None:
        return f"Schedule '{name}' not found."

    if condition_code is not None:
        if not condition_code.strip():
            return "Error: condition_code must not be empty."
        if len(condition_code) > MAX_CODE_SIZE:
            return f"Error: condition_code too large ({len(condition_code)} chars, max {MAX_CODE_SIZE})."

    # Coerce/validate untyped tool-bridge params
    if requires_net is not None:
        requires_net = bool(requires_net)
    if secrets is not None:
        if not isinstance(secrets, list) or not all(
            isinstance(s, str) for s in secrets
        ):
            return "Error: secrets must be a list of strings."
        secrets = list(dict.fromkeys(secrets))  # dedupe, preserve order

    try:
        return await sched.update_task(
            name,
            schedule=schedule,
            prompt=prompt,
            enabled=enabled,
            condition_code=condition_code,
            requires_net=requires_net,
            secrets=secrets,
        )
    except ValueError as e:
        return f"Error: invalid schedule expression — {e}"
    except Exception as e:
        logger.exception("Failed to update schedule")
        return f"Error: {e}"


@TOOL_REGISTRY.tool(
    name="scheduler.delete_schedule",
    description="Delete a scheduled task.",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name of the schedule to delete.",
        ),
    ],
)
async def delete_schedule(ctx: ToolContext, name: str) -> str:
    sched = ctx.scheduler
    if sched is None:
        return "Error: Scheduler not available."
    return await sched.delete_task(name)


@TOOL_REGISTRY.tool(
    name="scheduler.enable_schedule",
    description="Enable a scheduled task so it will fire on its schedule.",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name of the schedule to enable.",
        ),
    ],
)
async def enable_schedule(ctx: ToolContext, name: str) -> str:
    sched = ctx.scheduler
    if sched is None:
        return "Error: Scheduler not available."
    return await sched.enable_task(name)


@TOOL_REGISTRY.tool(
    name="scheduler.disable_schedule",
    description="Disable a scheduled task so it will not fire until re-enabled.",
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Name of the schedule to disable.",
        ),
    ],
)
async def disable_schedule(ctx: ToolContext, name: str) -> str:
    sched = ctx.scheduler
    if sched is None:
        return "Error: Scheduler not available."
    return await sched.disable_task(name)
