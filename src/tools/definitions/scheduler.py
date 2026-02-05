import json
import logging

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

logger = logging.getLogger(__name__)


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
                next_str = next_run.strftime("%Y-%m-%d %H:%M UTC")
            else:
                next_str = "pending"
        except Exception:
            desc = task.schedule
            next_str = "unknown"

        lines.append(f"- **{task.name}** [{status}] ({desc})")
        lines.append(f"  prompt: {task.prompt[:80]}")
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

    return json.dumps(task.to_dict(), indent=2)


SCHEDULE_SYNTAX_HELP = """Supported schedule expressions:
- Duration: `30m`, `2h`, `1h30m`, `90s` — fires every interval (min 1 minute)
- Keywords: `hourly`, `daily`, `weekly`
- Daily at time: `daily@9:00`, `daily@14:30` — 24h clock, UTC
- Weekly on day: `weekly@monday`, `weekly@fri`
- Weekly on day at time: `weekly@monday@9:00`
- Cron: `cron:0 9 * * *` (if croniter installed)"""


@TOOL_REGISTRY.tool(
    name="scheduler.create_schedule",
    description=f"""Create a new scheduled task that will fire a prompt on a recurring schedule. When the schedule fires, the agent runs the prompt in a fresh conversation (no prior history) with full access to tools and notes, and the run is saved as its own chat session.

Use this to give yourself recurring work — daily summaries, periodic checks, reminders. The prompt should be self-contained since scheduled runs start with no conversation context.

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
    ],
)
async def create_schedule(
    ctx: ToolContext, name: str, schedule: str, prompt: str
) -> str:
    sched = ctx.scheduler
    if sched is None:
        return "Error: Scheduler not available."

    if not name or not schedule or not prompt:
        return "Error: name, schedule, and prompt are all required."

    if sched.get_task(name) is not None:
        return f"Error: schedule '{name}' already exists. Use update_schedule to modify it, or delete_schedule first."

    from src.scheduler.scheduler import ScheduledTask

    try:
        task = ScheduledTask(name=name, schedule=schedule, prompt=prompt)
        return await sched.create_task(task)
    except ValueError as e:
        return f"Error: invalid schedule expression — {e}"
    except Exception as e:
        logger.exception("Failed to create schedule")
        return f"Error: {e}"


@TOOL_REGISTRY.tool(
    name="scheduler.update_schedule",
    description=f"""Update an existing scheduled task. Any unspecified field is left unchanged.

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
            description="New schedule expression.",
            required=False,
        ),
        ToolParameter(
            name="prompt",
            type="string",
            description="New prompt to fire on this schedule.",
            required=False,
        ),
        ToolParameter(
            name="enabled",
            type="boolean",
            description="Enable or disable the schedule.",
            required=False,
        ),
    ],
)
async def update_schedule(
    ctx: ToolContext,
    name: str,
    schedule: str | None = None,
    prompt: str | None = None,
    enabled: bool | None = None,
) -> str:
    sched = ctx.scheduler
    if sched is None:
        return "Error: Scheduler not available."

    if sched.get_task(name) is None:
        return f"Schedule '{name}' not found."

    try:
        return await sched.update_task(
            name, schedule=schedule, prompt=prompt, enabled=enabled
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
