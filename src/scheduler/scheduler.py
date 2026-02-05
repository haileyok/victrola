import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.scheduler.schedule import ScheduleConfig, parse_schedule

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 30


@dataclass
class ScheduledTask:
    name: str
    schedule: str  # schedule expression
    prompt: str  # what to tell the agent
    enabled: bool = True
    last_run: str | None = None  # ISO 8601

    # computed at runtime, not persisted
    _config: ScheduleConfig | None = field(default=None, repr=False, compare=False)

    @property
    def config(self) -> ScheduleConfig:
        if self._config is None:
            self._config = parse_schedule(self.schedule)
        return self._config

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "schedule": self.schedule,
            "prompt": self.prompt,
            "enabled": self.enabled,
        }
        if self.last_run:
            d["last_run"] = self.last_run
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledTask":
        return cls(
            name=data["name"],
            schedule=data["schedule"],
            prompt=data.get("prompt", ""),
            enabled=data.get("enabled", True),
            last_run=data.get("last_run"),
        )


class Scheduler:
    """Background scheduler that fires agent prompts on a schedule.

    Schedules are stored locally as a JSON file.
    When a task fires, it calls the `on_fire` callback with (task_name, prompt).
    """

    def __init__(
        self,
        path: Path,
        on_fire: Callable[[str, str], Awaitable[str]] | None = None,
    ) -> None:
        self._path = path
        self._on_fire = on_fire
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False

    def _save(self) -> None:
        """Persist all tasks to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [task.to_dict() for task in self._tasks.values()]
        self._path.write_text(json.dumps(data, indent=2) + "\n")

    async def load_tasks(self) -> None:
        """Load all scheduled tasks from the local JSON file."""
        self._tasks.clear()
        if not self._path.exists():
            return

        try:
            data = json.loads(self._path.read_text())
            if isinstance(data, list):
                for entry in data:
                    try:
                        task = ScheduledTask.from_dict(entry)
                        _ = task.config  # validate the schedule expression
                        self._tasks[task.name] = task
                    except (KeyError, ValueError) as e:
                        logger.warning("Failed to parse schedule entry: %s", e)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load schedules from %s: %s", self._path, e)

        logger.info("Loaded %d schedule(s)", len(self._tasks))

    def list_tasks(self) -> list[ScheduledTask]:
        return list(self._tasks.values())

    def get_task(self, name: str) -> ScheduledTask | None:
        return self._tasks.get(name)

    async def create_task(self, task: ScheduledTask) -> str:
        """Create a new scheduled task."""
        # validate the schedule expression
        _ = task.config

        # set last_run to now so the first fire is after one interval/at next time
        if not task.last_run:
            task.last_run = datetime.now(timezone.utc).isoformat()

        self._tasks[task.name] = task
        self._save()

        next_run = task.config.next_run(datetime.now(timezone.utc))
        return f"Schedule '{task.name}' created ({task.config}). Next run: {next_run.strftime('%Y-%m-%d %H:%M UTC')}."

    async def update_task(self, name: str, **fields: Any) -> str:
        """Update an existing scheduled task."""
        task = self._tasks.get(name)
        if task is None:
            return f"Schedule '{name}' not found."

        for key, value in fields.items():
            if value is not None:
                setattr(task, key, value)

        # re-validate and clear cache if schedule changed
        if "schedule" in fields:
            task._config = None
            _ = task.config

        self._save()
        return f"Schedule '{name}' updated."

    async def delete_task(self, name: str) -> str:
        """Delete a scheduled task."""
        if name not in self._tasks:
            return f"Schedule '{name}' not found."

        self._tasks.pop(name, None)
        self._save()
        return f"Schedule '{name}' deleted."

    async def enable_task(self, name: str) -> str:
        return await self.update_task(name, enabled=True)

    async def disable_task(self, name: str) -> str:
        return await self.update_task(name, enabled=False)

    async def run(self) -> None:
        """Main scheduler loop. Run as a background asyncio task."""
        self._running = True
        logger.info("Scheduler started with %d task(s)", len(self._tasks))

        while self._running:
            await self._tick()
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._running = False

    async def _tick(self) -> None:
        """Check all tasks and fire any that are due."""
        now = datetime.now(timezone.utc)

        for task in list(self._tasks.values()):
            if not task.enabled:
                continue

            try:
                if self._is_due(task, now):
                    await self._fire(task, now)
            except Exception:
                logger.exception("Error checking/firing schedule '%s'", task.name)

    def _is_due(self, task: ScheduledTask, now: datetime) -> bool:
        """Check if a task is due to run."""
        if task.last_run:
            last = datetime.fromisoformat(task.last_run)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
        else:
            # never run, no last_run — initialize it to now and skip this tick
            task.last_run = now.isoformat()
            return False

        next_run = task.config.next_run(last)
        return now >= next_run

    async def _fire(self, task: ScheduledTask, now: datetime) -> None:
        """Execute a scheduled task."""
        logger.info("Firing schedule '%s': %s", task.name, task.prompt[:100])

        # update last_run immediately to prevent double-firing
        task.last_run = now.isoformat()
        self._save()

        if self._on_fire:
            try:
                response = await self._on_fire(task.name, task.prompt)
                logger.info(
                    "Schedule '%s' completed: %s",
                    task.name,
                    (response[:200] if response else "(empty)"),
                )
            except Exception:
                logger.exception("Schedule '%s' callback failed", task.name)
        else:
            logger.info("Schedule '%s' fired (no callback wired)", task.name)
