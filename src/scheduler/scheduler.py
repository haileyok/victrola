import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.scheduler.schedule import ScheduleConfig, parse_schedule

if TYPE_CHECKING:
    from src.store.store import DocumentStore

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 30

SCHEDULE_RKEY_PREFIX = "schedule:"

# Valid schedule name pattern: alphanumeric with -_ (no path separators, no colons)
_SCHEDULE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9\-_.]{1,128}$")

# Condition code runner: (code, requires_net, secrets) -> execution result dict
ConditionRunner = Callable[[str, bool, list[str]], Awaitable[dict[str, Any]]]


@dataclass
class ScheduledTask:
    name: str
    schedule: str  # schedule expression
    prompt: str  # what to tell the agent
    enabled: bool = True
    last_run: str | None = None  # ISO 8601
    condition_code: str | None = None  # TypeScript; absent = always wake
    requires_net: bool = False  # does condition code need network
    secrets: list[str] = field(default_factory=list)  # secret names for condition env
    approved: bool = False  # approval gate (only relevant if condition_code present)

    # computed at runtime, not persisted
    _config: ScheduleConfig | None = field(default=None, repr=False, compare=False)
    _retry_count: int = field(default=0, repr=False, compare=False)
    _condition_retry_count: int = field(default=0, repr=False, compare=False)

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
        if self.condition_code is not None:
            d["condition_code"] = self.condition_code
            d["requires_net"] = self.requires_net
            d["secrets"] = self.secrets
            d["approved"] = self.approved
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledTask":
        return cls(
            name=data["name"],
            schedule=data["schedule"],
            prompt=data.get("prompt", ""),
            enabled=data.get("enabled", True),
            last_run=data.get("last_run"),
            condition_code=data.get("condition_code"),
            requires_net=data.get("requires_net", False),
            secrets=data.get("secrets", []),
            approved=data.get("approved", False),
        )


class Scheduler:
    """Background scheduler that fires agent prompts on a schedule.

    Schedules are stored in the DocumentStore (SQLite, rkey prefix ``schedule:``).
    When a task fires, it calls the ``on_fire`` callback with (task_name, prompt).

    Tasks with ``condition_code`` run the condition script first — the agent is
    only woken when the script returns ``{ wake: true }``.
    """

    def __init__(
        self,
        store: "DocumentStore",
        on_fire: Callable[[str, str], Awaitable[str]] | None = None,
        condition_runner: ConditionRunner | None = None,
        legacy_path: Path | None = None,
    ) -> None:
        self._store = store
        self._on_fire = on_fire
        self._condition_runner = condition_runner
        self._legacy_path = legacy_path
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False

    async def _save_task(self, task: ScheduledTask) -> None:
        """Upsert a single task to the document store."""
        rkey = f"{SCHEDULE_RKEY_PREFIX}{task.name}"
        content = json.dumps(task.to_dict())
        try:
            await self._store.create(rkey, content)
        except Exception as e:
            from src.store.store import StoreConflict

            if isinstance(e, StoreConflict):
                await self._store.update(rkey, content)
            else:
                raise

    async def _delete_task(self, name: str) -> None:
        """Remove a task document from the store."""
        rkey = f"{SCHEDULE_RKEY_PREFIX}{name}"
        try:
            await self._store.delete(rkey)
        except Exception:
            logger.exception("Failed to delete schedule doc %s", rkey)

    async def _migrate_legacy(self) -> None:
        """One-time migration from schedules.json to DocumentStore.

        Idempotent: if the JSON file was already renamed (.migrated), this
        is a no-op. If a partial migration left some tasks already in the
        store, they are skipped (upsert semantics handle duplicates).
        """
        if self._legacy_path is None or not self._legacy_path.exists():
            return

        try:
            data = json.loads(self._legacy_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read legacy schedules.json: %s", e)
            return

        if not isinstance(data, list):
            logger.warning("Legacy schedules.json is not a list — skipping migration")
            return

        migrated = 0
        for entry in data:
            try:
                task = ScheduledTask.from_dict(entry)
                _ = task.config  # validate schedule expression

                # Create-only: skip if a task with this name already exists
                # in the store (from a prior partial migration). Never
                # overwrite newer SQLite data with stale legacy JSON.
                if self._tasks.get(task.name) is not None:
                    continue

                rkey = f"{SCHEDULE_RKEY_PREFIX}{task.name}"
                content = json.dumps(task.to_dict())
                try:
                    await self._store.create(rkey, content)
                except Exception as e:
                    from src.store.store import StoreConflict

                    if isinstance(e, StoreConflict):
                        continue  # already exists in store — skip
                    raise

                self._tasks[task.name] = task
                migrated += 1
            except Exception as e:
                logger.warning("Failed to migrate schedule entry: %s", e)

        logger.info("Migrated %d schedule(s) from legacy JSON", migrated)

        # Rename the legacy file so we don't re-import
        try:
            self._legacy_path.rename(self._legacy_path.with_suffix(".json.migrated"))
        except OSError as e:
            logger.warning("Failed to rename legacy schedules.json: %s", e)

    async def load_tasks(self) -> None:
        """Load all scheduled tasks from the DocumentStore."""
        self._tasks.clear()

        # First, migrate from legacy JSON if needed
        await self._migrate_legacy()

        cursor: str | None = None
        while True:
            resp = await self._store.list(limit=100, cursor=cursor)
            documents = resp.get("documents", [])

            for doc in documents:
                rkey = doc.get("rkey", "")
                if not rkey.startswith(SCHEDULE_RKEY_PREFIX):
                    continue
                content = doc.get("content", "")
                if not content:
                    continue
                try:
                    data = json.loads(content)
                    task = ScheduledTask.from_dict(data)
                    _ = task.config  # validate the schedule expression
                    self._tasks[task.name] = task
                except Exception as e:
                    logger.warning("Failed to parse schedule entry %s: %s", rkey, e)

            cursor = resp.get("cursor")
            if not cursor or not documents:
                break

        logger.info("Loaded %d schedule(s)", len(self._tasks))

    def list_tasks(self) -> list[ScheduledTask]:
        return list(self._tasks.values())

    def get_task(self, name: str) -> ScheduledTask | None:
        return self._tasks.get(name)

    async def create_task(self, task: ScheduledTask) -> str:
        """Create a new scheduled task."""
        if not _SCHEDULE_NAME_PATTERN.match(task.name):
            return (
                f"Error: schedule name '{task.name}' is invalid. "
                "Allowed: alphanumeric, dash, underscore, dot (1-128 chars)"
            )

        # validate the schedule expression
        _ = task.config

        # set last_run to now so the first fire is after one interval/at next time
        if not task.last_run:
            task.last_run = datetime.now(timezone.utc).isoformat()

        # Create-only: reject duplicates rather than upserting
        rkey = f"{SCHEDULE_RKEY_PREFIX}{task.name}"
        content = json.dumps(task.to_dict())
        try:
            await self._store.create(rkey, content)
        except Exception as e:
            from src.store.store import StoreConflict

            if isinstance(e, StoreConflict):
                return f"Error: schedule '{task.name}' already exists."
            raise

        self._tasks[task.name] = task

        next_run = task.config.next_run(datetime.now(timezone.utc))
        msg = f"Schedule '{task.name}' created ({task.config}). Next run: {next_run.strftime('%Y-%m-%d %H:%M UTC')}."
        if task.condition_code is not None and not task.approved:
            msg += " Schedule has condition code — pending operator approval before it will fire."
        return msg

    ALLOWED_UPDATE_FIELDS = {
        "schedule",
        "prompt",
        "enabled",
        "last_run",
        "condition_code",
        "requires_net",
        "secrets",
    }

    async def update_task(self, name: str, **fields: Any) -> str:
        """Update an existing scheduled task."""
        task = self._tasks.get(name)
        if task is None:
            return f"Schedule '{name}' not found."

        skipped = []
        resets_approval = False
        # Snapshot for rollback if schedule validation fails
        old_values: dict[str, Any] = {}

        for key, value in fields.items():
            if key not in self.ALLOWED_UPDATE_FIELDS:
                skipped.append(key)
                continue
            if value is None:
                continue

            # Approval reset: condition_code changes, requires_net goes
            # False -> True (broadens permissions), or secrets are added
            # (new secret access not covered by prior approval).
            if key == "condition_code" and task.condition_code != value:
                resets_approval = True
            elif key == "requires_net" and not task.requires_net and value:
                resets_approval = True
            elif key == "secrets" and value is not None:
                old_set = set(task.secrets)
                new_set = set(value)
                if new_set - old_set:  # new secrets added (broadened)
                    resets_approval = True

            old_values[key] = getattr(task, key)
            setattr(task, key, value)

        if skipped:
            logger.warning(
                "update_task('%s') ignored unknown field(s): %s",
                name,
                ", ".join(skipped),
            )

        if resets_approval:
            task.approved = False

        # Validate the new schedule expression if it changed. Roll back
        # on failure so the in-memory task isn't poisoned.
        if "schedule" in fields and fields["schedule"] is not None:
            task._config = None
            try:
                _ = task.config
            except ValueError as e:
                # Roll back all changes
                for k, v in old_values.items():
                    setattr(task, k, v)
                task._config = None
                _ = task.config
                return f"Error: invalid schedule expression — {e}"

        await self._save_task(task)
        msg = f"Schedule '{name}' updated."
        if skipped:
            msg += f" Ignored unknown field(s): {', '.join(skipped)}."
        if resets_approval:
            msg += " Approval reset — pending operator re-approval."
        return msg

    async def delete_task(self, name: str) -> str:
        """Delete a scheduled task."""
        if name not in self._tasks:
            return f"Schedule '{name}' not found."

        self._tasks.pop(name, None)
        await self._delete_task(name)
        return f"Schedule '{name}' deleted."

    async def enable_task(self, name: str) -> str:
        return await self.update_task(name, enabled=True)

    async def disable_task(self, name: str) -> str:
        return await self.update_task(name, enabled=False)

    async def approve_task(self, name: str) -> str:
        """Approve a task's condition code for autonomous execution."""
        task = self._tasks.get(name)
        if task is None:
            return f"Schedule '{name}' not found."
        if task.condition_code is None:
            return f"Schedule '{name}' has no condition code — nothing to approve."

        task.approved = True
        await self._save_task(task)
        return f"Schedule '{name}' condition code approved."

    async def revoke_task(self, name: str) -> str:
        """Revoke approval for a task's condition code."""
        task = self._tasks.get(name)
        if task is None:
            return f"Schedule '{name}' not found."

        task.approved = False
        await self._save_task(task)
        return f"Schedule '{name}' condition code approval revoked."

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
            # never run — last_run is initialized in create_task, but if
            # we encounter a task without it, treat it as due now
            return True

        next_run = task.config.next_run(last)
        return now >= next_run

    async def _fire(self, task: ScheduledTask, now: datetime) -> None:
        """Execute a scheduled task."""
        logger.info("Firing schedule '%s': %s", task.name, task.prompt[:100])

        # --- Condition evaluation ---
        if task.condition_code:
            if not task.approved:
                logger.warning(
                    "Task '%s' has unapproved condition code — skipping", task.name
                )
                task.last_run = now.isoformat()
                await self._save_task(task)
                return

            outcome = await self._evaluate_condition(task)
            if outcome == "skip":
                task.last_run = now.isoformat()
                task._retry_count = 0
                task._condition_retry_count = 0
                await self._save_task(task)
                return
            elif outcome == "failed":
                await self._handle_condition_failure(task, now)
                return
            # outcome == "wake" — fall through to agent wake

        # --- Agent wake ---
        if self._on_fire:
            try:
                response = await self._on_fire(task.name, task.prompt)
                logger.info(
                    "Schedule '%s' completed: %s",
                    task.name,
                    (response[:200] if response else "(empty)"),
                )
                # Success — advance last_run and reset retry count
                task.last_run = now.isoformat()
                task._retry_count = 0
                task._condition_retry_count = 0
                await self._save_task(task)
            except Exception:
                task._retry_count += 1
                if task._retry_count >= 3:
                    logger.warning(
                        "Schedule '%s' failed %d consecutive times — "
                        "advancing last_run to prevent infinite retries",
                        task.name,
                        task._retry_count,
                    )
                    task.last_run = now.isoformat()
                    task._retry_count = 0
                    await self._save_task(task)
                else:
                    logger.exception(
                        "Schedule '%s' callback failed (attempt %d/3) — "
                        "will retry on next tick",
                        task.name,
                        task._retry_count,
                    )
        else:
            logger.info("Schedule '%s' fired (no callback wired)", task.name)
            task.last_run = now.isoformat()
            await self._save_task(task)

    async def _evaluate_condition(self, task: ScheduledTask) -> str:
        """Run the condition script. Returns 'wake', 'skip', or 'failed'."""
        if self._condition_runner is None:
            logger.error("No condition_runner wired for task '%s'", task.name)
            return "failed"

        try:
            result = await self._condition_runner(
                task.condition_code, task.requires_net, task.secrets
            )
        except Exception:
            logger.exception("Condition runner failed for task '%s'", task.name)
            return "failed"

        if not result.get("success"):
            logger.warning(
                "Condition script failed for '%s': %s",
                task.name,
                result.get("error", ""),
            )
            return "failed"

        output = result.get("output")
        if isinstance(output, dict) and output.get("wake", False):
            return "wake"
        return "skip"

    async def _handle_condition_failure(self, task: ScheduledTask, now: datetime) -> None:
        """Handle a condition script failure with 3-strike auto-disable."""
        task._condition_retry_count += 1
        if task._condition_retry_count >= 3:
            logger.warning(
                "Condition for '%s' failed %d consecutive times — auto-disabling",
                task.name,
                task._condition_retry_count,
            )
            task.enabled = False
            task.last_run = now.isoformat()
            task._condition_retry_count = 0
            await self._save_task(task)
        else:
            logger.warning(
                "Condition for '%s' failed (attempt %d/3) — will retry on next tick",
                task.name,
                task._condition_retry_count,
            )
