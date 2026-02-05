import logging
from datetime import datetime, timezone
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, TextArea

logger = logging.getLogger(__name__)


class ScheduleListScreen(Screen):
    """Screen showing all scheduled tasks with enable/disable/create controls."""

    BINDINGS = [
        Binding("n", "new_schedule", "New"),
        Binding("e", "enable_schedule", "Enable"),
        Binding("d", "disable_schedule", "Disable"),
        Binding("x", "delete_schedule", "Delete"),
        Binding("v", "view_schedule", "View"),
        Binding("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(name="Schedules")
        yield ListView(id="schedule-list")
        yield Footer()

    async def on_mount(self) -> None:
        await self._refresh_schedules()

    def _get_scheduler(self) -> Any:
        executor = getattr(self.app, "executor", None)
        if executor is None:
            return None
        return getattr(executor, "_scheduler", None)

    async def _refresh_schedules(self) -> None:
        list_view = self.query_one("#schedule-list", ListView)
        await list_view.clear()

        scheduler = self._get_scheduler()
        if scheduler is None:
            await list_view.append(
                ListItem(Label("Scheduler not available."))
            )
            return

        try:
            await scheduler.load_tasks()
        except Exception:
            logger.exception("Failed to reload schedules")

        tasks = scheduler.list_tasks()
        if not tasks:
            await list_view.append(
                ListItem(
                    Label(
                        "No scheduled tasks. Press [bold]n[/bold] to create one.",
                        markup=True,
                    )
                )
            )
            return

        now = datetime.now(timezone.utc)
        for task in tasks:
            status = "[green]enabled[/green]" if task.enabled else "[red]disabled[/red]"
            try:
                desc = str(task.config)
            except Exception:
                desc = task.schedule

            # next run
            next_str = ""
            if task.enabled and task.last_run:
                try:
                    last = datetime.fromisoformat(task.last_run)
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    next_run = task.config.next_run(last)
                    next_str = f"  next: {next_run.strftime('%m-%d %H:%M')}"
                except Exception:
                    pass

            label = f"{task.name}  {status}  {desc}{next_str}"
            item = ListItem(Label(label, markup=True))
            item._schedule_name = task.name  # type: ignore[attr-defined]
            await list_view.append(item)

    def _get_selected_name(self) -> str | None:
        list_view = self.query_one("#schedule-list", ListView)
        if list_view.highlighted_child is None:
            return None
        return getattr(list_view.highlighted_child, "_schedule_name", None)

    async def action_new_schedule(self) -> None:
        self.app.push_screen(
            ScheduleInputScreen(on_save=self._on_schedule_saved)
        )

    async def _on_schedule_saved(
        self, name: str, schedule: str, prompt: str
    ) -> None:
        scheduler = self._get_scheduler()
        if scheduler is None:
            self.notify("Scheduler not available", severity="error")
            return

        from src.scheduler.scheduler import ScheduledTask

        task = ScheduledTask(name=name, schedule=schedule, prompt=prompt, enabled=True)
        try:
            result = await scheduler.create_task(task)
            self.notify(result)
            await self._refresh_schedules()
        except (ValueError, RuntimeError) as e:
            self.notify(f"Invalid schedule: {e}", severity="error")
        except Exception:
            logger.exception("Failed to create schedule")
            self.notify("Failed to create schedule", severity="error")

    async def action_enable_schedule(self) -> None:
        scheduler = self._get_scheduler()
        name = self._get_selected_name()
        if not scheduler or not name:
            return

        try:
            result = await scheduler.enable_task(name)
            self.notify(result)
            await self._refresh_schedules()
        except Exception:
            logger.exception("Failed to enable schedule")
            self.notify("Failed to enable schedule", severity="error")

    async def action_disable_schedule(self) -> None:
        scheduler = self._get_scheduler()
        name = self._get_selected_name()
        if not scheduler or not name:
            return

        try:
            result = await scheduler.disable_task(name)
            self.notify(result)
            await self._refresh_schedules()
        except Exception:
            logger.exception("Failed to disable schedule")
            self.notify("Failed to disable schedule", severity="error")

    async def action_delete_schedule(self) -> None:
        scheduler = self._get_scheduler()
        name = self._get_selected_name()
        if not scheduler or not name:
            return

        try:
            result = await scheduler.delete_task(name)
            self.notify(result)
            await self._refresh_schedules()
        except Exception:
            logger.exception("Failed to delete schedule")
            self.notify("Failed to delete schedule", severity="error")

    async def action_view_schedule(self) -> None:
        scheduler = self._get_scheduler()
        name = self._get_selected_name()
        if not scheduler or not name:
            return

        task = scheduler.get_task(name)
        if task is None:
            self.notify(f"Schedule '{name}' not found", severity="error")
            return

        import json

        details = json.dumps(task.to_dict(), indent=2)
        if len(details) > 500:
            details = details[:500] + "\n..."
        self.notify(details, timeout=15)

    def action_back(self) -> None:
        self.app.pop_screen()


class ScheduleInputScreen(Screen):
    """Input screen for creating a new scheduled task."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, on_save: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._on_save = on_save

    def compose(self) -> ComposeResult:
        yield Header(name="New Schedule")
        yield Label("Schedule name:", id="name-label")
        yield Input(placeholder="my_task", id="schedule-name-input")
        yield Label("Schedule expression:", id="schedule-label")
        yield Input(
            placeholder='e.g. "daily@9:00", "30m", "weekly@monday"',
            id="schedule-expr-input",
        )
        yield Label("Prompt (instruction for the agent):", id="prompt-label")
        yield TextArea(id="schedule-prompt-input")
        yield Label(
            "Press Ctrl+S to save.  Formats: 30m, 2h, hourly, daily, "
            "daily@HH:MM, weekly@day, weekly@day@HH:MM, cron:expr",
            id="hint-label",
        )
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "schedule-name-input":
            self.query_one("#schedule-expr-input", Input).focus()
        elif event.input.id == "schedule-expr-input":
            self.query_one("#schedule-prompt-input", TextArea).focus()

    async def action_save(self) -> None:
        name = self.query_one("#schedule-name-input", Input).value.strip()
        schedule = self.query_one("#schedule-expr-input", Input).value.strip()
        prompt = self.query_one("#schedule-prompt-input", TextArea).text.strip()

        if not name:
            self.notify("Name is required", severity="error")
            return
        if not schedule:
            self.notify("Schedule expression is required", severity="error")
            return
        if not prompt:
            self.notify("Prompt is required", severity="error")
            return

        # validate the schedule expression before saving
        from src.scheduler.schedule import parse_schedule

        try:
            parse_schedule(schedule)
        except ValueError as e:
            self.notify(f"Invalid schedule: {e}", severity="error")
            return

        self.app.pop_screen()
        if self._on_save:
            await self._on_save(name, schedule, prompt)

    def action_cancel(self) -> None:
        self.app.pop_screen()
