"""Schedule expression parsing and next-run calculation.

Supported formats:
  - Duration:   "30m", "2h", "1h30m", "90s"
  - Keywords:   "hourly", "daily", "weekly"
  - Daily:      "daily@9:00", "daily@14:30"
  - Weekly:     "weekly@monday", "weekly@fri"
  - Cron:       "cron:0 9 * * *"  (requires croniter)
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# -- patterns --

DURATION_PATTERN = re.compile(r"^(\d+[hms])+$")
DAILY_TIME_PATTERN = re.compile(r"^daily@(\d{1,2}):(\d{2})$")
WEEKLY_PATTERN = re.compile(r"^weekly(?:@(\w+))?$")
WEEKLY_TIME_PATTERN = re.compile(r"^weekly@(\w+)@(\d{1,2}):(\d{2})$")
CRON_PATTERN = re.compile(r"^cron:(.+)$")

# Python weekday(): 0=Monday … 6=Sunday
WEEKDAY_NAMES: dict[str, int] = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

WEEKDAY_DISPLAY = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _parse_duration(expr: str) -> timedelta:
    """Parse a Go-style duration like '30m', '2h', '1h30m'."""
    total = timedelta()
    remaining = expr
    while remaining:
        m = re.match(r"(\d+)([hms])", remaining)
        if not m:
            raise ValueError(f"Invalid duration component in: {expr}")
        val = int(m.group(1))
        unit = m.group(2)
        if unit == "h":
            total += timedelta(hours=val)
        elif unit == "m":
            total += timedelta(minutes=val)
        elif unit == "s":
            total += timedelta(seconds=val)
        remaining = remaining[m.end() :]
    return total


@dataclass
class ScheduleConfig:
    type: str  # "interval", "hourly", "daily", "weekly", "cron"
    expression: str
    interval: timedelta | None = None
    hour: int = 0
    minute: int = 0
    weekday: int = 0  # 0=Monday
    cron_expr: str | None = None

    def next_run(self, from_time: datetime) -> datetime:
        """Calculate the next run time after `from_time`."""
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)

        match self.type:
            case "interval" | "hourly":
                assert self.interval is not None
                return from_time + self.interval

            case "daily":
                next_t = from_time.replace(
                    hour=self.hour, minute=self.minute, second=0, microsecond=0
                )
                if next_t <= from_time:
                    next_t += timedelta(days=1)
                return next_t

            case "weekly":
                next_t = from_time.replace(
                    hour=self.hour, minute=self.minute, second=0, microsecond=0
                )
                days_until = self.weekday - next_t.weekday()
                if days_until < 0 or (days_until == 0 and next_t <= from_time):
                    days_until += 7
                return next_t + timedelta(days=days_until)

            case "cron":
                try:
                    from croniter import croniter  # type: ignore[import-untyped]

                    return croniter(self.cron_expr, from_time).get_next(datetime)  # type: ignore[return-value]
                except ImportError:
                    raise RuntimeError(
                        "croniter package required for cron schedules: uv add croniter"
                    )

            case _:
                return from_time + timedelta(hours=1)

    def __str__(self) -> str:
        match self.type:
            case "interval":
                return f"every {self.interval}"
            case "hourly":
                return "every hour"
            case "daily":
                if self.hour == 0 and self.minute == 0:
                    return "daily at midnight"
                return f"daily at {self.hour:02d}:{self.minute:02d}"
            case "weekly":
                day = WEEKDAY_DISPLAY[self.weekday]
                if self.hour == 0 and self.minute == 0:
                    return f"weekly on {day}"
                return f"weekly on {day} at {self.hour:02d}:{self.minute:02d}"
            case "cron":
                return f"cron: {self.cron_expr}"
            case _:
                return self.expression


def parse_schedule(expr: str) -> ScheduleConfig:
    """Parse a human-friendly schedule expression into a ScheduleConfig."""
    expr = expr.strip().lower()
    if not expr:
        raise ValueError("Empty schedule expression")

    # cron:...
    m = CRON_PATTERN.match(expr)
    if m:
        cron_expr = m.group(1).strip()
        try:
            from croniter import croniter  # type: ignore[import-untyped]

            croniter(cron_expr)  # validate
        except ImportError:
            raise RuntimeError(
                "croniter package required for cron schedules: uv add croniter"
            )
        return ScheduleConfig(type="cron", expression=expr, cron_expr=cron_expr)

    # simple keywords
    if expr == "hourly":
        return ScheduleConfig(
            type="hourly", expression=expr, interval=timedelta(hours=1)
        )
    if expr == "daily":
        return ScheduleConfig(type="daily", expression=expr)
    if expr == "weekly":
        return ScheduleConfig(type="weekly", expression=expr, weekday=0)

    # daily@HH:MM
    m = DAILY_TIME_PATTERN.match(expr)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        if not 0 <= hour <= 23:
            raise ValueError(f"Invalid hour: {hour} (must be 0-23)")
        if not 0 <= minute <= 59:
            raise ValueError(f"Invalid minute: {minute} (must be 0-59)")
        return ScheduleConfig(
            type="daily", expression=expr, hour=hour, minute=minute
        )

    # weekly@day@HH:MM  (e.g. weekly@monday@9:00)
    m = WEEKLY_TIME_PATTERN.match(expr)
    if m:
        day_name = m.group(1)
        if day_name not in WEEKDAY_NAMES:
            raise ValueError(f"Invalid weekday: {day_name}")
        hour, minute = int(m.group(2)), int(m.group(3))
        if not 0 <= hour <= 23:
            raise ValueError(f"Invalid hour: {hour} (must be 0-23)")
        if not 0 <= minute <= 59:
            raise ValueError(f"Invalid minute: {minute} (must be 0-59)")
        return ScheduleConfig(
            type="weekly",
            expression=expr,
            weekday=WEEKDAY_NAMES[day_name],
            hour=hour,
            minute=minute,
        )

    # weekly@day
    m = WEEKLY_PATTERN.match(expr)
    if m:
        weekday = 0
        if m.group(1):
            day_name = m.group(1)
            if day_name not in WEEKDAY_NAMES:
                raise ValueError(f"Invalid weekday: {day_name}")
            weekday = WEEKDAY_NAMES[day_name]
        return ScheduleConfig(type="weekly", expression=expr, weekday=weekday)

    # duration: 30m, 2h, 1h30m
    if DURATION_PATTERN.match(expr):
        interval = _parse_duration(expr)
        if interval < timedelta(minutes=1):
            raise ValueError("Interval must be at least 1 minute")
        return ScheduleConfig(type="interval", expression=expr, interval=interval)

    raise ValueError(f"Unrecognized schedule format: {expr}")
