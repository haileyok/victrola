"""Absolute schedules (daily@/weekly@/cron) are interpreted in the operator's
local timezone and converted to UTC for firing — including across DST."""

from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from src.config import CONFIG
from src.scheduler.schedule import parse_schedule

LA = ZoneInfo("America/Los_Angeles")


def test_daily_default_utc_unchanged():
    cfg = parse_schedule("daily@9:00")
    with patch.object(CONFIG, "operator_timezone", "UTC"):
        nxt = cfg.next_run(datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc))
    assert (nxt.day, nxt.hour, nxt.minute) == (10, 9, 0)


def test_daily_interpreted_in_operator_tz_winter():
    """9:00 PST == 17:00 UTC."""
    cfg = parse_schedule("daily@9:00")
    with patch.object(CONFIG, "operator_timezone", "America/Los_Angeles"):
        # 2026-01-10 00:00 UTC == 2026-01-09 16:00 PST -> next 9:00 is 01-10.
        nxt = cfg.next_run(datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc))
    assert nxt.tzinfo is not None
    assert (nxt.day, nxt.hour) == (10, 17)
    assert nxt.astimezone(LA).hour == 9


def test_daily_interpreted_in_operator_tz_summer_dst():
    """9:00 PDT == 16:00 UTC (one hour different from winter -> DST handled)."""
    cfg = parse_schedule("daily@9:00")
    with patch.object(CONFIG, "operator_timezone", "America/Los_Angeles"):
        nxt = cfg.next_run(datetime(2026, 7, 10, 0, 0, tzinfo=timezone.utc))
    assert (nxt.day, nxt.hour) == (10, 16)
    assert nxt.astimezone(LA).hour == 9


def test_weekly_interpreted_in_operator_tz():
    cfg = parse_schedule("weekly@monday@9:00")
    with patch.object(CONFIG, "operator_timezone", "America/Los_Angeles"):
        nxt = cfg.next_run(datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc))
    local = nxt.astimezone(LA)
    assert local.weekday() == 0  # Monday
    assert (local.hour, local.minute) == (9, 0)


def test_interval_is_timezone_independent():
    cfg = parse_schedule("2h")
    from_time = datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc)
    with patch.object(CONFIG, "operator_timezone", "America/Los_Angeles"):
        nxt = cfg.next_run(from_time)
    assert nxt == from_time.replace(hour=2)


def test_cron_interpreted_in_operator_tz():
    pytest.importorskip("croniter")
    with patch.object(CONFIG, "operator_timezone", "America/Los_Angeles"):
        cfg = parse_schedule("cron:0 9 * * *")
        nxt = cfg.next_run(datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc))
    assert nxt.astimezone(LA).hour == 9
