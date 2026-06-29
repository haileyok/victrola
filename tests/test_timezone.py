"""Tests for operator-timezone-aware user-message timestamps."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src.agent.agent import _resolve_operator_tz, _timestamp_prefix
from src.config import CONFIG


def test_resolve_operator_tz_valid():
    with patch.object(CONFIG, "operator_timezone", "America/Los_Angeles"):
        assert _resolve_operator_tz() == ZoneInfo("America/Los_Angeles")


def test_resolve_operator_tz_invalid_falls_back_to_utc():
    with patch.object(CONFIG, "operator_timezone", "Not/AZone"):
        assert _resolve_operator_tz() == timezone.utc


def test_resolve_operator_tz_empty_resolves_to_utc():
    with patch.object(CONFIG, "operator_timezone", ""):
        tz = _resolve_operator_tz()
    # Empty -> "UTC" (a valid zone), so it resolves rather than hitting the
    # exception fallback; either way the offset must be zero.
    assert datetime(2026, 1, 1, tzinfo=tz).utcoffset() == timedelta(0)


def test_pacific_tz_handles_dst():
    tz = ZoneInfo("America/Los_Angeles")
    # January: PST (UTC-8). 08:00 UTC -> 00:00 local.
    assert datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc).astimezone(tz).hour == 0
    # July: PDT (UTC-7). 07:00 UTC -> 00:00 local.
    assert datetime(2026, 7, 1, 7, 0, tzinfo=timezone.utc).astimezone(tz).hour == 0


def test_timestamp_prefix_uses_operator_tz():
    with patch.object(CONFIG, "operator_timezone", "America/Los_Angeles"):
        out = _timestamp_prefix("hello")
    assert out.startswith("[") and out.endswith("] hello")
    # Stamped in Pacific time, not UTC.
    assert ("PST" in out) or ("PDT" in out)
    assert "UTC" not in out


def test_timestamp_prefix_defaults_to_utc():
    with patch.object(CONFIG, "operator_timezone", "UTC"):
        out = _timestamp_prefix("hi")
    assert "UTC" in out and out.endswith("] hi")
