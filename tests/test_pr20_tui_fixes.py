"""Tests for PR 20: Fix TUI — pop_screen ordering and confirmations."""

import pytest
import inspect


def test_confirm_screen_exists():
    """ConfirmScreen should exist and be a ModalScreen."""
    from src.tui.screens.confirm import ConfirmScreen
    from textual.screen import ModalScreen

    assert issubclass(ConfirmScreen, ModalScreen)


def test_confirm_screen_takes_message():
    """ConfirmScreen should accept a message parameter."""
    from src.tui.screens.confirm import ConfirmScreen

    screen = ConfirmScreen("Are you sure?")
    assert screen._message == "Are you sure?"


def test_confirm_screen_has_confirm_and_cancel_actions():
    """ConfirmScreen should have confirm and cancel actions."""
    from src.tui.screens.confirm import ConfirmScreen

    screen = ConfirmScreen("test")
    assert hasattr(screen, "action_confirm")
    assert hasattr(screen, "action_cancel")


def test_secret_input_clears_before_pop():
    """SecretInputScreen.on_input_submitted should clear inputs and await save before pop."""
    import ast

    with open("src/tui/screens/secrets.py") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "on_input_submitted":
                source = ast.unparse(node)
                # Check that pop_screen comes after _on_save
                pop_idx = source.find("pop_screen")
                save_idx = source.find("_on_save")
                assert pop_idx > save_idx, "pop_screen should come after _on_save"
                # Check that input values are cleared (empty string assignment)
                assert ".value = ''" in source or '.value = ""' in source, \
                    "Input values should be cleared"
                return
    assert False, "on_input_submitted method not found"


def test_schedule_save_awaits_before_pop():
    """ScheduleInputScreen.action_save should await _on_save before pop_screen."""
    import ast

    with open("src/tui/screens/schedules.py") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "action_save":
                source = ast.unparse(node)
                pop_idx = source.find("pop_screen")
                save_idx = source.find("_on_save")
                assert pop_idx > save_idx, "pop_screen should come after _on_save"
                return
    assert False, "action_save method not found"


def test_delete_actions_use_confirmation():
    """Delete/revoke actions should push a ConfirmScreen."""
    import subprocess

    files_and_actions = [
        ("src/tui/screens/secrets.py", "action_delete_secret"),
        ("src/tui/screens/schedules.py", "action_delete_schedule"),
        ("src/tui/screens/sessions.py", "action_delete_session"),
        ("src/tui/screens/tool_detail.py", "action_revoke"),
        ("src/tui/screens/tool_detail.py", "action_delete"),
    ]

    for filepath, action_name in files_and_actions:
        result = subprocess.run(
            ["grep", "-A15", action_name, filepath],
            capture_output=True,
            text=True,
        )
        assert "ConfirmScreen" in result.stdout, \
            f"{action_name} in {filepath} should use ConfirmScreen"
