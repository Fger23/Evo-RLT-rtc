"""Console input tests use fake keystrokes and never connect a robot."""

import subprocess
import sys
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.utils import control_utils


def _events():
    return {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "toggle_intervention": False,
        "episode_outcome": None,
    }


def _console(keys=()):
    pending = deque(keys)
    return SimpleNamespace(pending=pending, kbhit=lambda: bool(pending), getwch=pending.popleft)


def _windows_listener(keys=()):
    listener = control_utils.WindowsConsoleKeyboardListener(_events(), "i", "s", "f")
    listener._console = _console(keys)
    return listener


@pytest.mark.skipif(sys.platform != "win32", reason="Checks real native Windows imports without POSIX stubs")
def test_native_windows_recording_imports_without_posix_modules():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from lerobot.utils import control_utils; "
            "assert control_utils.termios is None and control_utils.tty is None; "
            "import lerobot.scripts.lerobot_rlt_record; "
            "import lerobot.scripts.lerobot_infer_trc; print('native-windows-import-ok')",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "native-windows-import-ok" in result.stdout


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["\xe0", "M"], {"exit_early": True}),
        (["\x00", "K"], {"exit_early": True, "rerecord_episode": True}),
        (["\x1b"], {"exit_early": True, "stop_recording": True}),
        (["S"], {"exit_early": True, "episode_outcome": "success"}),
        (["f"], {"exit_early": True, "episode_outcome": "failure"}),
        (["i"], {"toggle_intervention": True}),
    ],
)
def test_windows_keystrokes_update_recording_outcomes_and_control_flags(keys, expected):
    listener = _windows_listener(keys)
    listener._handle_key(listener._read_key())
    assert listener.events == {**_events(), **expected}


def test_windows_extended_key_can_arrive_across_polls_without_blocking():
    listener = _windows_listener(["\xe0"])
    assert listener._read_key() is None
    assert listener._read_key() is None
    listener._console.pending.append("M")
    assert listener._read_key() == "RIGHT"
    listener._console.pending.extend(["\x00", "H", "s"])
    assert listener._read_key() is None  # Ignore the complete unsupported arrow key.
    assert listener._read_key() == "s"


def test_console_intervention_key_is_case_insensitive_and_debounced(monkeypatch):
    listener = _windows_listener()
    monkeypatch.setattr(control_utils.time, "monotonic", Mock(side_effect=[1.0, 1.1, 1.7]))
    listener._handle_key("I")
    assert listener.events["toggle_intervention"]
    listener.events["toggle_intervention"] = False
    listener._handle_key("i")
    assert not listener.events["toggle_intervention"]
    listener._handle_key("i")
    assert listener.events["toggle_intervention"]


def test_windows_listener_processes_input_and_stops_while_idle(monkeypatch):
    console = _console(["s"])
    monkeypatch.setitem(sys.modules, "msvcrt", console)
    listener = _windows_listener()
    handled = threading.Event()
    original = listener._handle_key

    def handle(key):
        original(key)
        handled.set()

    listener._handle_key = handle
    listener.start()
    try:
        assert handled.wait(1.0)
        assert listener.events["episode_outcome"] == "success"
    finally:
        listener.stop()
    assert not listener.is_alive()


@pytest.mark.parametrize(
    ("platform", "expected_type"),
    [("win32", control_utils.WindowsConsoleKeyboardListener), ("linux", control_utils.TTYKeyboardListener)],
)
def test_headless_interactive_console_selects_platform_backend(monkeypatch, platform, expected_type):
    monkeypatch.setattr(
        control_utils,
        "sys",
        SimpleNamespace(platform=platform, stdin=SimpleNamespace(isatty=lambda: True, fileno=lambda: 3)),
    )
    monkeypatch.setattr(control_utils, "is_headless", lambda: True)
    monkeypatch.setattr(control_utils.WindowsConsoleKeyboardListener, "start", Mock())
    monkeypatch.setattr(control_utils.TTYKeyboardListener, "start", Mock())
    listener, events = control_utils.init_keyboard_listener(episode_success_key="s", episode_failure_key="f")
    assert isinstance(listener, expected_type)
    assert events == _events()
    expected_type.start.assert_called_once()


def test_failed_pynput_start_falls_back_to_windows_console(monkeypatch):
    failed_listener = SimpleNamespace(start=Mock(side_effect=OSError("hook unavailable")), stop=Mock())
    monkeypatch.setitem(
        sys.modules, "pynput", SimpleNamespace(keyboard=SimpleNamespace(Listener=lambda **_: failed_listener))
    )
    monkeypatch.setattr(control_utils, "is_headless", lambda: False)
    monkeypatch.setattr(
        control_utils, "sys", SimpleNamespace(platform="win32", stdin=SimpleNamespace(isatty=lambda: True))
    )
    monkeypatch.setattr(control_utils.WindowsConsoleKeyboardListener, "start", Mock())
    listener, _ = control_utils.init_keyboard_listener(episode_success_key="s", episode_failure_key="f")
    assert isinstance(listener, control_utils.WindowsConsoleKeyboardListener)
    failed_listener.stop.assert_called_once()


def test_redirected_stdin_does_not_attempt_console_keyboard_reads(monkeypatch):
    monkeypatch.setattr(control_utils, "is_headless", lambda: True)
    monkeypatch.setattr(
        control_utils, "sys", SimpleNamespace(platform="win32", stdin=SimpleNamespace(isatty=lambda: False))
    )
    start = Mock()
    monkeypatch.setattr(control_utils.WindowsConsoleKeyboardListener, "start", start)
    listener, events = control_utils.init_keyboard_listener()
    assert listener is None and events == _events()
    start.assert_not_called()


@pytest.mark.parametrize(
    ("sequence", "key"),
    [(b"\x1b[C", "RIGHT"), (b"\x1bOD", "LEFT"), (b"\x1b", "ESC"), (b"s", "s")],
)
def test_posix_terminal_key_sequences_remain_supported(monkeypatch, sequence, key):
    pending = deque(bytes([value]) for value in sequence)
    monkeypatch.setattr(control_utils, "sys", SimpleNamespace(stdin=SimpleNamespace(fileno=lambda: 3)))
    monkeypatch.setattr(control_utils, "os", SimpleNamespace(read=lambda *_: pending.popleft()))
    monkeypatch.setattr(
        control_utils, "select", SimpleNamespace(select=lambda *_: ([3] if pending else [], [], []))
    )
    listener = control_utils.TTYKeyboardListener(_events(), "i", "s", "f")
    assert listener._read_key() == key


def test_windows_calibration_enter_poll_does_not_select_stdin(monkeypatch):
    from lerobot.utils import utils

    monkeypatch.setattr(utils.platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(kbhit=lambda: True, getch=lambda: b"\r"))
    monkeypatch.setattr(
        utils.select, "select", Mock(side_effect=AssertionError("Windows stdin is not a socket"))
    )
    assert utils.enter_pressed()
