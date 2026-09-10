"""RLT operator controls use synthetic key events and never connect hardware."""

import pytest

from lerobot.utils.control_utils import _KeyboardEventHandler


def _handler(phase="recording"):
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "toggle_intervention": False,
        "episode_outcome": None,
    }
    if phase is not None:
        events.update(rlt_phase=phase, rlt_reset_requested=False, rlt_start_requested=False)
    return _KeyboardEventHandler(events, "i", "s", "f")


@pytest.mark.parametrize(("key", "outcome"), [("s", "success"), ("F", "failure")])
def test_episode_result_latches_until_save_finishes(key, outcome):
    handler = _handler()
    handler._handle_key(key)
    assert handler.events["episode_outcome"] == outcome
    assert handler.events["exit_early"]
    assert handler.events["rlt_phase"] == "saving"
    saved = handler.events.copy()
    for other_key in ("s", "f", "r", "t", "LEFT", "RIGHT", "i"):
        handler._handle_key(other_key)
        assert handler.events == saved


def test_failure_requires_reset_before_next_episode():
    handler = _handler("failed_wait")
    initial = handler.events.copy()
    for key in ("t", "s", "f", "LEFT", "RIGHT", "i"):
        handler._handle_key(key)
        assert handler.events == initial

    handler._handle_key("R")
    assert handler.events["rlt_reset_requested"]
    assert handler.events["rlt_phase"] == "resetting"
    handler.events["rlt_reset_requested"] = False  # Recorder consumes the request.
    for key in ("r", "t"):
        handler._handle_key(key)
        assert not handler.events["rlt_reset_requested"]
        assert not handler.events["rlt_start_requested"]

    handler.events["rlt_phase"] = "ready_wait"  # Reset completed successfully.
    handler._handle_key("t")
    assert handler.events["rlt_start_requested"]
    assert handler.events["rlt_phase"] == "starting"


def test_timeout_or_unlabelled_failure_needs_f_acknowledgment_before_reset():
    handler = _handler("failure_ack_wait")
    handler.events["episode_outcome"] = "failure"
    initial = handler.events.copy()
    for key in ("r", "t", "s", "LEFT", "RIGHT", "i"):
        handler._handle_key(key)
        assert handler.events == initial

    handler._handle_key("F")
    assert handler.events == {**initial, "rlt_phase": "failed_wait"}
    handler._handle_key("f")
    assert not handler.events["exit_early"]
    handler._handle_key("r")
    assert handler.events["rlt_reset_requested"]
    assert handler.events["rlt_phase"] == "resetting"


@pytest.mark.parametrize("phase", ["success_wait", "ready_wait"])
def test_success_or_completed_reset_accepts_one_start_without_reset(phase):
    handler = _handler(phase)
    initial = handler.events.copy()
    for key in ("r", "s", "f", "LEFT", "RIGHT", "i"):
        handler._handle_key(key)
        assert handler.events == initial

    handler._handle_key("T")
    assert handler.events["rlt_start_requested"]
    assert handler.events["rlt_phase"] == "starting"
    assert not handler.events["rlt_reset_requested"]
    handler.events["rlt_start_requested"] = False
    handler._handle_key("t")
    assert not handler.events["rlt_start_requested"]


@pytest.mark.parametrize("phase", ["saving", "resetting", "starting", "complete", "unknown"])
def test_busy_or_unknown_phase_ignores_control_keys(phase):
    handler = _handler(phase)
    initial = handler.events.copy()
    for key in ("r", "t", "s", "f", "LEFT", "RIGHT", "i"):
        handler._handle_key(key)
        assert handler.events == initial


@pytest.mark.parametrize(
    "phase",
    [
        "recording",
        "saving",
        "failure_ack_wait",
        "failed_wait",
        "resetting",
        "ready_wait",
        "success_wait",
        "starting",
        "complete",
    ],
)
def test_escape_always_stops_and_preserves_label(phase):
    handler = _handler(phase)
    handler.events["episode_outcome"] = "failure"
    handler._handle_key("ESC")
    assert handler.events["exit_early"]
    assert handler.events["stop_recording"]
    assert handler.events["episode_outcome"] == "failure"
    assert handler.events["rlt_phase"] == phase
    assert not handler.events["rlt_start_requested"]
    assert not handler.events["rlt_reset_requested"]
    stopped = handler.events.copy()
    for key in ("r", "t", "s", "f", "LEFT", "RIGHT"):
        handler._handle_key(key)
        assert handler.events == stopped


@pytest.mark.parametrize("key", ["r", "t"])
def test_reset_and_start_are_ignored_during_inference(key):
    handler = _handler()
    initial = handler.events.copy()
    handler._handle_key(key)
    assert handler.events == initial


@pytest.mark.parametrize(("key", "discard"), [("RIGHT", False), ("LEFT", True)])
def test_recording_arrows_latch_without_fabricating_an_explicit_failure(key, discard):
    handler = _handler()
    handler._handle_key(key)
    assert handler.events["rlt_phase"] == "saving"
    assert handler.events["exit_early"]
    assert handler.events["rerecord_episode"] is discard
    assert handler.events["episode_outcome"] is None
    handler._handle_key("f")
    handler._handle_key("r")
    assert handler.events["episode_outcome"] is None
    assert not handler.events["rlt_reset_requested"]


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("s", {"episode_outcome": "success", "exit_early": True}),
        ("f", {"episode_outcome": "failure", "exit_early": True}),
        ("RIGHT", {"exit_early": True}),
        ("LEFT", {"exit_early": True, "rerecord_episode": True}),
        ("ESC", {"exit_early": True, "stop_recording": True}),
        ("r", {}),
        ("t", {}),
    ],
)
def test_other_recorders_keep_legacy_controls_without_an_rlt_phase(key, expected):
    handler = _handler(None)
    initial = handler.events.copy()
    handler._handle_key(key)
    assert handler.events == {**initial, **expected}


def test_other_recorders_do_not_latch_episode_labels():
    handler = _handler(None)
    handler._handle_key("s")
    handler._handle_key("f")
    assert handler.events["episode_outcome"] == "failure"
    assert "rlt_phase" not in handler.events
