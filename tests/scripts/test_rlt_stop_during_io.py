"""Interrupted RLT I/O must preserve buffered frames without dispatching stale actions."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.scripts import recording_loop
from lerobot.utils.control_utils import _KeyboardEventHandler


@pytest.fixture
def capture(monkeypatch):
    events = {
        "rlt_phase": "recording",
        "exit_early": False,
        "stop_recording": False,
        "rerecord_episode": False,
        "toggle_intervention": False,
        "episode_outcome": None,
    }
    frames = []
    feature = {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}
    dataset = SimpleNamespace(
        fps=30,
        features={
            "action": feature,
            "observation.state": feature,
            "complementary_info.policy_action": feature,
        },
        add_frame=frames.append,
        image_writer=SimpleNamespace(stop=Mock()),
    )
    robot = SimpleNamespace(
        action_features={"joint.pos": float},
        get_observation=Mock(return_value={"joint.pos": 1.0}),
        send_action=Mock(side_effect=lambda action: dict(action)),
    )
    client = SimpleNamespace(
        reset=Mock(),
        get_action=Mock(return_value={"joint.pos": 2.0}),
        confirm_action_executed=Mock(),
    )
    handler = _KeyboardEventHandler(events, "i", "s", "f")
    monkeypatch.setattr(recording_loop, "precise_sleep", lambda _: None)

    def run(**overrides):
        return recording_loop.record_loop(
            robot=robot,
            events=events,
            fps=30,
            teleop_action_processor=lambda pair: pair[0],
            robot_action_processor=lambda pair: pair[0],
            robot_observation_processor=lambda observation: observation,
            dataset=dataset,
            remote_policy_client=client,
            control_time_s=1,
            single_task="Count banknotes on both sides",
            **overrides,
        )

    return SimpleNamespace(
        events=events,
        frames=frames,
        dataset=dataset,
        robot=robot,
        client=client,
        press=handler._handle_key,
        run=run,
    )


@pytest.mark.parametrize("key", ["s", "f", "ESC"])
@pytest.mark.parametrize("stage", ["observation", "rpc"])
def test_label_or_escape_during_io_does_not_send_the_returned_action(capture, key, stage):
    def interrupted_call(*args, **kwargs):
        capture.press(key)
        return {"joint.pos": 50.0}

    target = capture.robot.get_observation if stage == "observation" else capture.client.get_action
    target.side_effect = interrupted_call
    capture.run()

    capture.robot.send_action.assert_not_called()
    capture.client.confirm_action_executed.assert_not_called()
    assert capture.frames == []
    assert capture.events["episode_outcome"] == {"s": "success", "f": "failure", "ESC": None}[key]
    assert capture.events["stop_recording"] is (key == "ESC")
    capture.dataset.image_writer.stop.assert_not_called()


@pytest.mark.parametrize(
    ("stage", "error_type"),
    [
        ("observation", TimeoutError),
        ("observation", ConnectionError),
        ("rpc", TimeoutError),
        ("rpc", ConnectionError),
        ("rpc", RuntimeError),  # RTC wraps failed background RPCs in RuntimeError.
    ],
)
def test_pending_failure_and_io_error_return_existing_frames_for_save(capture, stage, error_type):
    calls = 0

    def interrupted_second_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            capture.press("f")
            raise error_type("I/O failed after the operator stopped")
        return {"joint.pos": 2.0 if stage == "rpc" else 1.0}

    target = capture.robot.get_observation if stage == "observation" else capture.client.get_action
    target.side_effect = interrupted_second_call
    capture.run()

    assert calls == 2
    capture.robot.send_action.assert_called_once_with({"joint.pos": 2.0})
    capture.client.confirm_action_executed.assert_called_once_with({"joint.pos": 2.0})
    assert len(capture.frames) == 1
    np.testing.assert_array_equal(capture.frames[0]["action"], [2.0])
    np.testing.assert_array_equal(capture.frames[0]["observation.state"], [1.0])
    assert capture.events["episode_outcome"] == "failure"
    # Returning normally leaves image writing available for finish_episode's save/flush.
    capture.dataset.image_writer.stop.assert_not_called()


@pytest.mark.parametrize("stage", ["observation", "rpc", "reset"])
def test_unrequested_io_timeout_is_not_silently_treated_as_a_completed_episode(capture, stage):
    target = {
        "observation": capture.robot.get_observation,
        "rpc": capture.client.get_action,
        "reset": capture.client.reset,
    }[stage]
    target.side_effect = TimeoutError("genuine timeout")
    with pytest.raises(TimeoutError, match="genuine timeout"):
        capture.run()
    capture.robot.send_action.assert_not_called()
    capture.client.confirm_action_executed.assert_not_called()
    assert capture.frames == []


def test_motor_runtime_error_is_not_suppressed_by_a_pending_failure_key(capture):
    def overloaded_observation():
        capture.press("f")
        raise RuntimeError("motor overload")

    capture.robot.get_observation.side_effect = overloaded_observation
    with pytest.raises(RuntimeError, match="motor overload"):
        capture.run()
    capture.robot.send_action.assert_not_called()
    capture.client.confirm_action_executed.assert_not_called()


@pytest.mark.parametrize("key", ["f", "ESC"])
def test_stop_during_initial_rtc_reset_timeout_returns_without_robot_dispatch(capture, key):
    def interrupted_reset():
        capture.press(key)
        raise TimeoutError("old RTC request did not finish")

    capture.client.reset.side_effect = interrupted_reset
    capture.run()
    capture.robot.get_observation.assert_not_called()
    capture.robot.send_action.assert_not_called()
    capture.client.confirm_action_executed.assert_not_called()
    assert capture.frames == []


@pytest.mark.parametrize("key", ["s", "f", "ESC"])
@pytest.mark.parametrize("synchronized", [False, True])
def test_stop_during_dispatch_retry_sleep_prevents_retry_and_false_confirmation(
    capture, monkeypatch, key, synchronized
):
    dispatch = Mock(side_effect=[ConnectionError("transient send failure"), {"joint.pos": 2.0}])
    overrides = {}
    if synchronized:
        overrides["policy_sync_executor"] = SimpleNamespace(send_action=dispatch)
    else:
        capture.robot.send_action = dispatch
    sleep = Mock(side_effect=lambda _: capture.press(key))
    monkeypatch.setattr(
        recording_loop,
        "time",
        SimpleNamespace(perf_counter=recording_loop.time.perf_counter, sleep=sleep),
    )
    capture.run(communication_retry_timeout_s=1, communication_retry_interval_s=0.01, **overrides)

    dispatch.assert_called_once_with({"joint.pos": 2.0})
    sleep.assert_called_once()
    if synchronized:
        capture.robot.send_action.assert_not_called()
    capture.client.confirm_action_executed.assert_not_called()
    assert capture.frames == []


@pytest.mark.parametrize("key", ["s", "f", "ESC"])
def test_key_latched_before_record_loop_does_not_reset_or_start_remote_inference(capture, key):
    capture.press(key)
    capture.run()
    capture.client.reset.assert_not_called()
    capture.client.get_action.assert_not_called()
    capture.robot.get_observation.assert_not_called()
    capture.robot.send_action.assert_not_called()
    capture.client.confirm_action_executed.assert_not_called()
