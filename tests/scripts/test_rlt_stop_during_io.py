"""Interrupted RLT I/O must preserve buffered frames without dispatching stale actions."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from lerobot.cameras.camera import CameraFrameTimeoutError
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


@pytest.mark.parametrize("stage", ["observation", "rpc"])
def test_camera_recovery_resumes_with_fresh_observation_without_a_failed_frame(capture, stage):
    operations = []
    capture.robot.bus = SimpleNamespace(sync_read=Mock(return_value={"joint": 7.0}))
    capture.client.suspend = Mock(side_effect=lambda: operations.append("suspend"))
    capture.client.reset.side_effect = lambda: operations.append("reset")
    observation_calls = 0
    action_calls = 0

    def observation():
        nonlocal observation_calls
        observation_calls += 1
        if observation_calls == 1:
            operations.append("initial_observation")
            if stage == "observation":
                raise CameraFrameTimeoutError("camera missed a frame")
            return {"joint.pos": 1.0}
        if observation_calls == 2:
            operations.append("probe")
            assert capture.client.reset.call_count == 1
            return {"joint.pos": 11.0}  # Before RTC drain; must never reach policy or data.
        operations.append("fresh_observation")
        assert capture.client.reset.call_count == 2
        return {"joint.pos": float(9 + observation_calls)}

    def action(*, observation, task, timestep):
        nonlocal action_calls
        action_calls += 1
        assert timestep == 0  # The interrupted action must not consume a recorded step.
        if stage == "rpc" and action_calls == 1:
            assert observation == {"joint.pos": 1.0}
            operations.append("internal_camera_timeout")
            raise CameraFrameTimeoutError("RTC queue refill camera missed a frame")
        assert observation == {"joint.pos": 12.0 if stage == "observation" else 13.0}
        operations.append("policy")
        return {"joint.pos": 22.0}

    def dispatch(action):
        operations.append("hold" if action == {"joint.pos": 7.0} else "dispatch")
        return dict(action)

    def save(frame):
        capture.frames.append(frame)
        capture.press("f")  # End after one accepted frame without another observation/read.

    capture.robot.get_observation.side_effect = observation
    capture.robot.send_action.side_effect = dispatch
    capture.client.get_action.side_effect = action
    capture.dataset.add_frame = save
    capture.run()

    assert operations[:2] == ["reset", "initial_observation"]
    recovery_start = operations.index("suspend")
    assert operations[recovery_start : recovery_start + 5] == [
        "suspend",
        "hold",
        "probe",
        "reset",
        "fresh_observation",
    ]
    assert operations[-2:] == ["policy", "dispatch"]
    capture.client.suspend.assert_called_once()
    capture.robot.bus.sync_read.assert_called_once_with("Present_Position")
    assert capture.robot.send_action.call_args_list == [call({"joint.pos": 7.0}), call({"joint.pos": 22.0})]
    capture.client.confirm_action_executed.assert_called_once_with({"joint.pos": 22.0})
    assert capture.client.get_action.call_count == (1 if stage == "observation" else 2)
    assert len(capture.frames) == 1
    np.testing.assert_array_equal(capture.frames[0]["action"], [22.0])
    np.testing.assert_array_equal(
        capture.frames[0]["observation.state"], [12.0 if stage == "observation" else 13.0]
    )
    assert capture.events["episode_outcome"] == "failure"
    capture.dataset.image_writer.stop.assert_not_called()


@pytest.mark.parametrize("stage", ["observation", "rpc"])
def test_camera_timeout_outside_rlt_does_not_enable_recovery(capture, stage):
    del capture.events["rlt_phase"]
    capture.client.suspend = Mock()
    target = capture.robot.get_observation if stage == "observation" else capture.client.get_action
    target.side_effect = CameraFrameTimeoutError("camera missed a frame")
    with pytest.raises(CameraFrameTimeoutError, match="camera missed a frame"):
        capture.run()
    capture.client.suspend.assert_not_called()
    capture.robot.send_action.assert_not_called()
    capture.client.confirm_action_executed.assert_not_called()
    assert capture.frames == []
