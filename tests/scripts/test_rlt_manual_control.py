"""Manual RLT episode sequencing; all robot commands use simulated buses."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lerobot.scripts import recording_rlt_control as control
from lerobot.utils.control_utils import _KeyboardEventHandler


def setup_controller():
    robot = SimpleNamespace(
        left_arm=SimpleNamespace(bus=Mock()),
        right_arm=SimpleNamespace(bus=Mock()),
        action_features={"left_joint.pos": float, "right_joint.pos": float},
        send_action=Mock(),
        get_observation=Mock(side_effect=AssertionError("Paused/reset control must not read cameras")),
    )
    robot.left_arm.bus.sync_read.return_value = {"joint": 10.0}
    robot.right_arm.bus.sync_read.return_value = {"joint": -10.0}

    def follow(action):
        robot.left_arm.bus.sync_read.return_value = {"joint": action["left_joint.pos"]}
        robot.right_arm.bus.sync_read.return_value = {"joint": action["right_joint.pos"]}
        return action

    robot.send_action.side_effect = follow
    controller = control.RLTManualEpisodeController(reset_duration_s=0.1)
    controller.on_record_connected(robot, None)
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "toggle_intervention": False,
        "episode_outcome": None,
        "rlt_phase": "recording",
    }
    handler = _KeyboardEventHandler(events, "i", "s", "f")
    return controller, robot, events, handler


def fake_dataset(handler, operations):
    dataset = SimpleNamespace(episode_buffer={"size": 3}, num_episodes=0)

    def save(**kwargs):
        operations.append(("save", kwargs["extra_episode_metadata"]["episode_success"]))
        assert handler.events["rlt_phase"] == "saving"
        for key in ("r", "t", "s", "f"):
            handler._handle_key(key)
        assert handler.events["rlt_phase"] == "saving"
        dataset.num_episodes += 1
        dataset.episode_buffer = {"size": 0}

    dataset.save_episode = Mock(side_effect=save)
    dataset.flush = Mock(side_effect=lambda: operations.append(("flush", None)))
    dataset.clear_episode_buffer = Mock()
    return dataset


def test_failure_flushes_then_reset_then_t_and_success_continues_without_reset(monkeypatch):
    controller, robot, events, handler = setup_controller()
    operations = []
    client = SimpleNamespace(suspend=Mock(side_effect=lambda: operations.append(("suspend", None))))
    dataset = fake_dataset(handler, operations)
    robot.left_arm.bus.sync_read.return_value = {"joint": 30.0}
    robot.right_arm.bus.sync_read.return_value = {"joint": -30.0}

    def press_when_ready(_):
        phase = events["rlt_phase"]
        if phase == "failed_wait":
            assert operations == [("suspend", None), ("save", "failure"), ("flush", None)]
            assert robot.send_action.call_count == 1  # Current-position hold only.
            handler._handle_key("t")
            assert events["rlt_phase"] == "failed_wait"
            handler._handle_key("r")
        elif phase == "resetting":
            handler._handle_key("t")
            assert not events.get("rlt_start_requested")
        elif phase in {"ready_wait", "success_wait"}:
            handler._handle_key("t")
        else:
            pytest.fail(f"Unexpected wait phase {phase}")

    monkeypatch.setattr(control.time, "sleep", press_when_ready)
    handler._handle_key("f")
    assert controller.finish_episode(robot, None, client, dataset, events, "failure", has_more=True)
    assert events["rlt_phase"] == "recording"
    assert robot.send_action.call_args_list[1].args[0] == {"left_joint.pos": 20.0, "right_joint.pos": -20.0}
    assert robot.send_action.call_args_list[-1].args[0] == controller.start_pose
    assert controller.start_pose == {"left_joint.pos": 10.0, "right_joint.pos": -10.0}
    robot.send_action.reset_mock()
    dataset.episode_buffer = {"size": 3}
    handler._handle_key("s")
    assert controller.finish_episode(robot, None, client, dataset, events, "success", has_more=True)
    assert robot.send_action.call_count == 1  # Hold only; no return-to-start commands.
    assert dataset.num_episodes == 2
    assert events["episode_outcome"] is None


def test_first_start_requires_t_and_captures_a_new_pose_each_session(monkeypatch):
    controller, robot, events, handler = setup_controller()
    events["rlt_phase"] = "ready_wait"

    def ready(_):
        handler._handle_key("r")
        assert not robot.send_action.called
        handler._handle_key("t")

    monkeypatch.setattr(control.time, "sleep", ready)
    controller.start_session(robot, None, events)
    assert events["rlt_phase"] == "recording"
    robot.left_arm.bus.sync_read.return_value = {"joint": 45.0}
    next_session = control.RLTManualEpisodeController()
    next_session.on_record_connected(robot, None)
    assert next_session.start_pose["left_joint.pos"] == 45.0
    assert controller.start_pose["left_joint.pos"] == 10.0


def test_esc_during_save_keeps_label_and_never_resets(monkeypatch):
    controller, robot, events, handler = setup_controller()
    dataset = fake_dataset(handler, [])
    client = SimpleNamespace(suspend=Mock())
    handler._handle_key("f")
    dataset.flush.side_effect = lambda: handler._handle_key("ESC")
    assert controller.finish_episode(robot, None, client, dataset, events, "failure", has_more=True)
    dataset.save_episode.assert_called_once_with(extra_episode_metadata={"episode_success": "failure"})
    assert robot.send_action.call_count == 1  # Hold before saving.


def test_esc_interrupts_reset_and_never_starts_inference(monkeypatch):
    controller, robot, events, handler = setup_controller()
    events["rlt_phase"] = "resetting"
    monkeypatch.setattr(control.time, "sleep", lambda _: handler._handle_key("ESC"))
    controller.reset(robot, None, events)
    assert robot.send_action.call_count == 2  # First reset step, then hold on interruption.
    assert events["stop_recording"]
    assert not events.get("rlt_start_requested")


def test_failed_disk_write_never_enables_reset_or_start():
    controller, robot, events, handler = setup_controller()
    dataset = fake_dataset(handler, [])
    dataset.flush.side_effect = OSError("disk full")
    handler._handle_key("f")
    with pytest.raises(OSError, match="disk full"):
        controller.finish_episode(
            robot, None, SimpleNamespace(suspend=Mock()), dataset, events, "failure", has_more=True
        )
    handler._handle_key("r")
    handler._handle_key("t")
    assert events["rlt_phase"] == "saving"
    assert robot.send_action.call_count == 1  # No motion beyond the initial hold.


@pytest.mark.parametrize("label", ["success", "failure"])
def test_final_episode_holds_until_esc_and_disallows_more_inference(monkeypatch, label):
    controller, robot, events, handler = setup_controller()
    dataset = fake_dataset(handler, [])
    handler._handle_key("s" if label == "success" else "f")

    def stop_after_reset(_):
        phase = events["rlt_phase"]
        handler._handle_key("t")
        assert not events.get("rlt_start_requested")
        if phase == "failed_wait":
            handler._handle_key("r")
        elif phase == "complete":
            handler._handle_key("ESC")

    monkeypatch.setattr(control.time, "sleep", stop_after_reset)
    controller.finish_episode(
        robot, None, SimpleNamespace(suspend=Mock()), dataset, events, label, has_more=False
    )
    assert events["stop_recording"]
    assert robot.send_action.call_count == (3 if label == "failure" else 1)


def test_overload_during_reset_is_not_ignored_or_followed_by_t(monkeypatch):
    controller, robot, events, handler = setup_controller()
    robot.send_action.side_effect = RuntimeError("Overload error")
    events["rlt_phase"] = "failed_wait"
    handler._handle_key("r")
    with pytest.raises(RuntimeError, match="Overload"):
        controller.wait_for_start(robot, None, events)
    handler._handle_key("t")
    assert not events.get("rlt_start_requested")


def test_hold_failure_still_saves_episode_but_does_not_enable_motion():
    controller, robot, events, handler = setup_controller()
    dataset = fake_dataset(handler, [])
    robot.send_action.side_effect = RuntimeError("Overload error")
    handler._handle_key("f")
    with pytest.raises(RuntimeError, match="could not hold"):
        controller.finish_episode(
            robot, None, SimpleNamespace(suspend=Mock()), dataset, events, "failure", has_more=True
        )
    dataset.save_episode.assert_called_once()
    dataset.flush.assert_called_once()
    handler._handle_key("r")
    handler._handle_key("t")
    assert events["rlt_phase"] == "saving"


def test_reset_that_does_not_reach_target_never_enables_t(monkeypatch):
    controller, robot, events, handler = setup_controller()
    robot.left_arm.bus.sync_read.return_value = {"joint": 50.0}
    robot.send_action.side_effect = lambda action: action  # Jammed motor does not follow commands.
    events["rlt_phase"] = "failed_wait"
    handler._handle_key("r")
    monkeypatch.setattr(control.time, "sleep", lambda _: None)
    monkeypatch.setattr(control.time, "monotonic", Mock(side_effect=[0.0, 4.0]))
    with pytest.raises(RuntimeError, match="did not reach"):
        controller.wait_for_start(robot, None, events)
    handler._handle_key("t")
    assert not events.get("rlt_start_requested")
    assert robot.send_action.call_args.args[0]["left_joint.pos"] == 50.0  # Final hold at measured pose.


def test_full_recorder_saves_before_manual_reset_and_never_runs_timed_reset(monkeypatch):
    from contextlib import nullcontext

    import draccus

    from lerobot.scripts import lerobot_record as recording, recording_trained_rtc
    from lerobot.scripts.lerobot_rlt_record import RLTRecordConfig, rlt_record

    _, robot, events, handler = setup_controller()
    robot.name = "bi_so_follower"
    robot.cameras = {}
    robot.observation_features = {}
    robot.config = SimpleNamespace()
    robot.is_connected = False
    robot.connect = Mock(side_effect=lambda: setattr(robot, "is_connected", True))
    robot.disconnect = Mock(side_effect=lambda: setattr(robot, "is_connected", False))
    operations = []
    dataset = fake_dataset(handler, operations)
    dataset.repo_id = "local/test"
    dataset.finalize = Mock()
    client = SimpleNamespace(start=Mock(), suspend=Mock(), stop=Mock())
    listener = Mock()
    cfg = draccus.parse(RLTRecordConfig, config_path="configs/rlt/windows_record.json", args=[])
    cfg.dataset.num_episodes = 2
    cfg.reset_duration_s = 0.1
    monkeypatch.setattr("lerobot.scripts.lerobot_rlt_record.record_to_target", lambda cfg, fn: fn(cfg))
    monkeypatch.setattr(recording, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(recording, "make_default_processors", lambda: (None, None, None))
    monkeypatch.setattr(recording, "aggregate_pipeline_dataset_features", lambda **_: {})
    monkeypatch.setattr(recording, "create_initial_features", lambda **_: {})
    monkeypatch.setattr(
        recording, "combine_feature_dicts", lambda *_: {"action": {"names": list(robot.action_features)}}
    )
    monkeypatch.setattr(recording, "_ensure_human_inloop_compatible_features", lambda *_, **__: None)
    monkeypatch.setattr(recording, "sanity_check_dataset_name", lambda *_: None)
    monkeypatch.setattr(recording.LeRobotDataset, "create", lambda *_, **__: dataset)
    monkeypatch.setattr(recording, "VideoEncodingManager", lambda _: nullcontext())
    monkeypatch.setattr(recording, "init_keyboard_listener", lambda **_: (listener, events))
    monkeypatch.setattr(recording_trained_rtc, "TrainedRTCRecordingClient", lambda **_: client)
    calls = []

    def episode(**kwargs):
        assert kwargs.get("dataset") is dataset  # No timed reset loop is permitted.
        assert events["rlt_phase"] == "recording"
        dataset.episode_buffer = {"size": 3}
        calls.append(len(calls))
        handler._handle_key("f" if len(calls) == 1 else "s")

    def operator(_):
        phase = events["rlt_phase"]
        if phase == "failed_wait":
            assert dataset.num_episodes == 1 and dataset.flush.call_count == 1
            handler._handle_key("r")
        elif phase == "ready_wait":
            handler._handle_key("t")
        elif phase == "complete":
            assert dataset.num_episodes == 2 and dataset.flush.call_count == 2
            handler._handle_key("ESC")

    monkeypatch.setattr(recording, "record_loop", episode)
    monkeypatch.setattr(control.time, "sleep", operator)
    rlt_record(cfg)
    assert calls == [0, 1]
    assert operations == [("save", "failure"), ("flush", None), ("save", "success"), ("flush", None)]
    assert cfg.dataset.video_encoding_batch_size == 1
    robot.disconnect.assert_called_once()
    client.stop.assert_called_once()
    listener.stop.assert_called_once()
