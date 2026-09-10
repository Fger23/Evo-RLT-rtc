# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

import pickle
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("grpc")

from lerobot.scripts.lerobot_infer_trc import TrainedRTCInferenceConfig, _AlignedActionQueue
from lerobot.scripts.recording_remote_policy import RemotePolicyRecordConfig
from lerobot.scripts.recording_trained_rtc import TrainedRTCRecordingClient


def _client(policy_type="pi05"):
    client = object.__new__(TrainedRTCRecordingClient)
    client.cfg = TrainedRTCInferenceConfig(
        robot=SimpleNamespace(),
        pretrained_name_or_path="round_0/pretrained_model",
        policy_type=policy_type,
        task="Count the banknotes",
        actions_per_chunk=5,
        rtc_prefix_steps=2,
        refill_threshold=0.5,
        rpc_timeout_s=0.1,
    )
    client.robot = SimpleNamespace(is_connected=True, action_features={"joint.pos": float})
    client.stub = SimpleNamespace(Ready=Mock(), SendPolicyInstructions=Mock())
    client.channel = SimpleNamespace(close=Mock())
    client._state_lock = threading.RLock()
    client._pending_event = threading.Event()
    client._stop_event = threading.Event()
    client._request_thread = None
    client._pending_result = None
    client._pending_error = None
    client._request_sequence = 8
    client._executed_steps = 0
    client._queue = _AlignedActionQueue()
    client._queue.replace(torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([[10.0], [20.0], [30.0]]))
    client._action_in_hand = None
    client._started = True
    client._suspended = False
    client.consecutive_late_chunks = 1
    return client


@pytest.mark.parametrize("policy_type", ["pi05", "rlt"])
def test_setup_uses_protocol_v2_for_both_initial_and_rlt_actor(monkeypatch, policy_type):
    client = _client(policy_type)
    client._started = False
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_infer_trc.map_robot_keys_to_lerobot_features", lambda robot: {}
    )
    client.start()
    setup = pickle.loads(client.stub.SendPolicyInstructions.call_args.args[0].data)
    assert setup.policy_type == policy_type
    assert setup.protocol_version == 2
    assert setup.training_time_rtc and setup.return_raw_actions and setup.acp_positive_prompt
    assert setup.use_cfg is False and setup.cfg_beta == 1.0
    assert setup.rtc_prefix_steps == 2


def test_queue_advances_only_after_hardware_confirmation():
    client = _client()
    action = client.get_action({}, client.cfg.task, timestep=17)
    assert action == {"joint.pos": 10.0}
    assert client._executed_steps == 0 and len(client._queue) == 3
    with pytest.raises(RuntimeError, match="Confirm the previous"):
        client.get_action({}, client.cfg.task, timestep=18)
    with pytest.raises(RuntimeError, match="Robot modified"):
        client.confirm_action_executed({"joint.pos": 9.0})
    assert client._executed_steps == 0
    client.confirm_action_executed(action)
    assert client._executed_steps == 1 and len(client._queue) == 2
    torch.testing.assert_close(client._queue.prefix(2)[0], torch.tensor([[2.0], [3.0]]))


def test_intervention_suspends_immediately_and_reset_discards_inflight_result():
    client = _client()
    release_rpc = threading.Event()
    reset_done = threading.Event()

    def worker():
        assert release_rpc.wait(1.0)
        with client._state_lock:
            client._pending_result = ("stale_request", "stale_actions", 0.0)
            client._pending_error = ("stale_request", RuntimeError("stale failure"))
            client._pending_event.set()

    client._request_thread = threading.Thread(target=worker)
    client._request_thread.start()
    client.suspend()
    assert client._request_thread.is_alive() and not client._queue
    with pytest.raises(RuntimeError, match="not active"):
        client.get_action({}, client.cfg.task, 0)
    reset_thread = threading.Thread(target=lambda: (client.reset(), reset_done.set()))
    reset_thread.start()
    assert not reset_done.wait(0.02)
    release_rpc.set()
    reset_thread.join(1.0)
    assert reset_done.is_set()
    assert client._pending_result is None and client._pending_error is None
    assert not client._pending_event.is_set() and not client._suspended
    assert client._executed_steps == 0 and client.consecutive_late_chunks == 0
    assert client._request_sequence == 8  # Request IDs stay unique across episodes.
    client.stub.Ready.assert_called_once()


def test_stop_does_not_disconnect_the_recorder_owned_robot():
    client = _client()
    client.robot.disconnect = Mock()
    client.stop()
    client.channel.close.assert_called_once()
    client.robot.disconnect.assert_not_called()


def test_rlt_short_queue_drains_before_cold_start_instead_of_sending_untrained_delay():
    client = _client("rlt")
    client._queue.replace(torch.tensor([[1.0]]), torch.tensor([[10.0]]))
    assert client._start_request({"joint.pos": 0.0}) is False
    assert client._request_sequence == 8 and client._request_thread is None
    action = client.get_action({}, client.cfg.task, timestep=0)
    client.confirm_action_executed(action)
    assert not client._queue
    client._request_actions = Mock(return_value="cold_start_response")
    assert client._start_request({"joint.pos": 10.0})
    assert client._pending_event.wait(1.0)
    request = client._request_actions.call_args.args[0]
    assert request.conditioned_prefix_steps == 0 and request.action_prefix is None


@pytest.mark.parametrize("invalid", [{"policy_type": "act"}, {"aggregate_fn_name": "average"}])
def test_trained_recording_rejects_incompatible_transport_options(invalid):
    args = {
        "enable": True,
        "training_time_rtc": True,
        "policy_type": "rlt",
        "pretrained_name_or_path": "unused",
    }
    args.update(invalid)
    with pytest.raises(ValueError):
        RemotePolicyRecordConfig(**args)


def test_record_loop_stores_accepted_action_and_confirms_before_next_step(monkeypatch):
    from lerobot.scripts.recording_loop import record_loop

    events = {"exit_early": False}
    order = []
    frames = []

    def send_action(action):
        order.append("send")
        events["exit_early"] = True
        return {"joint.pos": 1.0}  # Simulate robot-side command clipping.

    feature = {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}
    robot = SimpleNamespace(
        action_features={"joint.pos": float},
        get_observation=lambda: {"joint.pos": 0.0},
        send_action=send_action,
    )
    dataset = SimpleNamespace(
        fps=30,
        features={
            "action": feature,
            "observation.state": feature,
            "complementary_info.policy_action": feature,
        },
        add_frame=lambda frame: (order.append("save"), frames.append(frame)),
    )
    client = SimpleNamespace(
        reset=lambda: None,
        get_action=lambda **kwargs: {"joint.pos": 2.0},
        confirm_action_executed=lambda action: (
            order.append("confirm"),
            np.testing.assert_equal(action["joint.pos"], 1),
        ),
    )
    monkeypatch.setattr("lerobot.scripts.recording_loop.precise_sleep", lambda _: None)
    record_loop(
        robot=robot,
        events=events,
        fps=30,
        teleop_action_processor=lambda x: x[0],
        robot_action_processor=lambda x: x[0],
        robot_observation_processor=lambda x: x,
        dataset=dataset,
        remote_policy_client=client,
        single_task="Count the banknotes",
        control_time_s=1,
    )
    assert order == ["send", "confirm", "save"]
    np.testing.assert_array_equal(frames[0]["action"], [1.0])
    np.testing.assert_array_equal(frames[0]["complementary_info.policy_action"], [2.0])


def test_rlt_rollout_requires_labels_and_preserves_policy_provenance(monkeypatch):
    import draccus

    from lerobot.scripts.lerobot_rlt_record import RLTRecordConfig, rlt_record

    cfg = draccus.parse(RLTRecordConfig, config_path="configs/rlt/windows_record.json", args=[])
    cfg.remote_policy.policy_type = "rlt"
    cfg.remote_policy.pretrained_name_or_path = "round_2"
    monkeypatch.setattr("lerobot.scripts.lerobot_rlt_record.record_to_target", lambda cfg, recorder: cfg)
    configured = rlt_record.__wrapped__(cfg)
    assert configured.enable_episode_outcome_labeling and configured.require_episode_success_label
    assert configured.default_episode_success == "failure"
    assert configured.enable_collector_policy_id and configured.collector_policy_id_policy == "round_2"
    assert not configured.policy_sync_to_teleop
    assert configured.dataset.video_encoding_batch_size == 1
    assert configured._rlt_episode_controller.reset_duration_s == 5.0


def test_reset_without_teleoperator_expires_without_sending_actions(monkeypatch):
    from lerobot.scripts.recording_loop import record_loop

    events = {"exit_early": False}
    clock = [0.0]
    observations = []

    def perf_counter():
        clock[0] += 0.01
        return clock[0]

    def observation():
        observations.append(0)
        # Guard the test against hanging if the no-action clock update regresses.
        if len(observations) == 10:
            events["exit_early"] = True
        return {"joint.pos": 0.0}

    robot = SimpleNamespace(
        action_features={"joint.pos": float}, get_observation=observation, send_action=Mock()
    )
    sleep = Mock()
    monkeypatch.setattr("lerobot.scripts.recording_loop.time.perf_counter", perf_counter)
    monkeypatch.setattr("lerobot.scripts.recording_loop.precise_sleep", sleep)
    record_loop(
        robot=robot,
        events=events,
        fps=30,
        teleop_action_processor=lambda x: x[0],
        robot_action_processor=lambda x: x[0],
        robot_observation_processor=lambda x: x,
        control_time_s=0.05,
    )
    assert 0 < len(observations) < 10
    assert sleep.call_count == len(observations)
    robot.send_action.assert_not_called()
