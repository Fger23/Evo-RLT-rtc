# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Adapt learned-RTC execution to the existing episodic/HIL recorder.

The recorder owns robot connection and dispatch. The adapter owns only the
protocol-v2 request and aligned raw/processed queues. A queued command advances
only after the recorder confirms what the robot accepted.
"""

import threading

from lerobot.scripts.lerobot_infer_trc import (
    TrainedRTCInferenceClient,
    TrainedRTCInferenceConfig,
    _AlignedActionQueue,
    _validate_sent_action,
)
from lerobot.scripts.recording_remote_policy import RemotePolicyRecordConfig
from lerobot.transport import services_pb2


class TrainedRTCRecordingClient(TrainedRTCInferenceClient):
    def __init__(self, cfg: RemotePolicyRecordConfig, robot, fps: int, task: str):
        if not cfg.enable or not cfg.training_time_rtc:
            raise ValueError("TrainedRTCRecordingClient requires enabled training_time_rtc recording.")
        inference_cfg = TrainedRTCInferenceConfig(
            robot=robot.config,
            pretrained_name_or_path=cfg.pretrained_name_or_path,
            policy_type=cfg.policy_type,
            task=task,
            server_address=cfg.server_address,
            policy_device=cfg.policy_device,
            actions_per_chunk=cfg.actions_per_chunk,
            rtc_prefix_steps=cfg.rtc_prefix_steps,
            refill_threshold=cfg.chunk_size_threshold,
            fps=fps,
            rpc_timeout_s=cfg.rpc_timeout_s,
            setup_timeout_s=cfg.setup_timeout_s,
            queue_wait_timeout_s=cfg.queue_wait_timeout_s,
            max_consecutive_late_chunks=cfg.max_consecutive_late_chunks,
            rename_map=cfg.rename_map,
        )
        super().__init__(inference_cfg, robot=robot)
        self._action_in_hand = None
        self._suspended = False

    @property
    def policy_id(self) -> str:
        return self.cfg.pretrained_name_or_path

    def suspend(self) -> None:
        """Immediately relinquish control; drain the old RPC only on release."""
        with self._state_lock:
            self._suspended = True
            self._action_in_hand = None
            self._queue = _AlignedActionQueue()

    def reset(self) -> None:
        """Drain the previous generation before resetting server and local state.

        There can be no new request while the recorder resets, so joining the
        worker and then clearing its pending result prevents cross-episode or
        pre-intervention actions from reaching the robot.
        """
        self.suspend()
        with self._state_lock:
            request_thread = self._request_thread
        if request_thread is not None:
            request_thread.join(timeout=2 * self.cfg.rpc_timeout_s + 1.0)
            if request_thread.is_alive():
                raise TimeoutError("Cannot reset RTC while the previous inference request is still running.")
        with self._state_lock:
            self._request_thread = None
            self._pending_result = None
            self._pending_error = None
            self._pending_event.clear()
            self._executed_steps = 0
            self.consecutive_late_chunks = 0
        self.stub.Ready(services_pb2.Empty(), timeout=self.cfg.rpc_timeout_s)
        self._suspended = False

    def get_action(self, observation: dict, task: str | None, timestep: int) -> dict[str, float]:
        if not self._started or self._suspended:
            raise RuntimeError("RTC recording client is not active; start/reset it before taking actions.")
        if task != self.cfg.task:
            raise ValueError("The recorded task must match the RTC policy request task.")
        if self._action_in_hand is not None:
            raise RuntimeError("Confirm the previous robot command before requesting another RTC action.")
        # timestep is dataset provenance; RTC delay counts accepted policy
        # commands only and restarts after each human intervention.
        self._apply_pending_result()
        if not self._queue:
            self._start_request(observation)
            self._wait_until_actions(count_underrun=self._executed_steps > 0)
        if self._should_refill():
            self._start_request(observation)
        self._action_in_hand = self._peek_action_dict()
        return dict(self._action_in_hand)

    def confirm_action_executed(self, sent_action: dict[str, float]) -> None:
        if self._action_in_hand is None:
            raise RuntimeError("No RTC command is awaiting hardware confirmation.")
        _validate_sent_action(self._action_in_hand, sent_action)
        self._confirm_action_executed()
        self._action_in_hand = None

    def stop(self) -> None:
        # The outer recorder closes the robot after shutting down its executors.
        self._stop_event.set()
        self.channel.close()
        with self._state_lock:
            request_thread = self._request_thread
        if request_thread is not None and request_thread is not threading.current_thread():
            request_thread.join(timeout=self.cfg.rpc_timeout_s + 1.0)
        self._started = False
