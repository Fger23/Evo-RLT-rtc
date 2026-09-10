"""Manual episode boundaries and an operator-triggered return to the session start pose."""

import logging
import math
import time

from lerobot.cameras.camera import CameraFrameTimeoutError


def read_observation_with_camera_recovery(
    robot, remote_policy_client, teleop, events, timeout_s=2.0, *, initial_error=None
):
    """Pause on frame starvation, then restart RTC using post-reset camera data.

    Return None when the operator stops the episode. Only CameraFrameTimeoutError
    enters recovery; motor, disconnected-camera and other failures propagate.
    No synthetic frames or policy commands are produced during recovery.
    """

    def stopped():
        return events.get("exit_early", False) or events.get("stop_recording", False)

    if stopped():
        return None
    error = initial_error
    if error is None:
        try:
            observation = robot.get_observation()
            return None if stopped() else observation
        except CameraFrameTimeoutError as caught:
            error = caught
    if stopped():
        return None
    if timeout_s <= 0:
        raise error
    deadline = time.perf_counter() + timeout_s
    logging.warning(
        "RLT_CAMERA_WAIT: %s. Holding pose and retrying fresh frames for %.1fs.", error, timeout_s
    )
    remote_policy_client.suspend()
    RLTManualEpisodeController.hold_current_pose(robot, teleop)
    while time.perf_counter() < deadline:
        if stopped():
            return None
        try:
            # Probe availability only. This snapshot must not reach the policy:
            # reset() can take time while it drains an earlier in-flight RPC.
            robot.get_observation()
        except CameraFrameTimeoutError as caught:
            error = caught
            if stopped():
                return None
            time.sleep(min(0.05, max(0.0, deadline - time.perf_counter())))
            continue
        if stopped():
            return None
        try:
            remote_policy_client.reset()
        except Exception:
            if stopped():
                return None
            raise
        if stopped():
            return None
        try:
            observation = robot.get_observation()
        except CameraFrameTimeoutError as caught:
            error = caught
            remote_policy_client.suspend()
            continue
        if stopped():
            return None
        logging.info("RLT_CAMERA_RECOVERED: fresh observation acquired; old RTC actions discarded.")
        return observation
    if stopped():
        return None
    raise CameraFrameTimeoutError(
        f"Camera did not recover within the {timeout_s:.1f}s retry window: {error}"
    ) from error


def read_joint_positions(robot):
    """Read follower joints directly; paused/reset control must not depend on cameras."""
    if hasattr(robot, "left_arm") and hasattr(robot, "right_arm"):
        positions = {}
        for prefix, arm in (("left_", robot.left_arm), ("right_", robot.right_arm)):
            positions.update(
                {
                    f"{prefix}{key}.pos": float(value)
                    for key, value in arm.bus.sync_read("Present_Position").items()
                }
            )
    elif hasattr(robot, "bus"):
        positions = {
            f"{key}.pos": float(value) for key, value in robot.bus.sync_read("Present_Position").items()
        }
    else:
        raise ValueError("RLT keyboard reset requires a follower robot with readable joint buses")
    expected = set(robot.action_features)
    if set(positions) != expected or not all(math.isfinite(value) for value in positions.values()):
        raise ValueError("Cannot capture/reset RLT pose: joint names or values do not match the robot")
    return positions


class RLTManualEpisodeController:
    def __init__(self, reset_duration_s=5.0):
        if not math.isfinite(reset_duration_s) or reset_duration_s <= 0:
            raise ValueError("reset_duration_s must be finite and positive")
        self.reset_duration_s = reset_duration_s
        self.start_pose = None

    @staticmethod
    def hold_current_pose(robot, teleop):
        # Stop a last policy target that may still be in transit, instead of
        # merely retaining its (possibly distant) Goal_Position.
        pose = read_joint_positions(robot)
        robot.send_action(pose)
        if teleop is not None and hasattr(teleop, "send_feedback"):
            teleop.send_feedback(pose)

    def on_record_connected(self, robot, teleop):
        self.start_pose = read_joint_positions(robot)
        logging.info("RLT reset pose captured from BOTH arms at this command's startup: %s", self.start_pose)

    @staticmethod
    def _enter_wait(events, phase):
        events["exit_early"] = False
        events["rerecord_episode"] = False
        events["toggle_intervention"] = False
        events["rlt_reset_requested"] = False
        events["rlt_start_requested"] = False
        # Publish the phase last: keys pressed during save/reset cannot queue a future start.
        events["rlt_phase"] = phase

    def reset(self, robot, teleop, events):
        if self.start_pose is None:
            raise RuntimeError("No session start pose was captured")
        if events["stop_recording"]:
            return
        current = read_joint_positions(robot)
        if teleop is not None and hasattr(teleop, "set_manual_control"):
            teleop.set_manual_control(False)
        steps = max(1, math.ceil(self.reset_duration_s * 20))
        for index in range(1, steps + 1):
            if events["stop_recording"]:
                self.hold_current_pose(robot, teleop)
                return
            alpha = index / steps
            action = {key: value + (self.start_pose[key] - value) * alpha for key, value in current.items()}
            robot.send_action(action)
            if teleop is not None and hasattr(teleop, "send_feedback"):
                teleop.send_feedback(action)
            time.sleep(self.reset_duration_s / steps)
        # Use follower feedback before enabling the next inference episode.
        deadline = time.monotonic() + 3.0
        while True:
            if events["stop_recording"]:
                self.hold_current_pose(robot, teleop)
                return
            measured = read_joint_positions(robot)
            error = max(abs(measured[key] - target) for key, target in self.start_pose.items())
            if error <= 2.0:
                return
            if time.monotonic() >= deadline:
                self.hold_current_pose(robot, teleop)
                raise RuntimeError(
                    f"RLT reset did not reach the start pose (max joint error={error:.2f}); stopped"
                )
            time.sleep(0.05)

    def wait_for_start(self, robot, teleop, events, *, has_more=True):
        messages = {
            "ready_wait": "RLT_READY: press t to start inference; Esc to exit. r is disabled.",
            "success_wait": "RLT_SUCCESS_SAVED: holding current pose. Press t to continue from here; Esc to exit.",
            "failed_wait": "RLT_FAILURE_SAVED: holding current pose. Press r to reset; t is disabled until reset.",
            "failure_ack_wait": "RLT_FAILURE_SAVED: timeout/early end. Press f to enable reset, then r; Esc to exit.",
            "complete": "RLT_COMPLETE: target reached. Holding pose; press Esc to exit. No more inference.",
        }
        previous_phase = None
        while not events["stop_recording"]:
            phase = events["rlt_phase"]
            if phase != previous_phase and phase in messages:
                logging.info(messages[phase])
                previous_phase = phase
            if events.get("rlt_reset_requested"):
                events["rlt_reset_requested"] = False
                logging.info("RLT_RESETTING: returning to this session's start pose; Esc interrupts reset.")
                self.reset(robot, teleop, events)
                if events["stop_recording"]:
                    break
                logging.info("RLT_RESET_DONE: joint feedback confirms the session start pose.")
                self._enter_wait(events, "ready_wait" if has_more else "complete")
            elif events.get("rlt_start_requested") and has_more:
                events["rlt_start_requested"] = False
                events["episode_outcome"] = None
                events["exit_early"] = False
                events["rlt_phase"] = "recording"
                return
            else:
                time.sleep(0.02)

    def start_session(self, robot, teleop, events):
        self._enter_wait(events, "ready_wait")
        self.wait_for_start(robot, teleop, events)

    def finish_episode(self, robot, teleop, remote_policy_client, dataset, events, outcome, *, has_more):
        explicit_outcome = events.get("episode_outcome")
        events["rlt_phase"] = "saving"
        remote_policy_client.suspend()
        hold_error = None
        try:
            self.hold_current_pose(robot, teleop)
        except Exception as error:
            # Preserve the episode even if the hardware rejects the hold;
            # reset/start must remain disabled after such a failure.
            hold_error = error
        # Empty episodes cannot form valid training samples. A discarded/empty
        # attempt requires a new t as well, instead of restarting automatically.
        discarded = events["rerecord_episode"] or dataset.episode_buffer["size"] == 0
        if discarded:
            dataset.clear_episode_buffer()
            logging.info("RLT_DISCARDED: no episode saved. Press t to try again.")
            phase = "ready_wait"
        else:
            logging.info("RLT_SAVING: policy commands stopped; writing episode data and videos. Please wait.")
            dataset.save_episode(extra_episode_metadata={"episode_success": outcome})
            dataset.flush()
            logging.info(
                "RLT_SAVED: episode=%s label=%s total=%s",
                dataset.num_episodes - 1,
                outcome,
                dataset.num_episodes,
            )
            if outcome == "failure":
                phase = "failed_wait" if explicit_outcome == "failure" else "failure_ack_wait"
            else:
                phase = "success_wait" if has_more else "complete"
        if hold_error is not None:
            raise RuntimeError("RLT stopped: could not hold the current joint positions") from hold_error
        self._enter_wait(events, phase)
        self.wait_for_start(robot, teleop, events, has_more=has_more or discarded)
        return not discarded
