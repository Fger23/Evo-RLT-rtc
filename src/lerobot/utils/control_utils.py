# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################


import logging
import os
import select
import sys
import threading
import time
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import Any

try:
    import termios
    import tty
except ImportError:  # Native Windows has neither POSIX terminal module.
    termios = None
    tty = None

import numpy as np
import torch
from deepdiff import DeepDiff

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.robots import Robot
from lerobot.utils.recording_annotations import EPISODE_FAILURE, EPISODE_SUCCESS

# Minimum interval (seconds) between consecutive intervention toggle presses.
INTERVENTION_TOGGLE_COOLDOWN_S = 0.5


@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception as e:
        logging.info("pynput unavailable; using headless controls instead: %s", e)
        return True


class _KeyboardEventHandler:
    """Shared recording controls for GUI, POSIX terminal and Windows console input."""

    def __init__(
        self,
        events: dict[str, Any],
        intervention_toggle_key: str,
        episode_success_key: str | None,
        episode_failure_key: str | None,
    ):
        self.events = events
        self.intervention_toggle_key = intervention_toggle_key.lower()
        self.episode_success_key = episode_success_key.lower() if episode_success_key else None
        self.episode_failure_key = episode_failure_key.lower() if episode_failure_key else None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_intervention_time: float = 0.0

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)

    def _handle_key(self, key: str):
        normalized = key.lower() if len(key) == 1 else key
        if normalized == "ESC":
            print("Escape key pressed. Stopping data recording...")
            self.events["stop_recording"] = True
            self.events["exit_early"] = True
            return

        # Only the RLT recorder sets this phase. Other recording entry points
        # retain their existing controls and automatic episode transitions.
        if "rlt_phase" in self.events:
            if self.events.get("stop_recording"):
                return
            phase = self.events["rlt_phase"]
            if phase != "recording":
                if phase == "failure_ack_wait" and normalized == self.episode_failure_key:
                    print("Failure acknowledged. Press 'r' to reset the robot.")
                    self.events["rlt_phase"] = "failed_wait"
                elif phase == "failed_wait" and normalized == "r":
                    print("'r' key pressed. Requesting robot reset...")
                    self.events["rlt_phase"] = "resetting"
                    self.events["rlt_reset_requested"] = True
                elif phase in {"success_wait", "ready_wait"} and normalized == "t":
                    print("'t' key pressed. Requesting the next episode...")
                    self.events["rlt_phase"] = "starting"
                    self.events["rlt_start_requested"] = True
                return
            if normalized in {"r", "t"}:
                return
            if normalized in {"RIGHT", "LEFT"} or (
                normalized != self.intervention_toggle_key
                and normalized in {self.episode_success_key, self.episode_failure_key}
            ):
                # Latch before notifying the recording loop: repeated or
                # conflicting keys cannot relabel an episode during its save.
                self.events["rlt_phase"] = "saving"

        if normalized == "RIGHT":
            print("Right arrow key pressed. Exiting loop...")
            self.events["exit_early"] = True
        elif normalized == "LEFT":
            print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
        elif normalized == self.intervention_toggle_key:
            now = time.monotonic()
            if now - self._last_intervention_time < INTERVENTION_TOGGLE_COOLDOWN_S:
                return
            self._last_intervention_time = now
            print(f"'{self.intervention_toggle_key}' key pressed. Toggling intervention mode...")
            self.events["toggle_intervention"] = True
        elif self.episode_success_key and normalized == self.episode_success_key:
            print(f"'{self.episode_success_key}' key pressed. Marking episode as success and exiting loop...")
            self.events["episode_outcome"] = EPISODE_SUCCESS
            self.events["exit_early"] = True
        elif self.episode_failure_key and normalized == self.episode_failure_key:
            print(f"'{self.episode_failure_key}' key pressed. Marking episode as failure and exiting loop...")
            self.events["episode_outcome"] = EPISODE_FAILURE
            self.events["exit_early"] = True


class TTYKeyboardListener(_KeyboardEventHandler):
    """Read control keys directly from a POSIX TTY, including SSH sessions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fd = sys.stdin.fileno()
        self._old_attrs = None

    def start(self):
        if termios is None or tty is None:
            raise RuntimeError("POSIX terminal controls are unavailable on this platform.")
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, name="tty-keyboard-listener", daemon=True)
        self._thread.start()

    def stop(self):
        super().stop()
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def _run(self):
        while not self._stop_event.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                key = self._read_key()
                if key is not None:
                    self._handle_key(key)
            except Exception as e:
                logging.warning("TTY keyboard listener stopped after read error: %s", e)
                self._stop_event.set()

    def _read_key(self) -> str | None:
        chunk = os.read(self._fd, 1)
        if not chunk:
            return None
        if chunk == b"\x1b":
            sequence = bytearray(chunk)
            while True:
                ready, _, _ = select.select([self._fd], [], [], 0.01)
                if not ready:
                    break
                sequence.extend(os.read(self._fd, 1))
                last_byte = bytes(sequence[-1:])
                if len(sequence) >= 3 and last_byte in {b"A", b"B", b"C", b"D", b"~"}:
                    break
            sequence_bytes = bytes(sequence)
            if sequence_bytes in {b"\x1b[C", b"\x1bOC"}:
                return "RIGHT"
            if sequence_bytes in {b"\x1b[D", b"\x1bOD"}:
                return "LEFT"
            if sequence_bytes == b"\x1b":
                return "ESC"
            return None
        return chunk.decode("utf-8", errors="ignore")


class WindowsConsoleKeyboardListener(_KeyboardEventHandler):
    """Nonblocking native console controls when pynput is unavailable on Windows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._console = None
        self._extended_key = False

    def start(self):
        import msvcrt

        self._console = msvcrt
        self._thread = threading.Thread(target=self._run, name="windows-keyboard-listener", daemon=True)
        self._thread.start()

    def _read_key(self) -> str | None:
        if not self._console.kbhit():
            return None
        key = self._console.getwch()
        if self._extended_key:
            self._extended_key = False
            return {"K": "LEFT", "M": "RIGHT"}.get(key)
        if key in ("\x00", "\xe0"):
            # Extended keys are two console characters. The second character
            # may arrive in a later poll; never block getwch while stopping.
            self._extended_key = True
            return self._read_key()
        return "ESC" if key == "\x1b" else key

    def _run(self):
        while not self._stop_event.is_set():
            try:
                key = self._read_key()
                if key is not None:
                    self._handle_key(key)
                else:
                    self._stop_event.wait(0.01)
            except Exception as e:
                logging.warning("Windows console keyboard listener stopped after read error: %s", e)
                self._stop_event.set()


def _start_pynput_listener(events, intervention_toggle_key, episode_success_key, episode_failure_key):
    from pynput import keyboard

    handler = _KeyboardEventHandler(events, intervention_toggle_key, episode_success_key, episode_failure_key)

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                handler._handle_key("RIGHT")
            elif key == keyboard.Key.left:
                handler._handle_key("LEFT")
            elif key == keyboard.Key.esc:
                handler._handle_key("ESC")
            elif getattr(key, "char", None):
                handler._handle_key(key.char)
        except Exception as e:
            logging.warning("Error handling keyboard input: %s", e)

    listener = keyboard.Listener(on_press=on_press)
    try:
        listener.start()
    except Exception:
        listener.stop()
        raise
    return listener


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def init_keyboard_listener(
    intervention_toggle_key: str = "i",
    episode_success_key: str | None = None,
    episode_failure_key: str | None = None,
):
    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (right arrow, left arrow, escape, intervention
    toggle key, and optional episode success/failure keys) to control
    the program flow during execution, such as stopping recording or exiting loops. It gracefully
    handles headless environments where keyboard listening is not possible.

    Returns:
        A tuple containing:
        - A GUI/console keyboard listener, or `None` when no input backend is available.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["toggle_intervention"] = False
    events["episode_outcome"] = None

    listener = None
    if not is_headless():
        try:
            listener = _start_pynput_listener(
                events, intervention_toggle_key, episode_success_key, episode_failure_key
            )
            return listener, events
        except Exception as e:
            logging.warning("GUI keyboard listener unavailable; trying console controls: %s", e)

    if sys.stdin.isatty():
        listener_type = WindowsConsoleKeyboardListener if sys.platform == "win32" else TTYKeyboardListener
        listener = listener_type(
            events=events,
            intervention_toggle_key=intervention_toggle_key,
            episode_success_key=episode_success_key,
            episode_failure_key=episode_failure_key,
        )
        try:
            listener.start()
        except Exception:
            listener.stop()
            raise
        logging.warning(
            "Using %s over the current interactive console; focus this terminal for recording controls.",
            listener_type.__name__,
        )
        return listener, events

    logging.warning(
        "Headless environment detected without an interactive TTY. On-screen cameras display and keyboard inputs will not be available."
    )

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )


def sanity_check_bimanual_piper_pair(robot_cfg, teleop_cfg) -> None:
    """Ensure bimanual PiPER configs are not mixed between PiPER and PiPER-X variants."""
    if teleop_cfg is None:
        return

    robot_type = getattr(robot_cfg, "type", None)
    teleop_type = getattr(teleop_cfg, "type", None)
    expected_teleop_by_robot = {
        "bi_piper_follower": "bi_piper_leader",
        "bi_piperx_follower": "bi_piperx_leader",
    }
    expected_robot_by_teleop = {teleop: robot for robot, teleop in expected_teleop_by_robot.items()}

    if robot_type in expected_teleop_by_robot and teleop_type != expected_teleop_by_robot[robot_type]:
        expected = expected_teleop_by_robot[robot_type]
        raise ValueError(
            f"In bimanual PiPER mode, '{robot_type}' must be paired with '{expected}', got '{teleop_type}'."
        )
    if teleop_type in expected_robot_by_teleop and robot_type != expected_robot_by_teleop[teleop_type]:
        expected = expected_robot_by_teleop[teleop_type]
        raise ValueError(
            f"In bimanual PiPER mode, '{teleop_type}' must be paired with '{expected}', got '{robot_type}'."
        )
