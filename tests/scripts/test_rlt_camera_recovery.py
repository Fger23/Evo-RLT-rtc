"""Camera-only recovery uses measured holds and never manufactures training frames."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lerobot.cameras.camera import CameraFrameTimeoutError
from lerobot.scripts import recording_rlt_control as control


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.on_sleep = None

    def monotonic(self):
        return self.now

    perf_counter = monotonic

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


class CameraRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.time_patch = patch.object(control, "time", self.clock)
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.events = {
            "rlt_phase": "recording",
            "exit_early": False,
            "stop_recording": False,
            "episode_outcome": None,
            "rerecord_episode": False,
        }
        self.operations = []
        self.robot = SimpleNamespace(
            bus=SimpleNamespace(sync_read=Mock(return_value={"joint": 12.5})),
            action_features={"joint.pos": float},
            get_observation=Mock(),
            send_action=Mock(side_effect=lambda action: self.operations.append(("hold", action))),
        )
        self.client = SimpleNamespace(
            suspend=Mock(side_effect=lambda: self.operations.append(("suspend", None))),
            reset=Mock(side_effect=lambda: self.operations.append(("reset", None))),
            get_action=Mock(side_effect=AssertionError("Recovery must not infer robot actions")),
        )
        self.teleop = SimpleNamespace(send_feedback=Mock())

    def run_recovery(self, timeout_s=2.0, **kwargs):
        return control.read_observation_with_camera_recovery(
            self.robot, self.client, self.teleop, self.events, timeout_s=timeout_s, **kwargs
        )

    def press(self, key):
        self.events["exit_early"] = True
        self.events["episode_outcome"] = {"s": "success", "f": "failure", "ESC": None}[key]
        self.events["stop_recording"] = key == "ESC"

    def camera_timeout(self):
        self.clock.now += 0.2  # Simulate one camera async_read timeout.
        raise CameraFrameTimeoutError("wrist camera has no new frame")

    def assert_only_measured_hold(self):
        self.robot.bus.sync_read.assert_called_once_with("Present_Position")
        self.robot.send_action.assert_called_once_with({"joint.pos": 12.5})
        self.teleop.send_feedback.assert_called_once_with({"joint.pos": 12.5})
        self.client.get_action.assert_not_called()

    def test_healthy_observation_is_returned_without_recovery_or_joint_commands(self):
        observation = {"joint.pos": 2.0, "wrist": object()}
        self.robot.get_observation.return_value = observation

        self.assertIs(self.run_recovery(), observation)

        self.robot.get_observation.assert_called_once_with()
        self.robot.bus.sync_read.assert_not_called()
        self.robot.send_action.assert_not_called()
        self.client.suspend.assert_not_called()
        self.client.reset.assert_not_called()

    def test_recovery_suspends_holds_and_discards_the_pre_reset_probe(self):
        probe = {"joint.pos": 12.5, "wrist": object()}
        fresh = {"joint.pos": 12.5, "wrist": object()}
        sequence = iter([CameraFrameTimeoutError("camera stalled"), probe, fresh])

        def read():
            self.operations.append(("observation", None))
            value = next(sequence)
            if isinstance(value, Exception):
                raise value
            return value

        self.robot.get_observation.side_effect = read

        self.assertIs(self.run_recovery(), fresh)

        self.assertEqual(
            self.operations,
            [
                ("observation", None),
                ("suspend", None),
                ("hold", {"joint.pos": 12.5}),
                ("observation", None),
                ("reset", None),
                ("observation", None),
            ],
        )
        self.assert_only_measured_hold()

    def test_camera_timeout_already_seen_by_rtc_forces_recovery_before_reading(self):
        probe = {"joint.pos": 12.5, "wrist": object()}
        fresh = {"joint.pos": 12.5, "wrist": object()}
        observations = iter((probe, fresh))

        def read():
            self.operations.append(("observation", None))
            return next(observations)

        self.robot.get_observation.side_effect = read

        self.assertIs(
            self.run_recovery(initial_error=CameraFrameTimeoutError("RTC recapture timed out")), fresh
        )

        self.assertEqual(
            self.operations,
            [
                ("suspend", None),
                ("hold", {"joint.pos": 12.5}),
                ("observation", None),
                ("reset", None),
                ("observation", None),
            ],
        )
        self.assert_only_measured_hold()

    def test_repeated_camera_timeouts_raise_within_the_bounded_window(self):
        self.robot.get_observation.side_effect = self.camera_timeout

        with self.assertRaises(CameraFrameTimeoutError):
            self.run_recovery(timeout_s=2.0)

        self.assertGreater(self.robot.get_observation.call_count, 1)
        self.assertLessEqual(self.robot.get_observation.call_count, 12)
        self.assertLessEqual(self.clock.now, 2.5)  # Includes the first timed-out read.
        self.client.reset.assert_not_called()
        self.assert_only_measured_hold()

    def test_a_pending_label_or_escape_skips_reads_and_recovery(self):
        for key in ("s", "f", "ESC"):
            with self.subTest(key=key):
                self.press(key)
                self.assertIsNone(self.run_recovery())
        self.robot.get_observation.assert_not_called()
        self.client.suspend.assert_not_called()
        self.client.reset.assert_not_called()
        self.robot.send_action.assert_not_called()

    def test_stop_during_initial_camera_timeout_returns_for_episode_save(self):
        for key in ("s", "f", "ESC"):
            with self.subTest(key=key):
                self.events["exit_early"] = False
                self.events["stop_recording"] = False

                def interrupted_read(key=key):
                    self.press(key)
                    raise CameraFrameTimeoutError("camera stalled while the key was pressed")

                self.robot.get_observation.side_effect = interrupted_read
                self.assertIsNone(self.run_recovery())
                self.assertEqual(
                    self.events["episode_outcome"], {"s": "success", "f": "failure", "ESC": None}[key]
                )

        self.client.reset.assert_not_called()
        self.robot.send_action.assert_not_called()

    def test_failure_during_a_retry_preserves_the_label_and_never_resets_rtc(self):
        calls = 0

        def read():
            nonlocal calls
            calls += 1
            if calls == 2:
                self.press("f")
            return self.camera_timeout()

        self.robot.get_observation.side_effect = read

        self.assertIsNone(self.run_recovery())

        self.assertEqual(calls, 2)
        self.assertEqual(self.events["episode_outcome"], "failure")
        self.client.reset.assert_not_called()
        self.assert_only_measured_hold()

    def test_escape_during_retry_wait_stops_without_another_camera_read(self):
        self.robot.get_observation.side_effect = self.camera_timeout
        self.clock.on_sleep = lambda: self.press("ESC")

        self.assertIsNone(self.run_recovery())

        self.assertTrue(self.events["stop_recording"])
        self.assertLessEqual(self.robot.get_observation.call_count, 2)
        self.client.reset.assert_not_called()
        self.assert_only_measured_hold()

    def test_real_motor_faults_are_never_retried_as_camera_timeouts(self):
        for error in (TimeoutError("motor status timeout"), RuntimeError("motor overload")):
            with self.subTest(error=type(error).__name__):
                self.robot.get_observation.reset_mock(side_effect=True)
                self.robot.get_observation.side_effect = error
                with self.assertRaises(type(error)) as raised:
                    self.run_recovery()
                self.assertIs(raised.exception, error)
                self.robot.get_observation.assert_called_once_with()
        self.client.suspend.assert_not_called()
        self.client.reset.assert_not_called()
        self.robot.send_action.assert_not_called()

    def test_motor_fault_during_recovery_is_propagated_without_more_retries(self):
        error = RuntimeError("motor overload during camera recovery")
        self.robot.get_observation.side_effect = [CameraFrameTimeoutError("camera stalled"), error]

        with self.assertRaises(RuntimeError) as raised:
            self.run_recovery()

        self.assertIs(raised.exception, error)
        self.assertEqual(self.robot.get_observation.call_count, 2)
        self.client.reset.assert_not_called()
        self.assert_only_measured_hold()

    def test_failed_measured_hold_is_not_ignored_or_followed_by_camera_retry(self):
        error = RuntimeError("motor rejected measured-position hold")
        self.robot.send_action.side_effect = error
        self.robot.get_observation.side_effect = CameraFrameTimeoutError("camera stalled")

        with self.assertRaises(RuntimeError) as raised:
            self.run_recovery()

        self.assertIs(raised.exception, error)
        self.robot.get_observation.assert_called_once_with()
        self.client.reset.assert_not_called()

    def test_rtc_reset_error_without_a_stop_key_is_not_swallowed(self):
        error = TimeoutError("RPC reset did not finish")
        self.robot.get_observation.side_effect = [CameraFrameTimeoutError("camera stalled"), {"wrist": object()}]
        self.client.reset.side_effect = error

        with self.assertRaises(TimeoutError) as raised:
            self.run_recovery()

        self.assertIs(raised.exception, error)
        self.assertEqual(self.robot.get_observation.call_count, 2)
        self.assert_only_measured_hold()

    def test_key_during_rtc_reset_skips_the_final_observation(self):
        for fail_reset in (False, True):
            with self.subTest(fail_reset=fail_reset):
                self.events["exit_early"] = False
                self.events["stop_recording"] = False
                self.robot.get_observation.reset_mock(side_effect=True)
                self.robot.get_observation.side_effect = [
                    CameraFrameTimeoutError("camera stalled"),
                    {"wrist": object()},
                ]

                def reset(fail_reset=fail_reset):
                    self.press("f")
                    if fail_reset:
                        raise TimeoutError("RPC reset failed after stop")

                self.client.reset.side_effect = reset
                self.assertIsNone(self.run_recovery())
                self.assertEqual(self.robot.get_observation.call_count, 2)
                self.assertEqual(self.events["episode_outcome"], "failure")

    def test_key_during_post_reset_fresh_read_prevents_observation_reuse(self):
        calls = 0

        def read():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise CameraFrameTimeoutError("camera stalled")
            if calls == 3:
                self.press("s")
            return {"wrist": object()}

        self.robot.get_observation.side_effect = read

        self.assertIsNone(self.run_recovery())

        self.assertEqual(calls, 3)
        self.assertEqual(self.events["episode_outcome"], "success")
        self.assert_only_measured_hold()


if __name__ == "__main__":
    unittest.main()
