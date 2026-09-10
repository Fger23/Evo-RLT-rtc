"""Native recordings and failure cases for the documented Windows/H200 workflow."""

import argparse
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import av
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_rlt_round as jobs
from lerobot.scripts.lerobot_rlt_verify import CAMERAS, SEAL, verify_dataset, verify_transfer
from lerobot.scripts.recording_rlt_resume import record_to_target

POLICY = "/models/original_pi05"


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.source = Path(cls.shared.name) / "dianchao_0"
        features = {
            **{
                key: {"dtype": "float32", "shape": (12,), "names": [f"joint_{i}" for i in range(12)]}
                for key in ("action", "observation.state")
            },
            "complementary_info.collector_policy_id": {
                "dtype": "string",
                "shape": (1,),
                "names": ["collector_policy_id"],
            },
            **{
                key: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}
                for key in CAMERAS
            },
        }
        dataset = LeRobotDataset.create(
            "local/dianchao_0",
            fps=30,
            features=features,
            root=cls.source,
            image_writer_processes=0,
            image_writer_threads=2,
            video_backend="pyav",
            vcodec="h264",
        )
        try:
            for episode in range(2):
                for index in range(6):
                    dataset.add_frame(
                        {
                            "action": np.full(12, index, dtype=np.float32),
                            "observation.state": np.full(12, episode, dtype=np.float32),
                            "task": "Count banknotes",
                            "complementary_info.collector_policy_id": POLICY,
                            **{key: np.full((32, 32, 3), index * 20, dtype=np.uint8) for key in CAMERAS},
                        }
                    )
                dataset.save_episode(
                    extra_episode_metadata={"episode_success": "success" if episode else "failure"}
                )
            dataset.finalize()
        finally:
            dataset.stop_image_writer()
            dataset._close_writer()
            dataset.meta._close_writer()

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "dianchao_0"
        shutil.copytree(self.source, self.root)

    def seal(self):
        report = verify_dataset(self.root, 0, 2, POLICY)
        (self.root / SEAL).write_text(json.dumps(report), encoding="utf-8")
        return report

    def test_native_h264_dataset_and_transfer_pass(self):
        report = self.seal()
        self.assertEqual((report["frames"], report["success"], report["failure"]), (12, 1, 1))
        relocated = self.root.parent / "uploaded" / self.root.name
        shutil.copytree(self.root, relocated)
        self.assertEqual(verify_transfer(relocated, 0, 2, POLICY), report)

    def test_wrong_count_round_and_policy_fail(self):
        for round_number, episodes, policy in ((0, 20, POLICY), (1, 2, POLICY), (0, 2, "/models/wrong")):
            with self.assertRaises(ValueError):
                verify_dataset(self.root, round_number, episodes, policy)

    def test_missing_video_fails(self):
        next((self.root / "videos").rglob("*.mp4")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            verify_dataset(self.root, 0, 2, POLICY)

    def test_corrupt_video_fails(self):
        next((self.root / "videos").rglob("*.mp4")).write_bytes(b"broken video")
        with self.assertRaises(av.error.InvalidDataError):
            verify_dataset(self.root, 0, 2, POLICY)

    def test_modified_uploaded_data_fails_checksum(self):
        self.seal()
        path = next((self.root / "data").rglob("*.parquet"))
        with path.open("ab") as stream:
            stream.write(b"changed during upload")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            verify_transfer(self.root, 0, 2, POLICY)

    def test_missing_success_label_fails(self):
        import pyarrow.parquet as pq

        path = next((self.root / "meta/episodes").rglob("*.parquet"))
        table = pq.read_table(path).drop(["episode_success"])
        pq.write_table(table, path)
        with self.assertRaisesRegex(ValueError, "label"):
            verify_dataset(self.root, 0, 2, POLICY)

    def test_missing_data_frame_fails(self):
        import pyarrow.parquet as pq

        path = next((self.root / "data").rglob("*.parquet"))
        table = pq.read_table(path)
        pq.write_table(table.slice(1), path)
        with self.assertRaisesRegex(ValueError, "Frame count"):
            verify_dataset(self.root, 0, 2, POLICY)

    def recording_config(self, target=20):
        return SimpleNamespace(
            dataset=SimpleNamespace(
                root=self.root,
                repo_id="local/dianchao_0",
                num_episodes=target,
                fps=30,
                single_task="Count banknotes",
            ),
            remote_policy=SimpleNamespace(pretrained_name_or_path=POLICY),
            collector_policy_id_policy=None,
            resume=False,
        )

    def test_resume_appends_native_episode_and_reaches_total_without_overwrite(self):
        before = self.seal()
        cfg = self.recording_config(target=3)

        def append(recording_cfg):
            self.assertTrue(recording_cfg.resume)
            self.assertEqual(recording_cfg.dataset.num_episodes, 1)
            dataset = LeRobotDataset(cfg.dataset.repo_id, root=self.root, video_backend="pyav", vcodec="h264")
            try:
                for index in range(6):
                    dataset.add_frame(
                        {
                            "action": np.full(12, index, dtype=np.float32),
                            "observation.state": np.full(12, 2, dtype=np.float32),
                            "task": cfg.dataset.single_task,
                            "complementary_info.collector_policy_id": POLICY,
                            **{key: np.full((32, 32, 3), index * 20, dtype=np.uint8) for key in CAMERAS},
                        }
                    )
                dataset.save_episode(extra_episode_metadata={"episode_success": "failure"})
                dataset.finalize()
            finally:
                dataset.stop_image_writer()
                dataset._close_writer()
                dataset.meta._close_writer()

        record_to_target(cfg, append)
        self.assertEqual(cfg.dataset.num_episodes, 3)
        self.assertFalse(cfg.resume)
        self.assertFalse((self.root / SEAL).exists())
        after = verify_dataset(self.root, 0, 3, POLICY)
        self.assertEqual((after["frames"], after["success"], after["failure"]), (18, 1, 2))
        for name, digest in before["files"].items():
            if name.startswith("data/"):
                self.assertEqual(after["files"][name], digest)
        recorder = Mock()
        record_to_target(cfg, recorder)
        recorder.assert_not_called()

    def test_target_reached_or_exceeded_never_starts_recorder(self):
        self.seal()
        for target in (1, 2):
            recorder = Mock()
            record_to_target(self.recording_config(target), recorder)
            recorder.assert_not_called()
            self.assertTrue((self.root / SEAL).exists())

    def test_resume_rejects_bad_data_policy_or_task_before_recording(self):
        for field, value in (("policy", "/wrong/model"), ("task", "Bind banknotes")):
            cfg = self.recording_config()
            if field == "policy":
                cfg.remote_policy.pretrained_name_or_path = value
            else:
                cfg.dataset.single_task = value
            recorder = Mock()
            with self.assertRaises(ValueError):
                record_to_target(cfg, recorder)
            recorder.assert_not_called()
        next((self.root / "videos").rglob("*.mp4")).unlink()
        recorder = Mock()
        with self.assertRaisesRegex(ValueError, "Missing"):
            record_to_target(self.recording_config(), recorder)
        recorder.assert_not_called()

    def test_new_and_zero_episode_directories_start_fresh(self):
        cfg = self.recording_config()
        cfg.dataset.root = self.root.parent / "new_round"
        recorder = Mock()
        record_to_target(cfg, recorder)
        self.assertFalse(recorder.call_args.args[0].resume)
        self.assertEqual(recorder.call_args.args[0].dataset.num_episodes, 20)
        for with_metadata in (False, True):
            cfg.dataset.root.mkdir()
            if with_metadata:
                (cfg.dataset.root / "meta").mkdir()
                (cfg.dataset.root / "meta/info.json").write_text('{"total_episodes": 0}')
                (cfg.dataset.root / "unsaved-frame.png").write_bytes(b"preserve me")
            record_to_target(cfg, recorder)
            self.assertFalse(cfg.dataset.root.exists())
            self.assertFalse(recorder.call_args.args[0].resume)
        backups = list(self.root.parent.glob("new_round.empty-backup-*"))
        self.assertEqual(len(backups), 2)
        self.assertEqual(sum((path / "unsaved-frame.png").exists() for path in backups), 1)


class RoundJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.args = argparse.Namespace(
            round=0,
            gpu=5,
            episodes=20,
            token_steps=5000,
            ac_steps=5000,
            data_root=self.temporary.name + "/data",
            model_root=self.temporary.name + "/models",
        )

    def test_initial_and_later_plans_match_training_cli(self):
        from lerobot.scripts.lerobot_rlt_train import make_parser

        for number in (0, 1, 3):
            self.args.round = number
            plan = jobs.make_plan(self.args)
            self.assertTrue(plan["output"].endswith(f"round_{number + 1}"))
            self.assertEqual(len(plan["caches"]), number + 1)
            self.assertEqual(len(plan["specs"]), 2 if number == 0 else 1)
            for _, command in plan["commands"]:
                make_parser().parse_args(command[4:])
            ac = plan["commands"][-1][1]
            self.assertEqual("--resume" in ac, number > 0)
            self.assertEqual(any(stage == "token" for stage, _ in plan["commands"]), number == 0)

    def test_dry_run_does_not_create_job_or_launch_process(self):
        self.args.dry_run = True
        with patch.object(jobs.subprocess, "Popen") as launch, redirect_stdout(io.StringIO()):
            self.assertEqual(jobs.start(self.args), 0)
            launch.assert_not_called()
        self.assertFalse(Path(self.args.model_root).exists())

    def test_false_done_is_rejected(self):
        job = Path(self.args.model_root) / "jobs/from_round_0"
        job.mkdir(parents=True)
        jobs.write_json(job / "status.json", {"state": "DONE", "output": str(job / "absent")})
        self.args.tail, self.args.require_done = 0, True
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(jobs.status(self.args), 1)
        self.assertIn("FAILED", output.getvalue())

    @unittest.skipUnless(os.name == "posix", "The training worker runs on Ubuntu")
    def test_failed_training_writes_failed_not_done(self):
        plan = jobs.make_plan(self.args)
        job = Path(plan["job"])
        job.mkdir(parents=True)
        jobs.write_json(job / "request.json", vars(self.args))
        with (
            patch("lerobot.scripts.lerobot_rlt_verify.verify_transfer", return_value={"fps": 30}),
            patch.object(jobs, "run_stage", side_effect=RuntimeError("test training failure")),
        ):
            self.assertEqual(jobs.worker(job), 1)
        state = json.loads((job / "status.json").read_text())
        self.assertEqual(state["state"], "FAILED")
        self.assertEqual(state["stage"], "cache")
        self.assertIn("test training failure", state["error"])

    @unittest.skipUnless(os.name == "posix", "The training worker runs on Ubuntu")
    def test_successful_stages_require_final_ready_checkpoint(self):
        plan = jobs.make_plan(self.args)
        job = Path(plan["job"])
        job.mkdir(parents=True)
        jobs.write_json(job / "request.json", vars(self.args))

        def simulated_training(command):
            if command[4] != "ac":
                return
            output = Path(plan["output"])
            output.mkdir()
            jobs.write_json(
                output / "config.json",
                {
                    "type": "rlt",
                    "token_ready": True,
                    "actor_ready": True,
                    "chunk_size": 50,
                    "rtc_prefix_steps": 21,
                },
            )
            jobs.write_json(
                output / "training_manifest.json",
                {"caches": plan["caches"], "training_args": {"steps": 5000}},
            )
            for name in ("rlt_model.pt", "trainer_state.pt"):
                (output / name).write_bytes(b"test-only checkpoint placeholder")

        with (
            patch("lerobot.scripts.lerobot_rlt_verify.verify_transfer", return_value={"fps": 30}),
            patch.object(jobs, "run_stage", side_effect=simulated_training) as stage,
        ):
            self.assertEqual(jobs.worker(job), 0)
        self.assertEqual(stage.call_count, 3)
        self.assertEqual(json.loads((job / "status.json").read_text())["state"], "DONE")


if __name__ == "__main__":
    unittest.main()
