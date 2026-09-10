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
from unittest.mock import patch

import av
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_rlt_round as jobs
from lerobot.scripts.lerobot_rlt_verify import CAMERAS, SEAL, verify_dataset, verify_transfer

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
