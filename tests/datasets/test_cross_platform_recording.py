"""Device-free native recording portability checks; runnable directly on Windows.

These exercise spawn workers, real H.264/PyAV files and relocation of a complete
dataset. They do not claim to execute on an Ubuntu host when run on Windows.
"""

import io
import json
import multiprocessing
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import torch

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation, TrainingTimeRTCMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class NoConcretePathsUnpickler(pickle.Unpickler):
    """Reject pathlib objects a different OS cannot instantiate, before deserializing."""

    def find_class(self, module, name):
        if module == "pathlib" and name in {"WindowsPath", "PosixPath"}:
            raise AssertionError(f"Platform-specific path leaked into transport: {module}.{name}")
        return super().find_class(module, name)


class CrossPlatformRecordingTests(unittest.TestCase):
    def test_remote_checkpoint_and_rtc_prefix_transport_contain_no_native_paths(self):
        linux_checkpoint = "/data/checkpoints/count_banknotes/round1"
        setup = RemotePolicyConfig(
            policy_type="rlt",
            pretrained_name_or_path=linux_checkpoint,
            lerobot_features={},
            actions_per_chunk=50,
            device="cuda",
            protocol_version=2,
            return_raw_actions=True,
            training_time_rtc=True,
            rtc_prefix_steps=21,
            acp_positive_prompt=True,
        )
        restored = NoConcretePathsUnpickler(io.BytesIO(pickle.dumps(setup))).load()
        self.assertIsInstance(restored.pretrained_name_or_path, str)
        self.assertEqual(restored.pretrained_name_or_path, linux_checkpoint)
        observation = TimedObservation(
            timestamp=0.0,
            timestep=0,
            observation={"image": np.zeros((8, 8, 3), dtype=np.uint8)},
            rtc_metadata=TrainingTimeRTCMetadata(
                request_id="request-1", action_prefix=torch.ones(21, 2), prefix_steps=21
            ),
        )
        restored = NoConcretePathsUnpickler(io.BytesIO(pickle.dumps(observation))).load()
        self.assertEqual(restored.rtc_metadata.action_prefix.device.type, "cpu")
        self.assertTrue(
            torch.equal(restored.rtc_metadata.action_prefix, observation.rtc_metadata.action_prefix)
        )

    def test_spawn_h264_recording_and_relocated_dataset_readback(self):
        cameras = ["observation.images.left", "observation.images.right"]
        features = {
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["left", "right"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["left", "right"]},
            **{
                key: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}
                for key in cameras
            },
        }
        with tempfile.TemporaryDirectory(prefix="lerobot-portability-") as temporary:
            temporary = Path(temporary).resolve()
            source, transferred = temporary / "windows_recording", temporary / "transferred_dataset"
            dataset = LeRobotDataset.create(
                "local/portability_test",
                fps=30,
                features=features,
                root=source,
                image_writer_processes=1,
                image_writer_threads=2,
                video_backend="pyav",
                vcodec="h264",
            )
            try:
                for episode in range(2):
                    for frame in range(4):
                        color = np.zeros((32, 32, 3), dtype=np.uint8)
                        color[..., episode] = 160
                        dataset.add_frame(
                            {
                                "observation.state": np.array([episode, frame], dtype=np.float32),
                                "action": np.full(2, 10 * episode + frame, dtype=np.float32),
                                "task": "Count the banknotes",
                                **dict.fromkeys(cameras, color),
                            }
                        )
                    dataset.save_episode(
                        parallel_encoding=True,
                        extra_episode_metadata={"episode_success": "success" if episode else "failure"},
                    )
                dataset.finalize()
            finally:
                dataset.stop_image_writer()
                dataset._close_writer()
                dataset.meta._close_writer()
            info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
            self.assertNotIn("\\", info["video_path"])
            self.assertNotIn("\\", info["data_path"])
            self.assertNotIn(str(source), json.dumps(info))
            for video in source.rglob("*.mp4"):
                with av.open(str(video)) as container:
                    self.assertEqual(container.streams.video[0].codec_context.name, "h264")
                    self.assertEqual(len(list(container.decode(video=0))), 8)
            shutil.copytree(source, transferred)
            hidden_source = temporary / "source_hidden"
            self.assertEqual(source.parent, temporary)
            self.assertEqual(hidden_source.parent, temporary)
            source.rename(hidden_source)
            reread = LeRobotDataset(
                "local/portability_test",
                root=transferred,
                video_backend="pyav",
                delta_timestamps={"action": [i / 30 for i in range(4)]},
                tolerance_s=0.001,
            )
            self.assertEqual(reread.num_episodes, 2)
            self.assertEqual(reread.meta.episodes[0]["episode_success"], "failure")
            self.assertEqual(reread.meta.episodes[1]["episode_success"], "success")
            # Exercise the actual LeRobot data reader without depending on the
            # installed torchvision's deprecated VideoReader API. PyAV video
            # decoding is checked separately below against the relocated files.
            end_indices, _ = reread._get_query_indices(3, 0)
            start_indices, _ = reread._get_query_indices(4, 1)
            end_of_first = reread._query_hf_dataset(end_indices)
            start_of_second = reread._query_hf_dataset(start_indices)
            self.assertEqual(end_of_first["action"][:, 0].tolist(), [3, 3, 3, 3])
            self.assertEqual(start_of_second["action"][:, 0].tolist(), [10, 11, 12, 13])
            for key in cameras:
                video_path = transferred / reread.meta.get_video_file_path(1, key)
                self.assertTrue(video_path.is_relative_to(transferred))
                with av.open(str(video_path)) as container:
                    self.assertEqual(float(container.streams.video[0].average_rate), 30)
                    decoded = list(container.decode(video=0))
                    self.assertEqual(len(decoded), 8)
                    self.assertAlmostEqual(decoded[4].time, 4 / 30, places=5)
                    self.assertAlmostEqual(
                        reread.meta.episodes[1][f"videos/{key}/from_timestamp"], decoded[4].time, places=5
                    )
                    pixels = decoded[4].to_ndarray(format="rgb24")
                    self.assertEqual(pixels.shape, (32, 32, 3))
                    self.assertGreater(pixels[..., 1].mean(), 128)
                    self.assertLess(pixels[..., 0].mean(), 13)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    unittest.main()
