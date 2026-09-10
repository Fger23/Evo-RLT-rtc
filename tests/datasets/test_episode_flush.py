"""Native per-episode checkpoints without disconnecting a recording dataset."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class EpisodeFlushTests(unittest.TestCase):
    def make_dataset(self, root, *, video=True, batch_encoding_size=1):
        features = {
            key: {"dtype": "float32", "shape": (2,), "names": ["left", "right"]}
            for key in ("action", "observation.state")
        }
        if video:
            features["observation.images.fixed"] = {
                "dtype": "video",
                "shape": (32, 32, 3),
                "names": ["height", "width", "channels"],
            }
        dataset = LeRobotDataset.create(
            "local/episode_flush",
            fps=30,
            features=features,
            root=root,
            use_videos=video,
            image_writer_threads=2,
            batch_encoding_size=batch_encoding_size,
            video_backend="pyav",
            vcodec="h264",
        )
        # Exercise file and chunk advancement, including the chunk boundary.
        dataset.meta.update_chunk_settings(chunks_size=2)
        return dataset

    def add_episode(self, dataset, episode):
        for frame in range(4):
            values = np.array([episode * 10 + frame, episode], dtype=np.float32)
            data = {"action": values, "observation.state": values, "task": "Count banknotes"}
            if dataset.meta.video_keys:
                image = np.zeros((32, 32, 3), dtype=np.uint8)
                image[..., episode % 3] = 180
                data["observation.images.fixed"] = image
            dataset.add_frame(data)
        dataset.save_episode(
            extra_episode_metadata={"episode_success": "success" if episode % 2 else "failure"}
        )

    def assert_readable(self, dataset, episodes):
        root = dataset.root
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        self.assertEqual((info["total_episodes"], info["total_frames"]), (episodes, 4 * episodes))
        metadata = pa.concat_tables(
            [pq.read_table(path) for path in sorted((root / "meta/episodes").rglob("*.parquet"))]
        ).to_pydict()
        data = pa.concat_tables(
            [pq.read_table(path) for path in sorted((root / "data").rglob("*.parquet"))]
        ).to_pydict()
        self.assertEqual(metadata["episode_index"], list(range(episodes)))
        self.assertEqual(metadata["dataset_from_index"], list(range(0, 4 * episodes, 4)))
        self.assertEqual(metadata["dataset_to_index"], list(range(4, 4 * episodes + 1, 4)))
        self.assertEqual(
            metadata["episode_success"], ["success" if ep % 2 else "failure" for ep in range(episodes)]
        )
        self.assertEqual(data["index"], list(range(4 * episodes)))
        self.assertEqual(data["episode_index"], [ep for ep in range(episodes) for _ in range(4)])
        expected = np.array([[ep * 10 + frame, ep] for ep in range(episodes) for frame in range(4)])
        np.testing.assert_array_equal(data["action"], expected)
        stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(stats["action"]["mean"], expected.mean(axis=0))
        np.testing.assert_allclose(stats["action"]["std"], expected.std(axis=0), atol=1e-6)
        self.assertEqual(stats["action"]["count"], [4 * episodes])
        self.assertEqual(len(dataset.meta.episodes), episodes)
        # A different reader can consume the files while the recorder stays alive.
        reread = LeRobotDataset("local/episode_flush", root=root, video_backend="pyav")
        self.assertEqual((reread.num_episodes, len(reread)), (episodes, 4 * episodes))
        np.testing.assert_array_equal(reread.hf_dataset[4 * episodes - 1]["action"], expected[-1])
        if dataset.meta.video_keys:
            decoded = []
            for path in sorted((root / "videos").rglob("*.mp4")):
                with av.open(str(path)) as container:
                    decoded.extend(list(container.decode(video=0)))
            self.assertEqual(len(decoded), 4 * episodes)
            for ep in range(episodes):
                pixels = decoded[4 * ep].to_ndarray(format="rgb24")
                self.assertGreater(pixels[..., ep % 3].mean(), 160)
                self.assertAlmostEqual(
                    metadata["videos/observation.images.fixed/from_timestamp"][ep], ep * 4 / 30
                )

    def test_save_flush_and_append_preserves_every_episode(self):
        with tempfile.TemporaryDirectory(prefix="lerobot-flush-") as temporary:
            dataset = self.make_dataset(Path(temporary) / "dataset")
            try:
                dataset.flush()  # Also safe before the first episode.
                original_files = {}
                for episode in range(4):
                    self.add_episode(dataset, episode)
                    dataset.flush()
                    dataset.flush()  # Repeating a checkpoint must not advance file indices twice.
                    self.assertIsNone(dataset.writer)
                    self.assertIsNone(dataset.meta.writer)
                    self.assertEqual(dataset.meta.metadata_buffer, [])
                    self.assertEqual(dataset.episode_buffer["episode_index"], episode + 1)
                    self.assert_readable(dataset, episode + 1)
                    for path, digest in original_files.items():
                        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                    original_files = {
                        path: hashlib.sha256(path.read_bytes()).hexdigest()
                        for base in (dataset.root / "data", dataset.root / "meta/episodes")
                        for path in base.rglob("*.parquet")
                    }
                self.assertEqual(dataset.meta.episodes[-1]["data/chunk_index"], 1)
                self.assertEqual(dataset.meta.episodes[-1]["meta/episodes/chunk_index"], 1)
            finally:
                dataset.finalize()
                dataset.stop_image_writer()

    def test_flush_handles_buffered_metadata_and_resume_from_disk(self):
        with tempfile.TemporaryDirectory(prefix="lerobot-flush-resume-") as temporary:
            dataset = self.make_dataset(Path(temporary) / "dataset", video=False)
            try:
                for episode in range(2):
                    self.add_episode(dataset, episode)
                dataset.flush()
                self.assert_readable(dataset, 2)
                self.add_episode(dataset, 2)
                dataset.flush()
                self.assert_readable(dataset, 3)
                resumed = LeRobotDataset("local/episode_flush", root=dataset.root, video_backend="pyav")
                try:
                    self.add_episode(resumed, 3)
                    resumed.flush()
                    self.assert_readable(resumed, 4)
                    self.add_episode(resumed, 4)
                    resumed.flush()
                    self.assert_readable(resumed, 5)
                finally:
                    resumed.finalize()
            finally:
                dataset.finalize()
                dataset.stop_image_writer()

    def test_flush_rejects_unencoded_video_batch(self):
        with tempfile.TemporaryDirectory(prefix="lerobot-flush-pending-") as temporary:
            dataset = self.make_dataset(Path(temporary) / "dataset", batch_encoding_size=2)
            try:
                self.add_episode(dataset, 0)
                with self.assertRaisesRegex(ValueError, "batch_encoding_size=1"):
                    dataset.flush()
            finally:
                dataset.finalize()
                dataset.stop_image_writer()


if __name__ == "__main__":
    unittest.main()
