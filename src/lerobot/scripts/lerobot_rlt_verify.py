"""Check a banknote rollout and seal its data, metadata and videos for transfer."""

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.dataset as pads

CAMERAS = tuple(f"observation.images.{name}" for name in ("left_fixed", "left_wrist", "right_wrist"))
SEAL = "rlt_integrity.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dataset_files(root):
    files = {}
    for directory in ("meta", "data", "videos"):
        require((root / directory).is_dir(), f"Missing directory: {directory}")
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                require(path.resolve().is_relative_to(root.resolve()), f"External file: {path}")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                files[path.relative_to(root).as_posix()] = {
                    "size": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
    return files


def verify_transfer(root, round_number, expected_episodes, expected_policy):
    root = Path(root)
    report = json.loads((root / SEAL).read_text(encoding="utf-8"))
    require(
        report.get("format_version") == 1 and report.get("verified") is True, "Invalid integrity manifest"
    )
    require(root.name == f"dianchao_{round_number}", "Dataset directory does not match --round")
    require(report["round"] == round_number, "Uploaded dataset belongs to another round")
    require(report["episodes"] == expected_episodes, "Unexpected episode count in integrity manifest")
    require(report["policy"] == expected_policy, "Rollouts were collected by another policy")
    actual = dataset_files(root)
    expected = report["files"]
    require(
        actual.keys() == expected.keys(), "Uploaded file inventory differs; upload the complete directory"
    )
    for name in actual:
        require(actual[name] == expected[name], f"Size/SHA256 mismatch: {name}")
    return report


def verify_dataset(root, round_number, expected_episodes, expected_policy, fps=30, expected_task=None):
    root = Path(root)
    if round_number is not None:
        require(root.name == f"dianchao_{round_number}", "Dataset directory does not match --round")
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    require(
        all(key in stats for key in ("action", "observation.state")), "Missing observation/action statistics"
    )
    require(info["fps"] == fps, f"Expected {fps} fps, got {info['fps']}")
    episodes = pads.dataset(root / "meta/episodes", format="parquet").to_table().to_pylist()
    episodes.sort(key=lambda row: row["episode_index"])
    require(
        len(episodes) == expected_episodes == info["total_episodes"], "Episode count does not match target"
    )
    require(
        [row["episode_index"] for row in episodes] == list(range(expected_episodes)),
        "Episode index gaps/duplicates",
    )
    require(expected_episodes > 0, "Empty dataset")
    tasks = pads.dataset(root / "meta/tasks.parquet", format="parquet").to_table()
    require(tasks.num_rows > 0, "Missing tasks")
    if expected_task is not None:
        require(set(tasks.to_pandas().index) == {expected_task}, "Existing dataset uses a different task prompt")
    video_keys = {key for key, feature in info["features"].items() if feature["dtype"] == "video"}
    require(video_keys == set(CAMERAS), f"Expected three banknote cameras, got {video_keys}")
    policy_key = "complementary_info.collector_policy_id"
    columns = [
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "action",
        "observation.state",
        policy_key,
    ]
    table = pads.dataset(root / "data", format="parquet").to_table(columns=columns).sort_by("index")
    require(table.num_rows == info["total_frames"] > 0, "Frame count differs from meta/info.json")
    require(table["index"].to_pylist() == list(range(table.num_rows)), "Global frame index gaps/duplicates")
    require(
        set(table[policy_key].to_pylist()) == {expected_policy},
        "Wrong collector policy; check round/model mapping",
    )
    require(all(0 <= i < tasks.num_rows for i in table["task_index"].to_pylist()), "Invalid task indices")
    for key in ("action", "observation.state"):
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        require(values.shape == (table.num_rows, 12), f"{key} must contain 12 joints per frame")
        require(np.isfinite(values).all(), f"Nonfinite values in {key}")
    success = 0
    cursor = 0
    video_segments = {}
    for episode in episodes:
        index, length = episode["episode_index"], episode["length"]
        require(length > 0, f"Empty episode {index}")
        require(
            episode["dataset_from_index"] == cursor and episode["dataset_to_index"] == cursor + length,
            f"Broken frame range in episode {index}",
        )
        rows = table.slice(cursor, length)
        require(
            rows.num_rows == length and set(rows["episode_index"].to_pylist()) == {index},
            f"Mixed episode {index}",
        )
        require(rows["frame_index"].to_pylist() == list(range(length)), f"Missing frames in episode {index}")
        require(
            np.allclose(rows["timestamp"].to_numpy(), np.arange(length) / fps, atol=0.001, rtol=0),
            f"Incorrect timestamps in episode {index}",
        )
        label = episode.get("episode_success")
        require(label in ("success", "failure"), f"Missing success/failure label in episode {index}")
        success += label == "success"
        cursor += length
        for key in CAMERAS:
            prefix = f"videos/{key}"
            relative = info["video_path"].format(
                video_key=key,
                chunk_index=episode[f"{prefix}/chunk_index"],
                file_index=episode[f"{prefix}/file_index"],
            )
            path = (root / relative).resolve()
            require(
                path.is_relative_to(root.resolve()) and path.is_file(), f"Missing/locality error: {relative}"
            )
            start, end = episode[f"{prefix}/from_timestamp"], episode[f"{prefix}/to_timestamp"]
            require(
                abs((end - start) * fps - length) < 0.1, f"Video duration mismatch: episode {index}, {key}"
            )
            video_segments.setdefault((path, key), []).append((start, length))
    require(cursor == table.num_rows, "Unreferenced rows after final episode")
    for (path, key), segments in video_segments.items():
        segments.sort()
        expected_times = np.concatenate([start + np.arange(length) / fps for start, length in segments])
        require(np.all(np.diff(expected_times) > 0), f"Overlapping video segments: {path}")
        times = []
        shape = info["features"][key]["shape"]
        height, width = shape[1:] if shape[0] == 3 else shape[:2]
        with av.open(str(path)) as container:
            require(len(container.streams.video) == 1, f"Expected one video stream: {path}")
            for frame in container.decode(video=0):
                require(not frame.is_corrupt and frame.time is not None, f"Corrupt video frame: {path}")
                require(
                    frame.width == width and frame.height == height, f"Incorrect video dimensions: {path}"
                )
                times.append(frame.time)
        require(len(times) == len(expected_times), f"Missing/extra decoded video frames: {path}")
        require(
            np.allclose(times, expected_times, atol=0.5 / fps, rtol=0), f"Video timestamp mismatch: {path}"
        )
        print(f"VIDEO_OK {path.relative_to(root.resolve())}: {len(times)} frames", flush=True)
    return {
        "format_version": 1,
        "verified": True,
        "round": round_number,
        "repo_id": f"local/dianchao_{round_number}" if round_number is not None else None,
        "policy": expected_policy,
        "episodes": expected_episodes,
        "frames": cursor,
        "fps": fps,
        "success": success,
        "failure": expected_episodes - success,
        "files": dataset_files(root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--expected-policy", required=True)
    parser.add_argument("--fps", type=int, default=30)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write-manifest", action="store_true", help="Fully decode videos and write transfer checksums"
    )
    mode.add_argument(
        "--check-manifest",
        action="store_true",
        help="Compare uploaded files with the local verification manifest",
    )
    args = parser.parse_args()
    try:
        if args.check_manifest:
            report = verify_transfer(args.root, args.round, args.expected_episodes, args.expected_policy)
            require(report["fps"] == args.fps, "Wrong fps in integrity manifest")
        else:
            # Invalidate an earlier seal before a fresh check so failed rechecks cannot leave a valid-looking seal.
            (args.root / SEAL).unlink(missing_ok=True)
            report = verify_dataset(
                args.root, args.round, args.expected_episodes, args.expected_policy, args.fps
            )
            (args.root / SEAL).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"VERIFIED round={args.round} episodes={report['episodes']} frames={report['frames']} "
            f"success={report['success']} failure={report['failure']} files={len(report['files'])}"
        )
    except Exception as exc:
        parser.exit(1, f"VERIFICATION_FAILED: {exc}\n")


if __name__ == "__main__":
    main()
