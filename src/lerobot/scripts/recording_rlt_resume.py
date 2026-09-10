"""Resume a validated RLT dataset up to a total episode target, before connecting hardware."""

import copy
import json
from pathlib import Path
from uuid import uuid4

from lerobot.scripts.lerobot_rlt_verify import SEAL, verify_dataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def record_to_target(cfg, recorder):
    # The general recorder interprets num_episodes as additional episodes. Keep
    # that API unchanged, and apply total-target semantics only to RLT recording.
    target = cfg.dataset.num_episodes
    if target <= 0:
        raise ValueError("dataset.num_episodes must be a positive total target")
    root = Path(cfg.dataset.root) if cfg.dataset.root else HF_LEROBOT_HOME / cfg.dataset.repo_id
    existing = 0
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"Dataset root is not a directory: {root}")
        info_path = root / "meta/info.json"
        if info_path.is_file():
            existing = json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"]
            if not isinstance(existing, int) or existing < 0:
                raise ValueError(f"Invalid total_episodes in {info_path}")
        elif any(root.iterdir()):
            raise ValueError(f"Nonempty dataset has no meta/info.json; inspect it before resuming: {root}")
        if existing:
            print(f"RLT_CHECK: validating {existing} saved episodes in {root}", flush=True)
            verify_dataset(
                root,
                None,
                existing,
                cfg.collector_policy_id_policy or cfg.remote_policy.pretrained_name_or_path,
                cfg.dataset.fps,
                expected_task=cfg.dataset.single_task,
            )
        else:
            # An interrupted initialization may leave a zero-episode directory.
            # Preserve all of it before creating a fresh dataset at the same path.
            backup = root.with_name(f"{root.name}.empty-backup-{uuid4().hex}")
            root.rename(backup)
            print(f"RLT_EMPTY: preserved zero-episode directory at {backup}", flush=True)
    remaining = max(0, target - existing)
    print(f"RLT_PROGRESS: valid={existing} target={target} remaining={remaining}", flush=True)
    if not remaining:
        print("RLT_COMPLETE: target already reached; no robot connection or recording started", flush=True)
        return None
    # A seal describes the old file inventory and must be regenerated after append.
    (root / SEAL).unlink(missing_ok=True)
    recording_cfg = copy.copy(cfg)
    recording_cfg.dataset = copy.copy(cfg.dataset)
    recording_cfg.dataset.num_episodes = remaining
    recording_cfg.resume = existing > 0
    return recorder(recording_cfg)
