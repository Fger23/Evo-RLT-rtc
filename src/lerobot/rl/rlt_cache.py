"""Episode-safe RLT replay caches, independent of robots and large VLA dependencies.

Actions in this format are *recorded executed commands*, normalized with the frozen
base checkpoint's processor. References are separate, counterfactual VLA samples.
Re-encoding old episodes with an RTC prefix is an offline approximation: it does
not recover the original asynchronous request times or behavior-policy samples.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

CACHE_VERSION = 1


def success_label(value) -> bool:
    """Require explicit outcomes; ACP quality/intervention labels are not outcomes."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("episode_success must contain one explicit success/failure label")
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        if value.strip().lower() in ("success", "true", "1"):
            return True
        if value.strip().lower() in ("failure", "false", "0"):
            return False
    raise ValueError(f"Missing or invalid episode_success: {value!r}; label this episode explicitly")


def episode_split(episode_uid: str, validation_fraction: float = 0.1, seed: int = 42) -> str:
    """A stable split as new rollout datasets are added; no episode can leak across splits."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    digest = hashlib.sha256(f"{seed}:{episode_uid}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if fraction < validation_fraction else "train"


def assemble_episode(
    nodes: list[dict], *, episode_uid: str, success, source: str, chunk_size: int, gamma: float
) -> list[dict]:
    """Link a contiguous sequence of H-step blocks, preserving the short terminal block.

    Each node has normalized ``actions/ref_actions[H,A]``, encoded observation,
    ``start`` relative to this episode and ``actual_steps`` in [1,H]. A terminal
    reward is delivered after the final action, hence gamma ** (k-1).
    """
    outcome = success_label(success)
    if source not in ("demo", "rollout"):
        raise ValueError("source must be demo or rollout")
    if chunk_size < 1 or not math.isfinite(gamma) or not 0 < gamma <= 1 or not nodes:
        raise ValueError("Require a nonempty episode, positive chunk_size and gamma in (0,1]")
    transitions = []
    expected_start = 0
    for i, node in enumerate(nodes):
        k = int(node["actual_steps"])
        d = int(node["prefix_length"])
        if node["start"] != expected_start or not 1 <= k <= chunk_size or not 0 <= d <= k or d >= chunk_size:
            raise ValueError(
                "Blocks must be contiguous t -> t+k, with 0 <= prefix_length <= actual_steps <= H and d < H"
            )
        terminal = i == len(nodes) - 1
        if not terminal and k != chunk_size:
            raise ValueError("Only the terminal block may be shorter than H")
        for key in ("actions", "ref_actions"):
            if node[key].ndim != 2 or node[key].shape[0] != chunk_size:
                raise ValueError(f"{key} must have shape [H,A]")
            if not torch.isfinite(node[key]).all():
                raise ValueError(f"Nonfinite {key}")
        if node["actions"].shape != node["ref_actions"].shape:
            raise ValueError("Executed and reference action dimensions differ")
        if (
            node["tokens"].ndim != 2
            or node["token_mask"].shape != node["tokens"].shape[:1]
            or not node["token_mask"].bool().any()
            or node["proprio"].ndim != 1
        ):
            raise ValueError("Expected observation tokens [M,E], valid mask [M] and proprio [P]")
        if not torch.isfinite(node["tokens"]).all() or not torch.isfinite(node["proprio"]).all():
            raise ValueError("Cached observation features contain NaN or Inf")
        if not torch.equal(node["actions"][:d], node["ref_actions"][:d]):
            raise ValueError("RTC reference prefix must equal the actually committed normalized actions")
        next_node = node if terminal else nodes[i + 1]
        record = {key: node[key] for key in ("tokens", "token_mask", "proprio", "ref_actions", "actions")}
        record.update(
            {f"next_{key}": next_node[key] for key in ("tokens", "token_mask", "proprio", "ref_actions")}
        )
        record.update(
            intervention_mask=node.get("intervention_mask", torch.zeros(chunk_size, dtype=torch.bool)),
            action_mask=torch.arange(chunk_size) < k,
            next_action_mask=torch.arange(chunk_size) < int(next_node["actual_steps"]),
            prefix_length=torch.tensor(d, dtype=torch.long),
            next_prefix_length=torch.tensor(int(next_node["prefix_length"]), dtype=torch.long),
            reward=torch.tensor(float(outcome) * gamma ** (k - 1) if terminal else 0.0),
            discount=torch.tensor(0.0 if terminal else gamma**k),
            done=torch.tensor(terminal),
            actual_steps=torch.tensor(k, dtype=torch.long),
            source=torch.tensor(0 if source == "demo" else 1, dtype=torch.long),
            success=torch.tensor(outcome),
            start=torch.tensor(expected_start, dtype=torch.long),
            next_start=torch.tensor(expected_start + k, dtype=torch.long),
        )
        if record["intervention_mask"].shape != (chunk_size,):
            raise ValueError("intervention_mask must have shape [H]")
        transitions.append(record)
        expected_start += k
    return transitions


def atomic_torch_save(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_save(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class RLTReplayDataset(Dataset):
    """Lazy per-episode storage; duplicate episodes and incompatible caches are rejected."""

    def __init__(self, cache_paths: list[str | Path], split: str = "train", resident_episodes: int = 8):
        self.entries = []
        self.cumulative = []
        self.source_indices = {0: [], 1: []}
        self._resident = OrderedDict()
        self.resident_episodes = max(1, resident_episodes)
        seen = set()
        self.metadata = None
        self.roots = []
        total = 0
        for root in map(Path, cache_paths):
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("format_version") != CACHE_VERSION:
                raise ValueError(f"Unsupported cache format: {root}")
            if not manifest.get("complete", False):
                raise ValueError(f"Incomplete cache: {root}; finish the cache command before training")
            metadata = manifest["metadata"]
            if self.metadata is None:
                self.metadata = metadata
            elif metadata["contract"] != self.metadata["contract"]:
                raise ValueError("Cache normalization/base-policy/chunk/RTC/split contracts differ")
            self.roots.append(str(root.resolve()))
            for entry in manifest["episodes"]:
                uid = entry["episode_uid"]
                if uid in seen:
                    raise ValueError(f"Duplicate episode {uid}; do not include the same demonstrations twice")
                seen.add(uid)
                if entry["split"] != split:
                    continue
                self.entries.append({**entry, "path": root / entry["file"]})
                source_id = 0 if entry["source"] == "demo" else 1
                self.source_indices[source_id].extend(range(total, total + entry["count"]))
                total += entry["count"]
                self.cumulative.append(total)
        if self.metadata is None:
            raise ValueError("At least one cache is required")

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self.cumulative, index)
        if episode_index not in self._resident:
            payload = torch.load(self.entries[episode_index]["path"], map_location="cpu", weights_only=True)
            self._resident[episode_index] = payload["transitions"]
            if len(self._resident) > self.resident_episodes:
                self._resident.popitem(last=False)
        self._resident.move_to_end(episode_index)
        offset = self.cumulative[episode_index - 1] if episode_index else 0
        return self._resident[episode_index][index - offset]


def sample_mixed_batch(dataset: RLTReplayDataset, batch_size: int, demo_ratio: float, generator=None):
    """Mix sources explicitly, without amplifying the demo cache as rounds accumulate."""
    if not len(dataset) or batch_size < 1 or not 0 <= demo_ratio <= 1:
        raise ValueError("Require nonempty replay, positive batch_size and demo_ratio in [0,1]")
    demo, rollout = dataset.source_indices[0], dataset.source_indices[1]
    if demo and rollout:  # noqa: SIM108 - spell out the single-source fallback
        n_demo = round(batch_size * demo_ratio)
    else:
        n_demo = batch_size if demo else 0
    selected = []
    for pool, count in ((demo, n_demo), (rollout, batch_size - n_demo)):
        if count:
            offsets = torch.randint(len(pool), (count,), generator=generator).tolist()
            selected.extend(dataset[pool[offset]] for offset in offsets)
    return torch.utils.data.default_collate(selected)
