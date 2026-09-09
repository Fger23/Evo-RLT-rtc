#!/usr/bin/env python
"""RLT stages: cache frozen ACP+RTC features, train RL tokens, then chunk actor-critic.

Run ``python -m lerobot.scripts.lerobot_rlt_train {cache,token,ac} --help``.
Only ``cache`` loads PI0.5; token/AC training and checkpoint resume need no VLA GPU memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch

from lerobot.rl.rlt_cache import (
    CACHE_VERSION,
    RLTReplayDataset,
    assemble_episode,
    atomic_json_save,
    atomic_torch_save,
    episode_split,
    sample_mixed_batch,
    success_label,
)
from lerobot.rl.rlt_trainer import RLTActorCriticTrainer, RLTTrainConfig


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    cache = stages.add_parser("cache", help="Build episode shards from actual recorded actions")
    cache.add_argument("--base-policy", required=True, help="Local complete ACP+training-time-RTC checkpoint")
    cache.add_argument(
        "--dataset-spec", required=True, help="JSON list of {repo_id,root,source,success_labels?}"
    )
    cache.add_argument("--output", required=True)
    cache.add_argument("--chunk-size", type=int, default=50)
    cache.add_argument(
        "--rtc-delay", type=int, required=True, help="Committed prefix length matching deployment"
    )
    cache.add_argument("--gamma", type=float, default=0.995, help="Per robot-step discount")
    cache.add_argument("--validation-fraction", type=float, default=0.1)
    cache.add_argument("--token-pool-size", type=int, default=32)
    cache.add_argument(
        "--task", default=None, help="Override dataset task instruction before positive ACP tag"
    )
    cache.add_argument("--tokenizer-path", default=None)
    cache.add_argument("--video-backend", default="pyav")
    cache.add_argument("--tolerance-s", type=float, default=0.04)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--seed", type=int, default=42)
    cache.add_argument("--resume", action="store_true", help="Continue this exact interrupted cache build")

    for stage in ("token", "ac"):
        sub = stages.add_parser(stage)
        sub.add_argument(
            "--cache", required=True, action="append", help="Cache directory; repeat for cumulative data"
        )
        sub.add_argument("--output", required=True)
        sub.add_argument(
            "--steps",
            type=int,
            default=5000 if stage == "token" else 10000,
            help="Additional optimizer steps in this invocation",
        )
        sub.add_argument("--batch-size", type=int, default=64)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--demo-ratio", type=float, default=0.5)
        sub.add_argument("--log-every", type=int, default=100)
        sub.add_argument("--save-every", type=int, default=1000)
        sub.add_argument("--resume", action="store_true", help="Restore optimizer and targets from --policy")
        sub.add_argument(
            "--policy", default=None, help="Previous stage checkpoint directory for warm start/resume"
        )
        if stage == "token":
            sub.add_argument("--lr", type=float, default=3e-4)
        else:
            sub.add_argument(
                "--token-checkpoint", default=None, help="Trained token .pt, required for first AC round"
            )
            sub.add_argument("--actor-lr", type=float, default=1e-4)
            sub.add_argument("--critic-lr", type=float, default=3e-4)
            sub.add_argument("--demo-bc-weight", type=float, default=5.0)
            sub.add_argument("--reference-bc-weight", type=float, default=1.0)
            sub.add_argument(
                "--residual-scale",
                type=float,
                default=None,
                help="Initial round only; normalized action residual bound (default 0.1)",
            )
            sub.add_argument("--policy-delay", type=int, default=2)
            sub.add_argument("--tau", type=float, default=0.005)
            sub.add_argument("--target-noise", type=float, default=0.02)
    return parser


def _log(message):
    print(message if isinstance(message, str) else json.dumps(message, ensure_ascii=False), flush=True)


def _fingerprint_base(path: Path):
    """Hash saved processor/config content and weight inventory without rereading multi-GB weights."""
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if not file.is_file() or file.suffix not in (".json", ".safetensors"):
            continue
        digest.update(str(file.relative_to(path)).encode())
        if file.suffix == ".json" or "processor" in file.name:
            digest.update(file.read_bytes())
        else:
            stat = file.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _resolve_specs(path):
    path = Path(path).resolve()
    specs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not specs:
        raise ValueError("--dataset-spec must contain a nonempty JSON list")
    seen = set()
    for spec in specs:
        if spec.get("source") not in ("demo", "rollout") or not spec.get("repo_id") or not spec.get("root"):
            raise ValueError("Every dataset needs repo_id, root and source ('demo' or 'rollout')")
        if spec["repo_id"] in seen:
            raise ValueError("Dataset repo_id occurs twice in the manifest")
        seen.add(spec["repo_id"])
        for key in ("root", "success_labels"):
            if key in spec:
                candidate = Path(spec[key])
                spec[key] = str((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
    return specs


def _check_device(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use --device cpu for synthetic/small-network checks"
        )


def build_cache(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.rlt.configuration_rlt import RLTConfig
    from lerobot.policies.rlt.modeling_rlt import RLTPolicy
    from lerobot.rl.acp_tags import build_acp_tagged_task

    base_path = Path(args.base_policy).resolve()
    if not (base_path / "config.json").is_file():
        raise ValueError("--base-policy must be a local complete checkpoint directory containing config.json")
    if (
        not 0 <= args.rtc_delay < args.chunk_size
        or not math.isfinite(args.gamma)
        or not 0 < args.gamma <= 1
        or not 0 <= args.validation_fraction < 1
    ):
        raise ValueError("Require 0 <= rtc-delay < chunk-size and gamma in (0,1]")
    output = Path(args.output)
    specs = _resolve_specs(args.dataset_spec)
    prepared = []
    cache_fps = None
    # Preflight all labels before allocating the VLA or processing any video.
    for spec in specs:
        meta = LeRobotDatasetMetadata(spec["repo_id"], root=spec["root"])
        fps = float(meta.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{spec['repo_id']} has invalid dataset fps: {meta.fps!r}")
        if cache_fps is not None and fps != cache_fps:
            raise ValueError(
                "All RLT cache datasets must use the same fps: "
                f"{spec['repo_id']} uses {fps}, expected {cache_fps}. "
                "H, RTC delay and gamma are defined per recorded robot step."
            )
        cache_fps = fps
        labels = (
            json.loads(Path(spec["success_labels"]).read_text(encoding="utf-8"))
            if spec.get("success_labels")
            else {}
        )
        if not isinstance(labels, dict):
            raise ValueError(
                "success_labels must be a JSON object mapping episode indices to success/failure"
            )
        selected = set(spec["episodes"]) if "episodes" in spec else None
        episodes = []
        for row in meta.episodes:
            episode_id = int(row["episode_index"])
            if selected is not None and episode_id not in selected:
                continue
            label = labels.get(str(episode_id), row.get("episode_success"))
            try:
                success = success_label(label)
            except ValueError as exc:
                raise ValueError(f"{spec['repo_id']} episode {episode_id}: {exc}") from exc
            start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
            if stop <= start:
                raise ValueError(f"Empty episode {spec['repo_id']}:{episode_id}")
            episodes.append((episode_id, start, stop, success))
        if not episodes or (selected is not None and selected != {item[0] for item in episodes}):
            raise ValueError(f"No episodes or unknown selected episode indices for {spec['repo_id']}")
        prepared.append((spec, meta, episodes))

    policy = (
        RLTPolicy(
            RLTConfig(
                base_policy_path=str(base_path),
                device=args.device,
                chunk_size=args.chunk_size,
                token_pool_size=args.token_pool_size,
                rtc_prefix_steps=args.rtc_delay,
            )
        )
        .to(args.device)
        .eval()
    )
    if args.rtc_delay > policy.config.rtc_training_max_delay:
        raise ValueError("--rtc-delay exceeds the base checkpoint's trained rtc_training_max_delay")
    if policy.config.rtc_training_max_delay <= 0:
        raise ValueError("RLT+RTC requires a checkpoint trained with rtc_training_max_delay > 0")
    overrides = {"device_processor": {"device": args.device}}
    if args.tokenizer_path:
        overrides["tokenizer_processor"] = {"tokenizer_name": args.tokenizer_path}
    preprocessor, _ = make_pre_post_processors(
        policy.base_policy.config, pretrained_path=str(base_path), preprocessor_overrides=overrides
    )
    normalizers = [step for step in preprocessor.steps if hasattr(step, "stats")]
    if not any("action" in (step.stats or {}) for step in normalizers):
        raise ValueError("Base checkpoint preprocessor lacks action normalization statistics")
    contract = {
        "base_policy_path": str(base_path),
        "base_fingerprint": _fingerprint_base(base_path),
        "chunk_size": args.chunk_size,
        "rtc_delay": args.rtc_delay,
        "gamma": args.gamma,
        "fps": cache_fps,
        "token_pool_size": args.token_pool_size,
        "token_dim": policy.config.token_dim,
        "proprio_dim": policy.config.proprio_dim,
        "action_dim": policy.config.action_feature.shape[0],
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "task": args.task,
        "tokenizer_path": args.tokenizer_path,
        "acp_condition": "positive",
    }
    metadata = {
        "contract": contract,
        "dataset_specs": specs,
        "episode_plan": [
            {"repo_id": spec["repo_id"], "episodes": [list(ep) for ep in episodes]}
            for spec, _, episodes in prepared
        ],
        "reference_semantics": "offline_reencoded_frozen_base_with_recorded_committed_rtc_prefix",
        "action_semantics": "actual_recorded_commands_normalized_by_frozen_base_processor",
    }
    manifest = {"format_version": CACHE_VERSION, "complete": False, "metadata": metadata, "episodes": []}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"{manifest_path} already exists; use --resume or a new output")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["metadata"] != metadata:
            raise ValueError("Cannot resume: dataset specification or base/cache contract changed")
        manifest = existing
    elif args.resume:
        raise FileNotFoundError("--resume requires an existing cache manifest")
    output.mkdir(parents=True, exist_ok=True)
    policy.config.save_pretrained(output / "policy_config")
    manifest["complete"] = False
    atomic_json_save(manifest, manifest_path)
    completed = {entry["episode_uid"] for entry in manifest["episodes"]}
    for spec, meta, episodes in prepared:
        delta = {"action": [i / meta.fps for i in range(args.chunk_size)]}
        intervention_key = "complementary_info.is_intervention"
        if intervention_key in meta.features:
            delta[intervention_key] = delta["action"]
        dataset = LeRobotDataset(
            spec["repo_id"],
            root=spec["root"],
            delta_timestamps=delta,
            video_backend=args.video_backend,
            tolerance_s=args.tolerance_s,
        )
        for episode_id, start, stop, success in episodes:
            uid = f"{spec['repo_id']}:{episode_id}"
            if uid in completed:
                continue
            # Stable per-episode sampling, including after an interrupted build resumes.
            torch.manual_seed(
                int.from_bytes(hashlib.sha256(f"{args.seed}:{uid}".encode()).digest()[:4], "big")
            )
            nodes = []
            policy.reset()
            for absolute in range(start, stop, args.chunk_size):
                k = min(args.chunk_size, stop - absolute)
                # A short final block can lie entirely inside the committed prefix.
                # Only real recorded actions enter it; the rest is masked padding.
                d = 0 if absolute == start else min(args.rtc_delay, k)
                frame = dict(dataset[absolute])
                task = args.task if args.task is not None else frame.get("task", "")
                task = re.sub(r"(?:\n|^)Advantage:\s*(?:positive|negative)\s*", "", task).strip()
                frame["task"] = build_acp_tagged_task(task, is_positive=True)
                intervention = torch.as_tensor(frame.get(intervention_key, torch.zeros(args.chunk_size)))
                intervention = intervention.reshape(-1)
                if (
                    intervention.numel() != args.chunk_size
                    or not ((intervention == 0) | (intervention == 1)).all()
                ):
                    raise ValueError("Recorded intervention indicator must contain H binary per-step values")
                # AddBatchDimensionActionStep only batches rank-1 single actions;
                # a dataset H-action window therefore needs explicit collation.
                pre = preprocessor(torch.utils.data.default_collate([frame]))
                with torch.no_grad():
                    actions = pre["action"][:, : args.chunk_size, : contract["action_dim"]].float()
                    ref, tokens, token_mask = policy.reference_and_features(
                        pre, training_time_rtc=True, inference_delay=d, rtc_action_prefix=actions[:, :d]
                    )
                if actions.shape[1] != args.chunk_size:
                    raise ValueError("Dataset did not provide the full padded H-action window")
                nodes.append(
                    {
                        "start": absolute - start,
                        "actual_steps": k,
                        "prefix_length": d,
                        "tokens": tokens[0].detach().cpu().half(),
                        "token_mask": token_mask[0].cpu(),
                        "proprio": pre["observation.state"][0, : contract["proprio_dim"]].cpu().float(),
                        "ref_actions": ref[0, : args.chunk_size, : contract["action_dim"]].cpu().float(),
                        "actions": actions[0].cpu(),
                        "intervention_mask": intervention.bool(),
                    }
                )
            transitions = assemble_episode(
                nodes,
                episode_uid=uid,
                success=success,
                source=spec["source"],
                chunk_size=args.chunk_size,
                gamma=args.gamma,
            )
            relative = f"episodes/{hashlib.sha256(uid.encode()).hexdigest()[:24]}.pt"
            atomic_torch_save({"episode_uid": uid, "transitions": transitions}, output / relative)
            manifest["episodes"].append(
                {
                    "episode_uid": uid,
                    "source": spec["source"],
                    "success": success,
                    "split": episode_split(uid, args.validation_fraction, args.seed),
                    "file": relative,
                    "count": len(transitions),
                }
            )
            atomic_json_save(manifest, manifest_path)
            _log(f"Cached {uid}: {len(transitions)} transitions, success={success}")
        del dataset
    manifest["complete"] = True
    atomic_json_save(manifest, manifest_path)
    _log(f"Cache complete: {output.resolve()}")


def _training_inputs(args):
    if args.steps < 1 or args.batch_size < 1 or args.log_every < 1 or args.save_every < 1:
        raise ValueError("steps, batch-size, log-every and save-every must be positive")
    if not 0 <= args.demo_ratio <= 1:
        raise ValueError("demo-ratio must be in [0,1]")
    if args.resume and not args.policy:
        raise ValueError("--resume requires --policy pointing at a previous checkpoint")
    output = Path(args.output)
    if args.policy and output.resolve() == Path(args.policy).resolve():
        raise ValueError("Use a new --output directory to preserve the source checkpoint")
    if (output / "trainer_state.pt").exists() or (output / "token.pt").exists():
        raise FileExistsError("Output already contains a trained checkpoint; choose a new --output")
    train = RLTReplayDataset(args.cache, "train")
    val = RLTReplayDataset(args.cache, "val")
    if not len(train):
        raise ValueError("No training episodes; increase data or choose a different cache split seed")
    output.mkdir(parents=True, exist_ok=True)
    _log(
        {
            "train_transitions": len(train),
            "validation_transitions": len(val),
            "demo_transitions": len(train.source_indices[0]),
            "rollout_transitions": len(train.source_indices[1]),
        }
    )
    return output, train, val


def train_token(args):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.rlt.configuration_rlt import RLTConfig  # noqa: F401 - register config type
    from lerobot.policies.rlt.networks import RLTokenModule

    output, train, val = _training_inputs(args)
    config = PreTrainedConfig.from_pretrained(Path(args.cache[0]) / "policy_config")
    if args.policy:
        module = RLTokenModule.load(Path(args.policy) / "token.pt", args.device)
        previous = json.loads((Path(args.policy) / "training_manifest.json").read_text(encoding="utf-8"))
        if previous["cache_contract"] != train.metadata["contract"]:
            raise ValueError("Previous RL-token checkpoint and cache contracts differ")
    else:
        module = RLTokenModule(
            config.token_dim,
            config.latent_dim,
            config.token_heads,
            config.token_encoder_layers,
            config.token_decoder_layers,
            config.token_ff_dim,
            config.num_rl_tokens,
        ).to(args.device)
    if module.token_dim != train.metadata["contract"]["token_dim"]:
        raise ValueError("Token checkpoint dimensions do not match cache")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Token learning rate must be finite and positive")
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr)
    start = 0
    if args.resume:
        state = torch.load(Path(args.policy) / "token_train_state.pt", map_location="cpu", weights_only=True)
        if state["cache_contract"] != train.metadata["contract"] or state["lr"] != args.lr:
            raise ValueError("Token resume requires the original cache contract and learning rate")
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        torch.set_rng_state(state["torch_rng_state"])
        if state.get("cuda_rng_state") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    def save(path, step):
        module.save(path / "token.pt")
        atomic_torch_save(
            {
                "format_version": 1,
                "step": step,
                "optimizer": optimizer.state_dict(),
                "lr": args.lr,
                "cache_contract": train.metadata["contract"],
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            path / "token_train_state.pt",
        )
        atomic_json_save(
            {
                "cache_contract": train.metadata["contract"],
                "architecture": module.architecture(),
                "steps": step,
                "caches": train.roots,
            },
            path / "training_manifest.json",
        )

    module.train()
    for step in range(start + 1, start + args.steps + 1):
        batch = sample_mixed_batch(train, args.batch_size, args.demo_ratio)
        loss = module.reconstruction_loss(
            batch["tokens"].to(args.device).float(), batch["token_mask"].to(args.device)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite RL-token reconstruction loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == start + 1:
            metrics = {"step": step, "token_loss": loss.item()}
            if len(val):
                batch = sample_mixed_batch(val, args.batch_size, args.demo_ratio)
                with torch.no_grad():
                    metrics["validation_token_loss"] = module.reconstruction_loss(
                        batch["tokens"].to(args.device).float(), batch["token_mask"].to(args.device)
                    ).item()
            _log(metrics)
        if step % args.save_every == 0:
            save(output / "checkpoints" / f"step_{step:08d}", step)
    save(output, step)
    _log(f"Token checkpoint: {(output / 'token.pt').resolve()}")


def train_ac(args):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.rlt.configuration_rlt import RLTConfig  # noqa: F401 - register config type
    from lerobot.policies.rlt.modeling_rlt import RLTPolicy

    output, train, val = _training_inputs(args)
    cache_config = PreTrainedConfig.from_pretrained(Path(args.cache[0]) / "policy_config")
    if args.policy:
        if args.token_checkpoint or args.residual_scale is not None:
            raise ValueError(
                "Warm start keeps its trained token encoder/residual bound; omit those overrides"
            )
        policy_config = PreTrainedConfig.from_pretrained(args.policy)
        policy_config.device = args.device
        policy = RLTPolicy.from_pretrained(args.policy, config=policy_config, load_base=False).to(args.device)
        previous_manifest = json.loads(
            (Path(args.policy) / "training_manifest.json").read_text(encoding="utf-8")
        )
        if previous_manifest["cache_contract"] != train.metadata["contract"]:
            raise ValueError("Previous policy and replay cache contracts differ")
    else:
        if not args.token_checkpoint:
            raise ValueError("First AC round requires --token-checkpoint; run the token stage first")
        token_manifest = Path(args.token_checkpoint).parent / "training_manifest.json"
        if not token_manifest.is_file():
            raise ValueError(
                "Token checkpoint must have its training_manifest.json to verify feature compatibility"
            )
        token_metadata = json.loads(token_manifest.read_text(encoding="utf-8"))
        if token_metadata["cache_contract"] != train.metadata["contract"]:
            raise ValueError("Token checkpoint and replay cache contracts differ")
        cache_config.device = args.device
        cache_config.token_checkpoint = str(Path(args.token_checkpoint).resolve())
        if args.residual_scale is not None:
            cache_config.residual_scale = args.residual_scale
        policy = RLTPolicy(cache_config, load_base=False).to(args.device)
    if not policy.config.token_ready:
        raise ValueError("RL-token encoder is not trained")
    trainer = RLTActorCriticTrainer(
        policy,
        RLTTrainConfig(
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            policy_delay=args.policy_delay,
            target_noise=args.target_noise,
            demo_bc_weight=args.demo_bc_weight,
            reference_bc_weight=args.reference_bc_weight,
        ),
    )
    if args.resume:
        trainer.load_state_dict(
            torch.load(Path(args.policy) / "trainer_state.pt", map_location="cpu", weights_only=True)
        )

    def save(path):
        # A deployable checkpoint requires at least one actual actor update.
        policy.config.actor_ready = policy.config.actor_ready or trainer.step >= trainer.config.policy_delay
        policy.config.exploration_std = 0.0
        policy.save_pretrained(path)
        atomic_torch_save(trainer.state_dict(), path / "trainer_state.pt")
        atomic_json_save(
            {
                "cache_contract": train.metadata["contract"],
                "caches": train.roots,
                "steps": trainer.step,
                "parent_policy": args.policy,
                "training_args": vars(args),
            },
            path / "training_manifest.json",
        )

    for _ in range(args.steps):
        batch = sample_mixed_batch(train, args.batch_size, args.demo_ratio)
        metrics = trainer.train_step(batch)
        if trainer.step % args.log_every == 0 or trainer.step == 1:
            if len(val):
                metrics.update(
                    trainer.evaluate_batch(sample_mixed_batch(val, args.batch_size, args.demo_ratio))
                )
            _log(metrics)
        if trainer.step % args.save_every == 0:
            save(output / "checkpoints" / f"step_{trainer.step:08d}")
    save(output)
    _log(f"RLT checkpoint: {output.resolve()} (actor_ready={policy.config.actor_ready})")


def main(argv=None):
    args = make_parser().parse_args(argv)
    _check_device(args.device)
    torch.manual_seed(args.seed)
    {"cache": build_cache, "token": train_token, "ac": train_ac}[args.stage](args)


if __name__ == "__main__":
    main()
