"""CPU tests; also runnable directly without the repository's hardware fixtures."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.rlt.configuration_rlt import RLTConfig
from lerobot.policies.rlt.modeling_rlt import RLTPolicy
from lerobot.rl.rlt_cache import (
    RLTReplayDataset,
    assemble_episode,
    atomic_json_save,
    atomic_torch_save,
    episode_split,
    sample_mixed_batch,
    success_label,
)
from lerobot.rl.rlt_trainer import RLTActorCriticTrainer, RLTTrainConfig
from lerobot.scripts.lerobot_rlt_train import main


def small_config():
    return RLTConfig(
        base_policy_path="/synthetic/frozen-base-does-not-exist",
        device="cpu",
        token_dim=8,
        latent_dim=8,
        token_heads=2,
        token_encoder_layers=1,
        token_decoder_layers=1,
        token_ff_dim=16,
        num_rl_tokens=2,
        token_pool_size=3,
        chunk_size=4,
        n_action_steps=4,
        proprio_dim=3,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        rtc_training_max_delay=2,
        rtc_prefix_steps=1,
        input_features={
            "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
            "observation.state": PolicyFeature(FeatureType.STATE, (3,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )


def nodes(length=9):
    result = []
    for start in range(0, length, 4):
        k, d = min(4, length - start), 0 if start == 0 else 1
        actions = torch.full((4, 2), 0.4)
        reference = torch.zeros(4, 2)
        reference[:d] = actions[:d]
        result.append(
            {
                "start": start,
                "actual_steps": k,
                "prefix_length": d,
                "tokens": torch.randn(3, 8).half(),
                "token_mask": torch.ones(3, dtype=torch.bool),
                "proprio": torch.zeros(3),
                "ref_actions": reference,
                "actions": actions,
                "intervention_mask": torch.tensor([False, True, False, False]),
            }
        )
    return result


def write_cache(root, uid_prefix="synthetic"):
    root = Path(root)
    config = small_config()
    config.save_pretrained(root / "policy_config")
    contract = {
        "base_policy_path": config.base_policy_path,
        "base_fingerprint": "synthetic-v1",
        "chunk_size": 4,
        "rtc_delay": 1,
        "gamma": 0.9,
        "token_pool_size": 3,
        "token_dim": 8,
        "proprio_dim": 3,
        "action_dim": 2,
        "seed": 42,
    }
    manifest = {"format_version": 1, "complete": True, "metadata": {"contract": contract}, "episodes": []}
    for episode_id, (source, success, split) in enumerate(
        (("demo", True, "train"), ("rollout", False, "train"), ("rollout", True, "val"))
    ):
        uid = f"{uid_prefix}:{episode_id}"
        transitions = assemble_episode(
            nodes(), episode_uid=uid, success=success, source=source, chunk_size=4, gamma=0.9
        )
        relative = f"episodes/{episode_id}.pt"
        atomic_torch_save({"episode_uid": uid, "transitions": transitions}, root / relative)
        manifest["episodes"].append(
            {
                "episode_uid": uid,
                "source": source,
                "success": success,
                "split": split,
                "file": relative,
                "count": len(transitions),
            }
        )
    atomic_json_save(manifest, root / "manifest.json")
    return root


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_explicit_outcomes_are_rewards(self):
        for value in (None, "unknown", "positive", "", 2, float("nan"), {"acp_indicator": 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                success_label(value)
        self.assertTrue(success_label("success"))
        self.assertFalse(success_label("failure"))

    def test_actual_actions_temporal_alignment_terminal_mask_reward(self):
        blocks = nodes(length=10)
        transitions = assemble_episode(
            blocks, episode_uid="demo:0", success=True, source="demo", chunk_size=4, gamma=0.9
        )
        self.assertEqual([int(t["next_start"]) for t in transitions], [4, 8, 10])
        self.assertTrue(torch.equal(transitions[0]["actions"], blocks[0]["actions"]))
        self.assertFalse(torch.equal(transitions[0]["actions"], transitions[0]["ref_actions"]))
        self.assertAlmostEqual(transitions[0]["discount"].item(), 0.9**4)
        self.assertAlmostEqual(transitions[-1]["reward"].item(), 0.9)
        self.assertEqual(transitions[-1]["discount"].item(), 0)
        self.assertEqual(transitions[-1]["action_mask"].tolist(), [True, True, False, False])
        blocks[1]["start"] = 1
        with self.assertRaises(ValueError):
            assemble_episode(blocks, episode_uid="bad", success=1, source="demo", chunk_size=4, gamma=0.9)

    def test_split_stability_mix_and_duplicate_rejection(self):
        self.assertEqual(episode_split("demo:0"), episode_split("demo:0"))
        with tempfile.TemporaryDirectory() as directory:
            root = write_cache(Path(directory) / "cache")
            train, val = RLTReplayDataset([root]), RLTReplayDataset([root], "val")
            train_ids = {entry["episode_uid"] for entry in train.entries}
            self.assertTrue(train_ids.isdisjoint(entry["episode_uid"] for entry in val.entries))
            batch = sample_mixed_batch(train, 10, 0.3)
            self.assertEqual((batch["source"] == 0).sum().item(), 3)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                RLTReplayDataset([root, root])

    def test_terminal_target_has_no_bootstrap_and_hil_bc(self):
        config = small_config()
        policy = RLTPolicy(config, load_base=False)
        trainer = RLTActorCriticTrainer(policy, RLTTrainConfig(policy_delay=1))
        records = assemble_episode(
            nodes(), episode_uid="hil:0", success=1, source="rollout", chunk_size=4, gamma=0.9
        )
        batch = torch.utils.data.default_collate([records[0], records[-1]])
        prepared = trainer.prepare_batch(batch)
        _, following = trainer.states(prepared)
        target = trainer.td_target(prepared, following)
        self.assertEqual(target[-1].item(), 1.0)
        frozen = copy.deepcopy(policy.token_module.state_dict())
        metrics = trainer.train_step(batch)
        self.assertGreater(metrics["demo_bc"], 0, "Successful rollout HIL actions must receive BC")
        for key, tensor in policy.token_module.state_dict().items():
            self.assertTrue(torch.equal(tensor, frozen[key]))
        with self.assertRaises(ValueError):
            RLTTrainConfig(actor_lr=float("nan"))

    def test_token_ac_cli_save_warm_start_and_resume_without_base(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cache = write_cache(directory / "cache")
            token, first, resumed = directory / "token", directory / "round1", directory / "round2"
            shared = ["--cache", str(cache), "--device", "cpu", "--batch-size", "4", "--log-every", "2"]
            main(["token", *shared, "--output", str(token), "--steps", "3"])
            main(
                [
                    "ac",
                    *shared,
                    "--token-checkpoint",
                    str(token / "token.pt"),
                    "--output",
                    str(first),
                    "--steps",
                    "4",
                ]
            )
            original = RLTPolicy.from_pretrained(first, load_base=False, device="cpu")
            self.assertTrue(original.config.actor_ready)
            main(
                ["ac", *shared, "--policy", str(first), "--resume", "--output", str(resumed), "--steps", "2"]
            )
            final = RLTPolicy.from_pretrained(resumed, load_base=False, device="cpu")
            state = torch.load(resumed / "trainer_state.pt", weights_only=True)
            self.assertEqual(state["step"], 6)
            self.assertTrue(
                any(
                    not torch.equal(value, original.actor.state_dict()[key])
                    for key, value in final.actor.state_dict().items()
                )
            )
            for key, value in final.token_module.state_dict().items():
                self.assertTrue(torch.equal(value, original.token_module.state_dict()[key]))
            reference = torch.randn(2, 4, 2)
            actual = final.actor(torch.randn(2, 11), reference, prefix_lengths=torch.tensor([1, 2]))
            self.assertTrue(torch.equal(actual[0, :1], reference[0, :1]))
            self.assertTrue(torch.equal(actual[1, :2], reference[1, :2]))

    def test_cache_uses_real_processors_and_batched_normalized_actions(self):
        """Exercise the cache CLI with real AddBatch/Normalizer and lightweight fake VLA/data."""
        from lerobot.processor import (
            AddBatchDimensionProcessorStep,
            NormalizerProcessorStep,
            PolicyProcessorPipeline,
        )

        config = small_config()
        stats = {
            "action": {"min": torch.zeros(2), "max": torch.full((2,), 10.0)},
            "observation.state": {"min": torch.zeros(3), "max": torch.full((3,), 10.0)},
        }
        pipeline = PolicyProcessorPipeline(
            steps=[
                AddBatchDimensionProcessorStep(),
                NormalizerProcessorStep(
                    features={**config.input_features, **config.output_features},
                    norm_map={
                        "VISUAL": NormalizationMode.IDENTITY,
                        "STATE": NormalizationMode.MIN_MAX,
                        "ACTION": NormalizationMode.MIN_MAX,
                    },
                    stats=stats,
                ),
            ]
        )

        class FakeMetadata:
            fps = 10
            episodes = [
                {
                    "episode_index": 0,
                    "dataset_from_index": 0,
                    "dataset_to_index": 9,
                    "episode_success": "success",
                }
            ]
            features = {"complementary_info.is_intervention": {}}

            def __init__(self, *args, **kwargs):
                pass

        class FakeDataset:
            def __init__(self, *args, **kwargs):
                self.delta = kwargs["delta_timestamps"]
                assert self.delta["action"] == [0.0, 0.1, 0.2, 0.3]

            def __getitem__(self, index):
                return {
                    "action": torch.full((4, 2), 7.0),
                    "observation.state": torch.full((3,), 5.0),
                    "observation.images.top": torch.zeros(3, 8, 8),
                    "task": "count notes",
                    "complementary_info.is_intervention": torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
                }

        class FakeBase:
            def __init__(self):
                self.config = config

            def predict_action_chunk(self, pre, **kwargs):
                assert pre["action"].shape == (1, 4, 2)
                assert pre["task"] == ["count notes\nAdvantage: positive"]
                result = torch.zeros_like(pre["action"])
                d = kwargs["inference_delay"]
                result[:, :d] = kwargs["rtc_action_prefix"]
                return result

        class FakePolicy:
            def __init__(self, supplied):
                self.config = copy.deepcopy(config)
                self.config.base_policy_path = supplied.base_policy_path
                self.config.rtc_prefix_steps = supplied.rtc_prefix_steps
                self.base_policy = FakeBase()

            def to(self, device):
                return self

            def eval(self):
                return self

            def reset(self):
                pass

            def extract_prefix_features(self, pre):
                return torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.bool)

            def reference_and_features(self, pre, **kwargs):
                return self.base_policy.predict_action_chunk(pre, **kwargs), *self.extract_prefix_features(
                    pre
                )

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            base = directory / "base"
            base.mkdir()
            (base / "config.json").write_text("{}", encoding="utf-8")
            spec = directory / "datasets.json"
            spec.write_text(json.dumps([{"repo_id": "test/demo", "root": "data", "source": "demo"}]))
            output = directory / "cache"
            with (
                patch("lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", FakeMetadata),
                patch("lerobot.datasets.lerobot_dataset.LeRobotDataset", FakeDataset),
                patch("lerobot.policies.rlt.modeling_rlt.RLTPolicy", FakePolicy),
                patch("lerobot.policies.factory.make_pre_post_processors", return_value=(pipeline, None)),
            ):
                main(
                    [
                        "cache",
                        "--base-policy",
                        str(base),
                        "--dataset-spec",
                        str(spec),
                        "--output",
                        str(output),
                        "--chunk-size",
                        "4",
                        "--rtc-delay",
                        "1",
                        "--device",
                        "cpu",
                        "--validation-fraction",
                        "0",
                    ]
                )
            replay = RLTReplayDataset([output])
            self.assertEqual(replay.metadata["contract"]["fps"], 10)
            self.assertEqual(len(replay), 3)
            self.assertTrue(torch.allclose(replay[0]["actions"], torch.full((4, 2), 0.4)))
            self.assertTrue(torch.equal(replay[1]["ref_actions"][:1], replay[1]["actions"][:1]))
            self.assertEqual(replay[2]["prefix_length"].item(), 1)
            self.assertEqual(replay[2]["actual_steps"].item(), 1)

    def test_cache_rejects_mixed_fps_before_loading_vla(self):
        episode = {
            "episode_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": 9,
            "episode_success": "success",
        }
        metadata = [SimpleNamespace(fps=30, episodes=[episode]), SimpleNamespace(fps=25, episodes=[episode])]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            base = directory / "base"
            base.mkdir()
            (base / "config.json").write_text("{}", encoding="utf-8")
            spec = directory / "datasets.json"
            spec.write_text(
                json.dumps(
                    [
                        {"repo_id": "test/demo", "root": "demo", "source": "demo"},
                        {"repo_id": "test/rollout", "root": "rollout", "source": "rollout"},
                    ]
                ),
                encoding="utf-8",
            )
            with (
                patch("lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata", side_effect=metadata),
                patch("lerobot.policies.rlt.modeling_rlt.RLTPolicy") as policy_constructor,
                self.assertRaisesRegex(ValueError, "same fps"),
            ):
                main(
                    [
                        "cache",
                        "--base-policy",
                        str(base),
                        "--dataset-spec",
                        str(spec),
                        "--output",
                        str(directory / "cache"),
                        "--rtc-delay",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )
            policy_constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
