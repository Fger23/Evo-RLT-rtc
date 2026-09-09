"""Exercise saved RLT heads and the server's [d,A] RTC prefix without a real VLA."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.rlt.configuration_rlt import RLTConfig
from lerobot.policies.rlt.modeling_rlt import RLTPolicy


def config(**kwargs):
    values = {
        "base_policy_path": "/unchanged/acp_rtc/checkpoint",
        "device": "cpu",
        "token_dim": 12,
        "latent_dim": 8,
        "token_heads": 2,
        "token_encoder_layers": 1,
        "token_decoder_layers": 1,
        "token_ff_dim": 16,
        "num_rl_tokens": 2,
        "token_pool_size": 3,
        "chunk_size": 6,
        "n_action_steps": 6,
        "proprio_dim": 2,
        "actor_hidden_dim": 12,
        "critic_hidden_dim": 12,
        "rtc_training_max_delay": 3,
        "rtc_prefix_steps": 2,
        "input_features": {
            "observation.images.left": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        },
        "output_features": {"action": PolicyFeature(FeatureType.ACTION, (2,))},
        "normalization_mapping": {"ACTION": NormalizationMode.QUANTILES},
    }
    values.update(kwargs)
    return RLTConfig(**values)


def test_compact_checkpoint_does_not_need_original_token_or_base(tmp_path):
    policy = RLTPolicy(config(token_ready=True, actor_ready=True), load_base=False)
    policy._save_pretrained(tmp_path)
    payload = torch.load(tmp_path / "rlt_model.pt", weights_only=True)
    assert set(payload) == {"format_version", "token_module", "actor", "critic"}
    loaded = RLTPolicy.from_pretrained(tmp_path, load_base=False, device="cpu")
    assert loaded.base_policy is None
    assert loaded.config.actor_ready
    state, reference = torch.randn(1, 10), torch.randn(1, 6, 2)
    torch.testing.assert_close(loaded.actor(state, reference), policy.actor(state, reference))


@pytest.mark.parametrize("batched", [False, True])
def test_server_prefix_shape_and_first_chunk(batched):
    policy = RLTPolicy(config(token_ready=True, actor_ready=True), load_base=False)
    reference, state = torch.randn(1, 6, 2), torch.randn(1, 10)
    policy.reference_and_state = lambda batch, **kwargs: (reference, state)
    with torch.no_grad():
        policy.actor.net[-1].bias.fill_(1)
    prefix = torch.tensor([[0.4, -0.3], [0.1, -0.2]])
    if batched:
        prefix = prefix.unsqueeze(0)
    action = policy.predict_action_chunk(
        {}, training_time_rtc=True, inference_delay=2, rtc_action_prefix=prefix
    )
    assert action.shape == (1, 6, 2)
    assert torch.equal(action[0, :2], prefix.reshape(2, 2))
    first = policy.predict_action_chunk({}, training_time_rtc=True, inference_delay=0)
    assert not torch.equal(first[:, :2], reference[:, :2])
    with pytest.raises(ValueError, match="RTC delay differs"):
        policy.predict_action_chunk({}, training_time_rtc=True, inference_delay=1)


def test_untrained_actor_is_not_deployable():
    policy = RLTPolicy(config(), load_base=False)
    with pytest.raises(RuntimeError, match="completed actor-critic"):
        policy.predict_action_chunk({})


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=12)))
        self.calls = 0

    def forward(self, *, inputs_embeds):
        self.calls += 1
        return [inputs_embeds[0] + 1, None], None


class FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = FakeBackbone()

    def embed_prefix(self):
        return torch.ones(1, 5, 12), torch.ones(1, 5, dtype=torch.bool), None


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        cfg = config()
        self.config = SimpleNamespace(
            type="pi05",
            input_features=cfg.input_features,
            output_features=cfg.output_features,
            chunk_size=6,
            n_action_steps=6,
            image_features={"observation.images.left": cfg.input_features["observation.images.left"]},
            robot_state_feature=cfg.input_features["observation.state"],
            normalization_mapping=cfg.normalization_mapping,
            rtc_training_max_delay=3,
            rtc_config=None,
        )
        self.model = FakeCore()
        self.fail = False

    def predict_action_chunk(self, batch, **kwargs):
        prefix, _, _ = self.model.embed_prefix()
        self.model.paligemma_with_expert.forward(inputs_embeds=[prefix, None])
        if self.fail:
            raise RuntimeError("denoising failed")
        return torch.zeros(1, 6, 2)

    def reset(self):
        pass


def test_same_sampling_pass_capture_and_restoration():
    base = FakeBase()
    policy = RLTPolicy(config(), base_policy=base)
    assert not base.weight.requires_grad
    policy.train()
    assert not base.training
    batch = {"observation.images.left": torch.zeros(1, 3, 8, 8), "observation.state": torch.ones(1, 2)}
    reference, state = policy.reference_and_state(batch)
    assert reference.shape == (1, 6, 2) and state.shape == (1, 10)
    assert base.model.paligemma_with_expert.calls == 1
    assert "forward" not in base.model.paligemma_with_expert.__dict__
    assert "embed_prefix" not in base.model.__dict__
    reference, tokens, mask = policy.reference_and_features(batch)
    assert tokens.shape == (1, 3, 12) and mask.shape == (1, 3)
    assert base.model.paligemma_with_expert.calls == 2
    base.fail = True
    with pytest.raises(RuntimeError, match="denoising failed"):
        policy.reference_and_state(batch)
    assert "forward" not in base.model.paligemma_with_expert.__dict__
    assert "embed_prefix" not in base.model.__dict__


def test_camera_order_and_token_width_are_strict():
    base = FakeBase()
    with pytest.raises(ValueError, match="camera order"):
        RLTPolicy(
            config(input_features=dict(reversed(list(base.config.input_features.items())))), base_policy=base
        )
    with pytest.raises(ValueError, match="differs from base VLA width"):
        RLTPolicy(config(token_dim=16), base_policy=base)
    base.config.compile_model = True
    with pytest.raises(ValueError, match="uncompiled"):
        RLTPolicy(config(), base_policy=base)
