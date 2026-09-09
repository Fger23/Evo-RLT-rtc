"""RLT capture through the native sampler and a real, tiny Gemma prefix model."""

import inspect

import pytest
import torch
from torch import nn

from lerobot.policies.rlt.modeling_rlt import RLTPolicy
from lerobot.policies.rlt.networks import pool_prefix_tokens


def test_native_sampler_capture_matches_standalone_prefix_and_preserves_rtc(monkeypatch):
    from transformers import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaModel

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import (
        PaliGemmaWithExpertModel,
        PI05Policy,
        PI05Pytorch,
    )
    from lerobot.policies.rlt.configuration_rlt import RLTConfig
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if "adarms_cond" not in inspect.signature(GemmaModel.forward).parameters:
        pytest.skip("This integration requires the repository's pinned AdaRMS transformers fork.")
    torch.manual_seed(3)
    base_config = PI05Config(
        device="cpu",
        chunk_size=6,
        n_action_steps=6,
        max_action_dim=2,
        rtc_training_max_delay=3,
        num_inference_steps=2,
        input_features={
            "observation.images.left": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
    )
    language_config = GemmaConfig(
        hidden_size=12,
        intermediate_size=24,
        head_dim=4,
        num_attention_heads=3,
        num_key_value_heads=1,
        num_hidden_layers=1,
        vocab_size=32,
        use_adarms=False,
    )
    language_config._attn_implementation = "eager"
    language_model = GemmaModel(language_config).eval()
    paligemma = nn.Module()
    paligemma.config = type("Config", (), {"text_config": language_config})()
    paligemma.language_model = language_model
    backbone = object.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(backbone)
    backbone.paligemma = paligemma
    backbone.freeze_vision_encoder = False
    backbone.train_expert_only = False
    core = object.__new__(PI05Pytorch)
    nn.Module.__init__(core)
    core.config = base_config
    core.paligemma_with_expert = backbone
    core.rtc_processor = None
    embeddings = torch.randn(1, 5, 12)
    valid = torch.tensor([[True, True, True, True, False]])
    attention = torch.zeros(1, 5, dtype=torch.bool)
    monkeypatch.setattr(core, "embed_prefix", lambda *args: (embeddings, valid, attention))
    monkeypatch.setattr(core, "sample_noise", lambda shape, device: torch.ones(shape, device=device))
    monkeypatch.setattr(core, "denoise_step", lambda **kwargs: torch.zeros_like(kwargs["x_t"]))
    base = object.__new__(PI05Policy)
    nn.Module.__init__(base)
    base.config = base_config
    base.model = core
    monkeypatch.setattr(base, "_preprocess_images", lambda batch: ([], []))
    cfg = RLTConfig(
        device="cpu",
        latent_dim=8,
        token_heads=2,
        token_encoder_layers=1,
        token_decoder_layers=1,
        token_ff_dim=16,
        token_pool_size=3,
        num_rl_tokens=2,
        actor_hidden_dim=12,
        critic_hidden_dim=12,
        token_ready=True,
        actor_ready=True,
        rtc_prefix_steps=2,
    )
    policy = RLTPolicy(cfg, base_policy=base)
    calls = []
    language_forward = language_model.forward

    def counted_forward(*args, **kwargs):
        output = language_forward(*args, **kwargs)
        calls.append(output.last_hidden_state.detach())
        return output

    monkeypatch.setattr(language_model, "forward", counted_forward)
    batch = {
        "observation.images.left": torch.zeros(1, 3, 8, 8),
        "observation.state": torch.zeros(1, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
    }
    prefix = torch.tensor([[0.2, 0.4], [-0.2, -0.5]])
    action = policy.predict_action_chunk(
        batch, training_time_rtc=True, inference_delay=2, rtc_action_prefix=prefix
    )
    assert len(calls) == 1
    assert torch.equal(action[0, :2], prefix)
    assert torch.equal(action[0, 2:], torch.ones(4, 2))
    captured_pooled, captured_mask = pool_prefix_tokens(calls[0], valid, 3)
    independent_pooled, independent_mask = policy.extract_prefix_features(batch)
    assert len(calls) == 2
    torch.testing.assert_close(independent_pooled, captured_pooled)
    assert torch.equal(independent_mask, captured_mask)
    assert not any(parameter.requires_grad for parameter in base.parameters())
