"""All RLT rounds use the original PI05 checkpoint's processor statistics."""

from types import SimpleNamespace

import pytest

from lerobot.policies import factory
from lerobot.policies.rlt.configuration_rlt import RLTConfig


def test_rlt_processors_always_load_frozen_base(monkeypatch):
    config = RLTConfig(base_policy_path="original_acp_rtc", device="cpu")
    calls = []
    monkeypatch.setattr(
        factory.PreTrainedConfig,
        "from_pretrained",
        lambda path: SimpleNamespace(type="pi05", device="cuda:7"),
    )

    def load(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(factory.PolicyProcessorPipeline, "from_pretrained", load)
    overrides = {"device_processor": {"device": "cpu"}}
    factory.make_pre_post_processors(
        config,
        pretrained_path="round_3_actor",
        preprocessor_overrides=overrides,
        dataset_stats={"action": {"new_rollout_statistics": 123}},
    )
    assert len(calls) == 2
    assert all(call["pretrained_model_name_or_path"] == "original_acp_rtc" for call in calls)
    assert calls[0]["overrides"] == overrides


def test_rlt_processors_reject_another_rlt_as_base(monkeypatch):
    config = RLTConfig(base_policy_path="wrong_base", device="cpu")
    monkeypatch.setattr(
        factory.PreTrainedConfig, "from_pretrained", lambda path: SimpleNamespace(type="rlt")
    )
    with pytest.raises(ValueError, match="frozen PI05"):
        factory.make_pre_post_processors(config)
