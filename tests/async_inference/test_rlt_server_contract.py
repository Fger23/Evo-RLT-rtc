"""RLT must share the deployed learned-RTC transport contract."""

from types import SimpleNamespace

import pytest

from lerobot.async_inference.policy_server import PolicyServer


def _setup(*, policy_type="rlt", delay=21, trained_delay=37, horizon=50):
    server = object.__new__(PolicyServer)
    server.policy = SimpleNamespace(
        config=SimpleNamespace(
            chunk_size=horizon, rtc_training_max_delay=trained_delay, rtc_config=None, rtc_prefix_steps=21
        )
    )
    specs = SimpleNamespace(
        policy_type=policy_type,
        training_time_rtc=True,
        protocol_version=2,
        return_raw_actions=True,
        acp_positive_prompt=True,
        use_cfg=False,
        cfg_beta=1.0,
        rtc_prefix_steps=delay,
        actions_per_chunk=horizon,
    )
    return server, specs


@pytest.mark.parametrize("policy_type", ["pi05", "rlt"])
def test_base_and_rlt_accept_the_same_trained_prefix(policy_type):
    server, specs = _setup(policy_type=policy_type)
    server._validate_training_time_rtc_setup(specs)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("protocol_version", 1, "protocol_version"),
        ("return_raw_actions", False, "return_raw_actions"),
        ("acp_positive_prompt", False, "positive ACP"),
        ("use_cfg", True, "CFG"),
        ("cfg_beta", 2.0, "cfg_beta"),
        ("rtc_prefix_steps", 38, "exceeds"),
        ("actions_per_chunk", 10, "chunk_size"),
    ],
)
def test_rlt_rejects_incompatible_transport(field, value, match):
    server, specs = _setup()
    setattr(specs, field, value)
    with pytest.raises(ValueError, match=match):
        server._validate_training_time_rtc_setup(specs)


def test_rlt_rejects_base_without_learned_rtc():
    server, specs = _setup(trained_delay=0)
    with pytest.raises(ValueError, match="not trained"):
        server._validate_training_time_rtc_setup(specs)


def test_rlt_rejects_legacy_remote_execution():
    server, specs = _setup()
    specs.training_time_rtc = False
    with pytest.raises(ValueError, match="requires.*training_time_rtc"):
        server._validate_training_time_rtc_setup(specs)


def test_rlt_rejects_delay_change_within_base_training_range():
    server, specs = _setup(delay=10)
    with pytest.raises(ValueError, match="prefix used for actor training"):
        server._validate_training_time_rtc_setup(specs)


def test_rlt_rejects_mixing_gradient_guidance_with_learned_rtc():
    server, specs = _setup()
    server.policy.config.rtc_config = SimpleNamespace(enabled=True)
    with pytest.raises(ValueError, match="gradient-guided"):
        server._validate_training_time_rtc_setup(specs)
