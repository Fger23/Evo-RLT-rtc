"""Configuration for a frozen PI0.5 plus a learned RL-token residual policy."""

import math
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("rlt")
@dataclass
class RLTConfig(PreTrainedConfig):
    # This is the COMPLETE, locally fine-tuned ACP + RTC checkpoint, not a base VLA.
    base_policy_path: str = ""
    token_checkpoint: str | None = None
    token_dim: int | None = None
    latent_dim: int = 128
    token_heads: int = 4
    token_encoder_layers: int = 2
    token_decoder_layers: int = 2
    token_ff_dim: int = 512
    num_rl_tokens: int = 4
    token_pool_size: int = 32
    chunk_size: int | None = None
    n_action_steps: int | None = None
    proprio_dim: int | None = None
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    residual_scale: float = 0.1
    exploration_std: float = 0.0
    token_ready: bool = False
    actor_ready: bool = False
    rtc_training_max_delay: int = 0
    rtc_prefix_steps: int | None = None
    rtc_config: RTCConfig | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)
    optimizer_lr: float = 3e-4
    optimizer_weight_decay: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        positive_dims = (
            "latent_dim",
            "token_heads",
            "token_encoder_layers",
            "token_decoder_layers",
            "token_ff_dim",
            "num_rl_tokens",
            "token_pool_size",
            "actor_hidden_dim",
            "critic_hidden_dim",
            "token_dim",
            "chunk_size",
            "n_action_steps",
        )
        for key in positive_dims:
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"RLT {key} must be a positive integer.")
        for key in ("proprio_dim", "rtc_training_max_delay", "rtc_prefix_steps"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"RLT {key} must be a non-negative integer.")
        if self.latent_dim <= 0 or self.token_heads <= 0 or self.latent_dim % self.token_heads:
            raise ValueError("RLT latent_dim must be positive and divisible by token_heads.")
        if self.token_pool_size < 1 or self.num_rl_tokens < 1:
            raise ValueError("RLT token_pool_size and num_rl_tokens must be positive.")
        if (
            not math.isfinite(self.residual_scale)
            or not math.isfinite(self.exploration_std)
            or self.residual_scale <= 0
            or self.exploration_std < 0
        ):
            raise ValueError("RLT residual_scale must be positive; exploration_std must be non-negative.")

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        if self.chunk_size is None:
            raise ValueError("Load the base PI0.5 configuration before requesting action_delta_indices.")
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        if not self.image_features or self.action_feature is None:
            raise ValueError("RLT requires the base PI0.5 image and action features.")
