"""Frozen native PI0.5 adapter with RL-token features and chunk actor-critic heads."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from threading import RLock

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rlt.configuration_rlt import RLTConfig
from lerobot.policies.rlt.networks import (
    ResidualChunkActor,
    RLTokenModule,
    TwinChunkCritic,
    pool_prefix_tokens,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


class RLTPolicy(PreTrainedPolicy):
    """Learn small RLT modules while preserving a complete ACP + RTC checkpoint.

    Inputs and outputs use the BASE checkpoint's processor and normalized action
    space. Checkpoints reference the frozen base by path, never duplicate its
    multi-GB weights. ``load_base=False`` supports offline CPU/GPU head training
    using fully resolved config and cached prefix features.
    """

    config_class = RLTConfig
    name = "rlt"
    weights_name = "rlt_model.pt"

    def __init__(
        self,
        config: RLTConfig,
        *,
        base_policy: nn.Module | None = None,
        load_base: bool = True,
        load_token_checkpoint: bool = True,
        **kwargs,
    ):
        super().__init__(config)
        if kwargs.get("device") is not None:
            config.device = str(kwargs["device"])
        if not config.base_policy_path and base_policy is None:
            raise ValueError(
                "RLT needs base_policy_path pointing to the complete ACP + RTC PI0.5 checkpoint."
            )
        self.base_policy = base_policy
        if load_base and self.base_policy is None:
            from lerobot.policies.pi05.configuration_pi05 import PI05Config
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy

            base_config = PreTrainedConfig.from_pretrained(config.base_policy_path)
            if not isinstance(base_config, PI05Config):
                raise ValueError("RLT base_policy_path must identify a native PI0.5 checkpoint.")
            base_config.device = config.device
            # Prefix capture needs eager Python calls; checkpoint training may
            # have enabled compilation, which must not carry into this adapter.
            base_config.compile_model = False
            self.base_policy = PI05Policy.from_pretrained(
                config.base_policy_path, config=base_config, strict=True, device=config.device
            )
        if self.base_policy is not None:
            self._sync_base_config()
            self.base_policy.requires_grad_(False)
            self.base_policy.eval()
        self._validate_resolved_config()
        self.token_module = self._make_token()
        if load_token_checkpoint and config.token_checkpoint:
            self.token_module = RLTokenModule.load(config.token_checkpoint)
            architecture = self.token_module.architecture()
            if architecture["token_dim"] != config.token_dim:
                raise ValueError(
                    f"Token checkpoint width {architecture['token_dim']} does not match "
                    f"PI0.5 prefix width {config.token_dim}."
                )
            for source, target in (
                ("latent_dim", "latent_dim"),
                ("nhead", "token_heads"),
                ("num_enc_layers", "token_encoder_layers"),
                ("num_dec_layers", "token_decoder_layers"),
                ("ff_dim", "token_ff_dim"),
                ("num_rl_tokens", "num_rl_tokens"),
            ):
                setattr(config, target, architecture[source])
            config.token_ready = True
        state_dim = config.latent_dim + config.proprio_dim
        action_dim = config.output_features[ACTION].shape[0]
        self.actor = ResidualChunkActor(
            state_dim, config.chunk_size, action_dim, config.actor_hidden_dim, config.residual_scale
        )
        self.critic = TwinChunkCritic(state_dim, config.chunk_size, action_dim, config.critic_hidden_dim)
        self._action_queue = deque()
        self._inference_lock = RLock()
        self.to(config.device)

    def _make_token(self):
        cfg = self.config
        return RLTokenModule(
            cfg.token_dim,
            cfg.latent_dim,
            cfg.token_heads,
            cfg.token_encoder_layers,
            cfg.token_decoder_layers,
            cfg.token_ff_dim,
            cfg.num_rl_tokens,
        )

    def _sync_base_config(self):
        cfg, base = self.config, self.base_policy.config
        if getattr(base, "type", None) != "pi05":
            raise ValueError("RLT only supports this repository's PI0.5 policy.")
        if getattr(base, "compile_model", False):
            raise ValueError("RLT needs an uncompiled base PI0.5 policy for single-pass prefix capture.")
        for key in ("input_features", "output_features"):
            supplied, expected = getattr(cfg, key), getattr(base, key)
            if supplied and (supplied != expected or list(supplied) != list(expected)):
                raise ValueError(
                    f"RLT {key} (including camera order) must exactly match its base checkpoint."
                )
            setattr(cfg, key, deepcopy(expected))
        for key in ("chunk_size", "n_action_steps"):
            expected = getattr(base, key)
            if getattr(cfg, key) is not None and getattr(cfg, key) != expected:
                raise ValueError(f"RLT {key} must equal base checkpoint value {expected}.")
            setattr(cfg, key, expected)
        proprio_dim = base.robot_state_feature.shape[0] if base.robot_state_feature else 0
        if cfg.proprio_dim is not None and cfg.proprio_dim != proprio_dim:
            raise ValueError("RLT proprio_dim must equal the unpadded base observation.state width.")
        cfg.proprio_dim = proprio_dim
        cfg.normalization_mapping = deepcopy(base.normalization_mapping)
        cfg.rtc_training_max_delay = base.rtc_training_max_delay
        cfg.rtc_config = deepcopy(base.rtc_config)
        language_config = self.base_policy.model.paligemma_with_expert.paligemma.config.text_config
        token_dim = language_config.hidden_size
        if cfg.token_dim is not None and cfg.token_dim != token_dim:
            raise ValueError(f"Configured token_dim={cfg.token_dim} differs from base VLA width {token_dim}.")
        cfg.token_dim = token_dim

    def _validate_resolved_config(self):
        cfg = self.config
        if cfg.token_dim is None or cfg.chunk_size is None or cfg.proprio_dim is None:
            raise ValueError(
                "load_base=False needs resolved token_dim, chunk_size and proprio_dim from the cache."
            )
        cfg.validate_features()
        if cfg.rtc_prefix_steps is not None and not 0 <= cfg.rtc_prefix_steps <= cfg.rtc_training_max_delay:
            raise ValueError("RLT rtc_prefix_steps exceeds the base checkpoint's trained RTC delay.")
        if cfg.n_action_steps is None:
            cfg.n_action_steps = cfg.chunk_size

    def train(self, mode: bool = True):
        super().train(mode)
        if self.base_policy is not None:
            self.base_policy.eval()
        return self

    def reset(self):
        self._action_queue.clear()
        if self.base_policy is not None:
            self.base_policy.reset()

    def get_optim_params(self):
        return [
            p
            for module in (self.token_module, self.actor, self.critic)
            for p in module.parameters()
            if p.requires_grad
        ]

    @torch.no_grad()
    def extract_prefix_features(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return ordered, pooled FINAL PI0.5 prefix states [B,pool_size,E] and mask."""
        with self._inference_lock:
            return self._extract_prefix_features(batch)

    def _extract_prefix_features(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if self.base_policy is None:
            raise RuntimeError(
                "RLT was loaded with load_base=False; feature extraction needs the base checkpoint."
            )
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        self.base_policy.eval()
        base = self.base_policy
        # Require every saved camera; silently reordering missing camera slots
        # would make cached RLT states disagree with deployment.
        missing = [key for key in base.config.image_features if key not in batch]
        if missing:
            raise ValueError(f"RLT observations are missing base checkpoint camera features: {missing}")
        images, image_masks = base._preprocess_images(batch)
        embedded, padding, attention = base.model.embed_prefix(
            images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        )
        attention_4d = base.model._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
        language_model = base.model.paligemma_with_expert.paligemma.language_model
        language_model.config._attn_implementation = "eager"
        outputs, _ = base.model.paligemma_with_expert.forward(
            attention_mask=attention_4d,
            position_ids=padding.long().cumsum(1) - 1,
            past_key_values=None,
            inputs_embeds=[embedded, None],
            use_cache=False,
        )
        tokens, mask = pool_prefix_tokens(outputs[0], padding, self.config.token_pool_size)
        if tokens.shape[-1] != self.config.token_dim:
            raise ValueError("Extracted PI0.5 features do not match the RL-token checkpoint width.")
        return tokens, mask

    @torch.no_grad()
    def encode_observation(self, batch: dict[str, Tensor]) -> Tensor:
        self.token_module.eval()
        return self.token_module.encode(*self.extract_prefix_features(batch))

    @torch.no_grad()
    def reference_and_features(self, batch: dict[str, Tensor], **rtc_kwargs) -> tuple[Tensor, Tensor, Tensor]:
        """Sample the reference and capture pooled tokens/mask in one frozen VLA pass."""
        if self.base_policy is None:
            raise RuntimeError("reference_and_features requires loading the frozen base policy.")
        missing = [key for key in self.base_policy.config.image_features if key not in batch]
        if missing:
            raise ValueError(f"RLT observations are missing base checkpoint camera features: {missing}")
        # Native PI0.5 calls these functions directly, bypassing nn.Module hooks.
        # Capture the final prefix output from that SAME sampling pass. The lock
        # serializes wrapper inference; try/finally restores both methods even if
        # denoising fails. Do not independently call base_policy concurrently.
        with self._inference_lock:
            core = self.base_policy.model
            backbone = core.paligemma_with_expert
            original_embed, original_forward = core.embed_prefix, backbone.forward
            prior_embed = core.__dict__.get("embed_prefix")
            prior_forward = backbone.__dict__.get("forward")
            captured = {}

            def capture_embed(*args, **kwargs):
                result = original_embed(*args, **kwargs)
                captured["padding"] = result[1]
                return result

            def capture_forward(*args, **kwargs):
                result = original_forward(*args, **kwargs)
                if result[0][0] is not None:
                    captured["tokens"], captured["mask"] = pool_prefix_tokens(
                        result[0][0], captured["padding"], self.config.token_pool_size
                    )
                return result

            core.embed_prefix, backbone.forward = capture_embed, capture_forward
            try:
                reference = self.base_policy.predict_action_chunk(batch, **rtc_kwargs)
            finally:
                if prior_embed is None:
                    del core.embed_prefix
                else:
                    core.embed_prefix = prior_embed
                if prior_forward is None:
                    del backbone.forward
                else:
                    backbone.forward = prior_forward
            if "tokens" not in captured:
                raise RuntimeError("The base PI0.5 sampler did not expose its final VLA prefix features.")
            return reference, captured["tokens"], captured["mask"]

    @torch.no_grad()
    def reference_and_state(self, batch: dict[str, Tensor], **rtc_kwargs) -> tuple[Tensor, Tensor]:
        reference, tokens, mask = self.reference_and_features(batch, **rtc_kwargs)
        self.token_module.eval()
        latent = self.token_module.encode(tokens, mask)
        if self.config.proprio_dim:
            state = batch[OBS_STATE]
            if state.shape != (len(latent), self.config.proprio_dim):
                raise ValueError("RLT observation.state must be the normalized, unpadded [B,P] vector.")
            latent = torch.cat((latent, state.to(device=latent.device, dtype=latent.dtype)), dim=-1)
        return reference, latent

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if not self.config.token_ready or not self.config.actor_ready:
            raise RuntimeError(
                "RLT deployment requires a trained token and completed actor-critic checkpoint."
            )
        self.eval()
        delay = 0
        if kwargs.get("training_time_rtc", False):
            delay = kwargs.get("inference_delay") or 0
            if self.config.rtc_prefix_steps is not None and delay not in (0, self.config.rtc_prefix_steps):
                raise ValueError("RTC delay differs from the delay used to train this RLT actor.")
        noise_std = kwargs.pop("exploration_std", self.config.exploration_std)
        reference, state = self.reference_and_state(batch, **kwargs)
        action = self.actor.sample(state, reference, noise_std=noise_std, prefix_lengths=delay)
        if delay:
            prefix = kwargs.get("rtc_action_prefix")
            if prefix is not None and prefix.ndim == 2:
                prefix = prefix.unsqueeze(0)
            if (
                prefix is None
                or prefix.ndim != 3
                or prefix.shape[0] != len(action)
                or prefix.shape[1] < delay
            ):
                raise ValueError("Training-Time RTC requires the exact normalized clean prefix.")
            if prefix.shape[-1] != action.shape[-1]:
                raise ValueError("RLT RTC prefix action width must match the unpadded robot action width.")
            action[:, :delay] = prefix[:, :delay].to(action)
        if not torch.isfinite(action).all():
            raise ValueError("RLT produced non-finite actions.")
        return action

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if kwargs.get("training_time_rtc") or (self.config.rtc_config and self.config.rtc_config.enabled):
            raise ValueError("Use predict_action_chunk with the asynchronous RTC client.")
        if not self._action_queue:
            self._action_queue.extend(
                self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps].transpose(0, 1)
            )
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Token reconstruction only. Actor-critic training uses the RLT trainer."""
        tokens, mask = self.extract_prefix_features(batch)
        loss = self.token_module.reconstruction_loss(tokens, mask)
        return loss, {"token_reconstruction_loss": loss.detach().item()}

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config._save_pretrained(save_directory)
        torch.save(
            {
                "format_version": 1,
                "token_module": self.token_module.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            save_directory / self.weights_name,
        )

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, load_base=True, device=None, **kwargs):
        path = Path(pretrained_name_or_path)
        if not path.is_dir():
            raise ValueError("Native RLT currently loads local exported checkpoint directories.")
        if config is None:
            config = PreTrainedConfig.from_pretrained(path)
        if not isinstance(config, RLTConfig):
            raise ValueError("Expected an RLT config.json.")
        if device is not None:
            config.device = str(device)
        policy = cls(config, load_base=load_base, load_token_checkpoint=False, **kwargs)
        data = torch.load(path / cls.weights_name, map_location="cpu", weights_only=True)
        if data.get("format_version") != 1:
            raise ValueError("Unsupported native RLT checkpoint format.")
        for name in ("token_module", "actor", "critic"):
            getattr(policy, name).load_state_dict(data[name], strict=True)
        policy.eval()
        return policy
