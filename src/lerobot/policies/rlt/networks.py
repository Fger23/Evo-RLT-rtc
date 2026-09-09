"""Torch-only RL-token, bounded residual actor, and twin chunk critics.

Adapted conceptually from MINT-SJTU/Evo-RLT (474c669): an RL-token
autoencoder, reference-conditioned actor, and twin Q functions. This integration
uses a projected token bottleneck and bounded, zero-initialized action residuals
to retain the existing PI0.5 policy at the start of actor training.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


def pool_prefix_tokens(tokens: Tensor, valid_mask: Tensor, pool_size: int) -> tuple[Tensor, Tensor]:
    """Pool ordered valid prefix embeddings identically for caching and deployment."""
    if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2] or pool_size < 1:
        raise ValueError("Expected tokens [B,M,E], mask [B,M], and a positive pool size.")
    pooled = []
    for sample, valid in zip(tokens, valid_mask.bool(), strict=True):
        selected = sample[valid]
        if not len(selected):
            raise ValueError("A VLA prefix must contain at least one valid token.")
        pooled.append(F.adaptive_avg_pool1d(selected.T.unsqueeze(0).float(), pool_size)[0].T)
    result = torch.stack(pooled)
    return result, torch.ones(result.shape[:2], dtype=torch.bool, device=result.device)


def _positions(length: int, width: int, device: torch.device) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000) / width))
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies[: width // 2])
    return result


class RLTokenModule(nn.Module):
    """Masked transformer autoencoder with learned RL tokens and causal reconstruction."""

    def __init__(
        self,
        token_dim: int,
        latent_dim: int = 128,
        nhead: int = 4,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        ff_dim: int = 512,
        num_rl_tokens: int = 4,
    ):
        super().__init__()
        if min(token_dim, latent_dim, nhead, num_enc_layers, num_dec_layers, ff_dim, num_rl_tokens) < 1:
            raise ValueError("All token architecture dimensions must be positive.")
        if latent_dim % nhead:
            raise ValueError("latent_dim must be divisible by nhead.")
        self._architecture = {
            "token_dim": token_dim,
            "latent_dim": latent_dim,
            "nhead": nhead,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "ff_dim": ff_dim,
            "num_rl_tokens": num_rl_tokens,
        }
        self.token_dim, self.latent_dim, self.num_rl_tokens = token_dim, latent_dim, num_rl_tokens
        self.input_proj = nn.Linear(token_dim, latent_dim)
        self.rl_token_embed = nn.Parameter(torch.randn(1, num_rl_tokens, latent_dim) * 0.02)
        self.bos = nn.Parameter(torch.zeros(1, 1, latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            latent_dim, nhead, ff_dim, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_enc_layers, enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(
            latent_dim, nhead, ff_dim, dropout=0.0, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_dec_layers)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, token_dim)

    def architecture(self) -> dict:
        return dict(self._architecture)

    def _check(self, tokens: Tensor, valid_mask: Tensor | None) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected VLA tokens [B,M,{self.token_dim}], got {tuple(tokens.shape)}.")
        if valid_mask is None:
            valid_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if valid_mask.shape != tokens.shape[:2] or not valid_mask.bool().any(dim=1).all():
            raise ValueError("Token mask must be [B,M] and include valid tokens in every sample.")
        return valid_mask.bool()

    def encode_multi(self, tokens: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        valid_mask = self._check(tokens, valid_mask)
        tokens = tokens.detach().float().masked_fill(~valid_mask.unsqueeze(-1), 0)
        x = self.input_proj(tokens) + _positions(tokens.shape[1], self.latent_dim, tokens.device)
        x = torch.cat((x, self.rl_token_embed.expand(len(x), -1, -1)), dim=1)
        padding = torch.cat(
            (~valid_mask, torch.zeros(len(x), self.num_rl_tokens, dtype=torch.bool, device=x.device)), dim=1
        )
        return self.latent_norm(self.encoder(x, src_key_padding_mask=padding)[:, -self.num_rl_tokens :])

    def encode(self, tokens: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        return self.encode_multi(tokens, valid_mask).mean(dim=1)

    def reconstruction_loss(self, tokens: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        valid_mask = self._check(tokens, valid_mask)
        target = tokens.detach().float().masked_fill(~valid_mask.unsqueeze(-1), 0)
        memory = self.encode_multi(target, valid_mask)
        teacher = self.input_proj(target)
        shifted = torch.cat((self.bos.expand(len(target), -1, -1), teacher[:, :-1]), dim=1)
        shifted = shifted + _positions(target.shape[1], self.latent_dim, target.device)
        causal = torch.ones(target.shape[1], target.shape[1], device=target.device, dtype=torch.bool).triu(1)
        # Decoder input at i is target i-1, so its padding mask must also be shifted.
        padding = torch.cat(
            (torch.zeros(len(target), 1, device=target.device, dtype=torch.bool), ~valid_mask[:, :-1]), dim=1
        )
        predicted = self.out_proj(
            self.decoder(shifted, memory, tgt_mask=causal, tgt_key_padding_mask=padding)
        )
        errors = (predicted - target).square().mean(dim=-1)
        return (errors * valid_mask).sum() / valid_mask.sum()

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"format_version": 1, "config": self.architecture(), "state_dict": self.state_dict()}, path
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> RLTokenModule:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format_version") != 1:
            raise ValueError("Unsupported RL-token checkpoint format.")
        module = cls(**data["config"])
        module.load_state_dict(data["state_dict"], strict=True)
        return module.to(device)


def prefix_action_mask(reference: Tensor, prefix_lengths: int | Tensor = 0) -> Tensor:
    """Return [B,H,1] mask for action positions which may be changed."""
    delay = torch.as_tensor(prefix_lengths, device=reference.device)
    if delay.dtype == torch.bool or delay.is_floating_point():
        raise ValueError("RTC prefix lengths must be integers.")
    if delay.ndim == 0:
        delay = delay.expand(reference.shape[0])
    if delay.shape != (reference.shape[0],) or (delay < 0).any() or (delay >= reference.shape[1]).any():
        raise ValueError("RTC prefix lengths must have shape [B] and satisfy 0 <= d < H.")
    return (torch.arange(reference.shape[1], device=reference.device)[None] >= delay[:, None]).unsqueeze(-1)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ResidualChunkActor(nn.Module):
    """Full-horizon reference + bounded residual; initially exactly the reference."""

    def __init__(
        self,
        state_dim: int,
        chunk_size: int,
        action_dim: int,
        hidden_dim: int = 256,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be positive.")
        self.state_dim, self.chunk_size, self.action_dim = state_dim, chunk_size, action_dim
        self.residual_scale = residual_scale
        self.net = _mlp(state_dim + chunk_size * action_dim, hidden_dim, chunk_size * action_dim)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _raw(self, state: Tensor, reference: Tensor) -> Tensor:
        if reference.shape != (len(state), self.chunk_size, self.action_dim):
            raise ValueError(f"Expected reference [B,{self.chunk_size},{self.action_dim}].")
        if state.shape != (len(reference), self.state_dim):
            raise ValueError(f"Expected state [B,{self.state_dim}].")
        return self.net(torch.cat((state.float(), reference.float().flatten(1)), dim=-1)).view_as(reference)

    def forward(self, state: Tensor, reference: Tensor, prefix_lengths: int | Tensor = 0) -> Tensor:
        return self.sample(state, reference, noise_std=0.0, prefix_lengths=prefix_lengths)

    def sample(
        self, state: Tensor, reference: Tensor, noise_std: float = 0.0, prefix_lengths: int | Tensor = 0
    ) -> Tensor:
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be non-negative.")
        raw = self._raw(state, reference)
        if noise_std:
            raw = raw + noise_std * torch.randn_like(raw)
        residual = self.residual_scale * torch.tanh(raw)
        # torch.where copies the clean prefix bit-for-bit, including signed zero.
        return torch.where(prefix_action_mask(reference, prefix_lengths), reference + residual, reference)


class TwinChunkCritic(nn.Module):
    """Twin Q(s, executed chunk, mask), including the real execution length."""

    def __init__(self, state_dim: int, chunk_size: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim, self.chunk_size, self.action_dim = state_dim, chunk_size, action_dim
        input_dim = state_dim + chunk_size * action_dim + chunk_size
        self.q1 = _mlp(input_dim, hidden_dim, 1)
        self.q2 = _mlp(input_dim, hidden_dim, 1)

    def forward(
        self, state: Tensor, actions: Tensor, action_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if actions.shape != (len(state), self.chunk_size, self.action_dim):
            raise ValueError("Critic actions must have shape [B,H,A].")
        if action_mask is None:
            action_mask = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
        if action_mask.ndim == 3 and action_mask.shape[-1] == 1:
            action_mask = action_mask.squeeze(-1)
        if action_mask.shape != actions.shape[:2]:
            raise ValueError("Critic action_mask must have shape [B,H].")
        action_mask = action_mask.bool()
        visible = actions.float().masked_fill(~action_mask.unsqueeze(-1), 0)
        x = torch.cat((state.float(), visible.flatten(1), action_mask.float()), dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, state: Tensor, actions: Tensor, action_mask: Tensor | None = None) -> Tensor:
        return torch.minimum(*self(state, actions, action_mask))
