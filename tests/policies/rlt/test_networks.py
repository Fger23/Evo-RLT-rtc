"""CPU contracts for RLT, independent of transformers/robot hardware."""

import pytest
import torch

from lerobot.policies.rlt.networks import (
    ResidualChunkActor,
    RLTokenModule,
    TwinChunkCritic,
    pool_prefix_tokens,
)


def test_token_mask_invariance_and_reconstruction_gradients(tmp_path):
    torch.manual_seed(7)
    token = RLTokenModule(
        12, latent_dim=8, nhead=2, num_enc_layers=1, num_dec_layers=1, ff_dim=16, num_rl_tokens=2
    )
    features = torch.randn(2, 5, 12, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    changed = features.detach().clone()
    changed[~mask] = 1e7
    torch.testing.assert_close(token.encode(features, mask), token.encode(changed, mask))
    loss = token.reconstruction_loss(features, mask)
    loss.backward()
    assert features.grad is None  # Frozen VLA boundary.
    assert token.rl_token_embed.grad.abs().sum() > 0
    token.save(tmp_path / "token.pt")
    loaded = RLTokenModule.load(tmp_path / "token.pt")
    torch.testing.assert_close(loaded.encode(features, mask), token.encode(features, mask))
    with pytest.raises(ValueError, match="Expected VLA tokens"):
        loaded.encode(torch.zeros(2, 5, 13))


def test_actor_identity_bounded_residual_and_prefix_gradient():
    actor = ResidualChunkActor(5, 6, 3, hidden_dim=12, residual_scale=0.15)
    state, ref = torch.randn(2, 5), torch.randn(2, 6, 3)
    assert torch.equal(actor(state, ref), ref)
    with torch.no_grad():
        actor.net[-1].bias.fill_(10)
    action = actor.sample(state, ref, noise_std=0.3, prefix_lengths=torch.tensor([0, 3]))
    assert torch.equal(action[1, :3], ref[1, :3])
    assert (action - ref).abs().max() <= 0.150001
    assert not torch.equal(action[0, 0], ref[0, 0])
    actor.zero_grad()
    actor(state, ref, prefix_lengths=3)[:, :3].sum().backward()
    assert actor.net[-1].weight.grad.abs().sum() == 0
    with pytest.raises(ValueError, match="prefix lengths"):
        actor(state, ref, prefix_lengths=6)


def test_critic_ignores_unexecuted_actions():
    critic = TwinChunkCritic(4, 6, 2, hidden_dim=12)
    state, actions = torch.randn(2, 4), torch.randn(2, 6, 2)
    mask = torch.arange(6)[None] < torch.tensor([2, 4])[:, None]
    changed = actions.clone()
    changed[~mask] = 1e8
    a, b = critic(state, actions, mask), critic(state, changed, mask)
    for expected, actual in zip(a, b, strict=True):
        torch.testing.assert_close(expected, actual)


def test_pooling_removes_padding_and_preserves_order():
    values = torch.tensor([[[1.0], [1000.0], [3.0], [5.0]]])
    result, mask = pool_prefix_tokens(values, torch.tensor([[True, False, True, True]]), 3)
    assert result.flatten().tolist() == [1.0, 3.0, 5.0]
    assert mask.all()
    with pytest.raises(ValueError, match="valid token"):
        pool_prefix_tokens(values, torch.zeros(1, 4, dtype=torch.bool), 3)
