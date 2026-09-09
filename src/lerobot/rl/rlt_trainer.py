"""Small-network chunk TD3 + successful-demo BC, with a frozen RL-token encoder."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F  # noqa: N812


@dataclass
class RLTTrainConfig:
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.02
    demo_bc_weight: float = 5.0
    reference_bc_weight: float = 1.0
    grad_clip: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("RLT trainer parameters must be finite")
        if self.actor_lr <= 0 or self.critic_lr <= 0 or not 0 < self.tau <= 1 or self.policy_delay < 1:
            raise ValueError("Invalid RLT optimizer/target-update parameters")
        if min(self.target_noise, self.demo_bc_weight, self.reference_bc_weight) < 0 or self.grad_clip <= 0:
            raise ValueError("Noise/BC weights must be nonnegative, grad_clip positive")


def masked_mse(prediction, target, mask):
    weights = mask.unsqueeze(-1).to(prediction.dtype).expand_as(prediction)
    return ((prediction - target).square() * weights).sum() / weights.sum().clamp_min(1)


class RLTActorCriticTrainer:
    def __init__(self, policy, config: RLTTrainConfig | None = None):
        self.policy = policy
        self.config = config or RLTTrainConfig()
        self.policy.token_module.requires_grad_(False).eval()
        self.target_actor = copy.deepcopy(policy.actor).requires_grad_(False).eval()
        self.target_critic = copy.deepcopy(policy.critic).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=self.config.critic_lr)
        self.step = 0

    @property
    def device(self):
        return next(self.policy.actor.parameters()).device

    def prepare_batch(self, batch):
        return {
            key: value.to(self.device).float() if value.is_floating_point() else value.to(self.device)
            for key, value in batch.items()
        }

    @torch.no_grad()
    def states(self, batch):
        self.policy.token_module.eval()
        current = self.policy.token_module.encode(batch["tokens"], batch["token_mask"])
        following = self.policy.token_module.encode(batch["next_tokens"], batch["next_token_mask"])
        return torch.cat((current, batch["proprio"]), -1), torch.cat((following, batch["next_proprio"]), -1)

    @torch.no_grad()
    def td_target(self, batch, next_state):
        actions = self.target_actor.sample(
            next_state,
            batch["next_ref_actions"],
            noise_std=self.config.target_noise,
            prefix_lengths=batch["next_prefix_length"],
        )
        q1, q2 = self.target_critic(next_state, actions, action_mask=batch["next_action_mask"])
        return batch["reward"].unsqueeze(-1) + batch["discount"].unsqueeze(-1) * torch.minimum(q1, q2)

    def train_step(self, batch):
        batch = self.prepare_batch(batch)
        state, next_state = self.states(batch)
        target = self.td_target(batch, next_state)
        self.policy.critic.train()
        q1, q2 = self.policy.critic(state, batch["actions"], action_mask=batch["action_mask"])
        critic_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Nonfinite critic loss; inspect replay normalization")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()
        self.step += 1
        metrics = {"step": self.step, "critic_loss": critic_loss.item(), "target_q": target.mean().item()}
        if self.step % self.config.policy_delay == 0:
            self.policy.actor.train()
            self.policy.critic.requires_grad_(False)
            try:
                actions = self.policy.actor(
                    state, batch["ref_actions"], prefix_lengths=batch["prefix_length"]
                )
                actor_q, _ = self.policy.critic(state, actions, action_mask=batch["action_mask"])
                indices = torch.arange(actions.shape[1], device=self.device).unsqueeze(0)
                postfix = batch["action_mask"] & (indices >= batch["prefix_length"].unsqueeze(1))
                trusted = (batch["source"] == 0).unsqueeze(1) | batch["intervention_mask"].bool()
                demo_mask = postfix & trusted & batch["success"].bool().unsqueeze(1)
                demo_bc = masked_mse(actions, batch["actions"], demo_mask)
                reference_bc = masked_mse(actions, batch["ref_actions"], postfix)
                actor_loss = (
                    -actor_q.mean()
                    + self.config.demo_bc_weight * demo_bc
                    + self.config.reference_bc_weight * reference_bc
                )
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError("Nonfinite actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config.grad_clip)
                self.actor_optimizer.step()
            finally:
                self.policy.critic.requires_grad_(True)
            with torch.no_grad():
                for target_module, module in (
                    (self.target_actor, self.policy.actor),
                    (self.target_critic, self.policy.critic),
                ):
                    for target_param, param in zip(
                        target_module.parameters(), module.parameters(), strict=True
                    ):
                        target_param.lerp_(param, self.config.tau)
            metrics.update(
                actor_loss=actor_loss.item(), demo_bc=demo_bc.item(), reference_bc=reference_bc.item()
            )
        return metrics

    @torch.no_grad()
    def evaluate_batch(self, batch):
        batch = self.prepare_batch(batch)
        state, next_state = self.states(batch)
        target = self.td_target(batch, next_state)
        q1, q2 = self.policy.critic(state, batch["actions"], action_mask=batch["action_mask"])
        actions = self.policy.actor(state, batch["ref_actions"], prefix_lengths=batch["prefix_length"])
        return {
            "validation_td_mse": (F.mse_loss(q1, target) + F.mse_loss(q2, target)).item(),
            "validation_action_mse": masked_mse(actions, batch["actions"], batch["action_mask"]).item(),
        }

    def state_dict(self):
        return {
            "format_version": 1,
            "step": self.step,
            "config": asdict(self.config),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def load_state_dict(self, state):
        if state.get("format_version") != 1 or state["config"] != asdict(self.config):
            raise ValueError("Resume trainer configuration differs; use warm start without --resume")
        self.target_actor.load_state_dict(state["target_actor"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.step = int(state["step"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if state.get("cuda_rng_state") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state"]])
