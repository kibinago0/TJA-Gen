from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from taiko_ai.environment.actions import ACTION_COUNT
from taiko_ai.environment.state import OBSERVATION_SIZE


class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy = nn.Linear(64, ACTION_COUNT)
        self.value = nn.Linear(64, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(observations)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: int
    log_probability: float
    reward: float
    value: float
    done: bool


class PPOAgent:
    def __init__(
        self,
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        state = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state)
            distribution = Categorical(logits=logits)
            action = (
                torch.argmax(logits, dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_probability = distribution.log_prob(action)
        return (
            int(action.item()),
            float(log_probability.item()),
            float(value.item()),
        )

    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        return self.act(observation, deterministic=deterministic)[0]

    def update(
        self,
        transitions: list[Transition],
        *,
        epochs: int = 4,
        minibatch_size: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
    ) -> float:
        if not transitions:
            raise ValueError("PPO update requires at least one transition")

        advantages = np.zeros(len(transitions), dtype=np.float32)
        returns = np.zeros(len(transitions), dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            item = transitions[index]
            mask = 0.0 if item.done else 1.0
            delta = item.reward + gamma * next_value * mask - item.value
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[index] = gae
            returns[index] = gae + item.value
            next_value = item.value
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        observations = torch.as_tensor(
            np.stack([item.observation for item in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in transitions],
            dtype=torch.long,
            device=self.device,
        )
        old_log_probabilities = torch.as_tensor(
            [item.log_probability for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        advantage_tensor = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        return_tensor = torch.as_tensor(
            returns, dtype=torch.float32, device=self.device
        )

        total_loss = 0.0
        updates = 0
        count = len(transitions)
        for _ in range(epochs):
            indices = torch.randperm(count, device=self.device)
            for batch_start in range(0, count, minibatch_size):
                batch = indices[batch_start : batch_start + minibatch_size]
                logits, values = self.model(observations[batch])
                distribution = Categorical(logits=logits)
                log_probabilities = distribution.log_prob(actions[batch])
                ratio = torch.exp(log_probabilities - old_log_probabilities[batch])
                unclipped = ratio * advantage_tensor[batch]
                clipped = torch.clamp(
                    ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
                ) * advantage_tensor[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (values - return_tensor[batch]).square().mean()
                entropy_loss = distribution.entropy().mean()
                loss = policy_loss + value_loss * 0.5 - entropy_loss * 0.01

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss += float(loss.detach().item())
                updates += 1
        return total_loss / max(updates, 1)

    def save(
        self,
        path: str | Path,
        metadata: dict[str, object] | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, object] = {
            "observation_size": OBSERVATION_SIZE,
            "action_count": ACTION_COUNT,
            "state_dict": self.model.state_dict(),
        }
        if metadata is not None:
            checkpoint["metadata"] = metadata
        torch.save(
            checkpoint,
            destination,
        )

    def load(self, path: str | Path) -> dict[str, object]:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=True
        )
        observation_size = checkpoint.get("observation_size")
        if observation_size != OBSERVATION_SIZE:
            raise ValueError(
                f"model observation size {observation_size} is incompatible with "
                f"the current size {OBSERVATION_SIZE}; retrain the model"
            )
        action_count = checkpoint.get("action_count", 3)
        if action_count != ACTION_COUNT:
            raise ValueError(
                f"model action count {action_count} is incompatible with "
                f"the current count {ACTION_COUNT}; retrain the model"
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("model metadata is invalid")
        return metadata
