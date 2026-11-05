"""
A2C (Advantage Actor-Critic) manual para espacios de acción continuos.

Misma interfaz que CustomPPO, así train.py lo usa igual:

    agent = CustomA2C(state_dim, action_dim)
    action = agent.select_action(obs)
    agent.store_transition(reward, terminated, truncated, next_obs)
    agent.update(last_state=obs)
    agent.save(path) / agent.load(path)

Diferencias con PPO:
  - UNA sola actualización de gradiente por rollout (PPO hace K épocas).
  - Sin ratio ni clipping: la pérdida es  -logprob(a|s) * ventaja.
  - No necesita "política vieja": los datos se recolectan con la misma red
    que se actualiza (por eso es on-policy puro).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reutilizamos la red actor-crítico y el buffer de PPO (misma arquitectura,
# así la comparación PPO vs A2C es justa).
from agents.custom_ppo import ActorCritic, RolloutBuffer


class CustomA2C:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=7e-4,
        gamma=0.99,
        gae_lambda=1.0,          # 1.0 = A2C clásico (retornos n-step); <1 = GAE
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        normalize_advantages=True,
        action_std_init=1.0,
        device=None,
        **_ignored,              # train.py puede pasar k_epochs / mini_batch_size: A2C no los usa
    ):
        # train.py pasa None cuando no se indica el hiperparámetro
        self.gamma = 0.99 if gamma is None else gamma
        lr = 7e-4 if lr is None else lr

        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

    # ------------------------------------------------------------------ #
    # Interacción con el entorno
    # ------------------------------------------------------------------ #
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)

            if deterministic:
                return self.policy.actor_mean(state_t).cpu().numpy().flatten()

            action, _, value = self.policy.act(state_t)

        self.buffer.states.append(state_t.squeeze(0))
        self.buffer.actions.append(action.squeeze(0))
        self.buffer.state_values.append(value.squeeze(0))
        return action.cpu().numpy().flatten()

    def store_transition(self, reward, terminated, truncated=False, next_state=None):
        # Si el episodio se corta por tiempo (truncated), se hace bootstrap con V(next_state)
        if truncated and not terminated and next_state is not None:
            with torch.no_grad():
                ns = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).reshape(1, -1)
                reward = reward + self.gamma * self.policy.critic(ns).item()

        self.buffer.rewards.append(float(reward))
        self.buffer.is_terminals.append(bool(terminated or truncated))

    # ------------------------------------------------------------------ #
    # Ventajas
    # ------------------------------------------------------------------ #
    def _compute_advantages(self, rewards, values, dones, last_value):
        n = len(rewards)
        advantages = torch.zeros(n, device=self.device)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------ #
    # Actualización A2C
    # ------------------------------------------------------------------ #
    def update(self, last_state=None):
        n = len(self.buffer)
        if n == 0:
            return None

        states = torch.stack(self.buffer.states).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        old_values = torch.stack(self.buffer.state_values).to(self.device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.buffer.is_terminals, dtype=torch.float32, device=self.device)

        # Bootstrap si el rollout terminó a mitad de un episodio
        last_value = 0.0
        if last_state is not None and dones[-1].item() == 0.0:
            with torch.no_grad():
                ls = torch.as_tensor(last_state, dtype=torch.float32, device=self.device).reshape(1, -1)
                last_value = self.policy.critic(ls).item()

        advantages, returns = self._compute_advantages(rewards, old_values, dones, last_value)
        if self.normalize_advantages and n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Re-evaluar con la red actual (aquí sí hay gradiente)
        logprobs, values, entropy = self.policy.evaluate(states, actions)

        policy_loss = -(logprobs * advantages.detach()).mean()
        value_loss = self.MseLoss(values, returns.detach())
        entropy_bonus = entropy.mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.buffer.clear()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_bonus.item()),
        }

    # ------------------------------------------------------------------ #
    # Guardado / carga
    # ------------------------------------------------------------------ #
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
