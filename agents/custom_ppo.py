"""
PPO (Proximal Policy Optimization) manual para espacios de acción continuos.

Uso esperado desde train.py:

    from custom_ppo import CustomPPO

    agent = CustomPPO(state_dim, action_dim)
    state, _ = env.reset()
    for step in range(1, total_steps + 1):
        action = agent.select_action(state)                       # np.ndarray (action_dim,)
        env_action = np.clip(action, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(env_action)
        agent.store_transition(reward, terminated, truncated, next_state)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
        if step % update_every == 0:
            agent.update(last_state=state)

IMPORTANTE: la acción devuelta por select_action NO está recortada (es la muestra
de la gaussiana, y esa es la que se usa para calcular el log-prob). Recórtala
solo al enviarla al entorno, como en el ejemplo.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def _init_layer(layer, gain=np.sqrt(2)):
    """Inicialización ortogonal (práctica estándar en PPO)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=1.0):
        super().__init__()

        # Crítico: V(s) -> escalar
        self.critic = nn.Sequential(
            _init_layer(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            _init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            _init_layer(nn.Linear(64, 1), gain=1.0),
        )

        # Actor: media de la acción continua
        self.actor_mean = nn.Sequential(
            _init_layer(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            _init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            _init_layer(nn.Linear(64, action_dim), gain=0.01),
        )

        # log(std) entrenable, independiente del estado (exploración)
        self.actor_log_std = nn.Parameter(
            torch.full((1, action_dim), float(np.log(action_std_init)))
        )

    def _dist(self, state):
        mean = self.actor_mean(state)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, state):
        """state: (B, state_dim). Devuelve action (B, A), logprob (B,), value (B,)."""
        dist = self._dist(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_value = self.critic(state).squeeze(-1)
        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action):
        """Re-evalúa (state, action) guardados. Devuelve logprobs (B,), values (B,), entropy (B,)."""
        dist = self._dist(state)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        # squeeze(-1) es clave: (B,1) vs (B,) en MSELoss haría broadcasting a (B,B)
        state_values = self.critic(state).squeeze(-1)
        return action_logprobs, state_values, dist_entropy


class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.logprobs, self.state_values, self.is_terminals = [], [], []

    def clear(self):
        del self.states[:], self.actions[:], self.rewards[:]
        del self.logprobs[:], self.state_values[:], self.is_terminals[:]

    def __len__(self):
        return len(self.rewards)


class CustomPPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        K_epochs=80,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        mini_batch_size=None,   # None = batch completo en cada época
        action_std_init=1.0,
        device=None,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.mini_batch_size = mini_batch_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        self.optimizer = optim.Adam([
            {"params": self.policy.actor_mean.parameters(), "lr": lr_actor},
            {"params": [self.policy.actor_log_std], "lr": lr_actor},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ])

        # Política "vieja": la que recolecta datos durante el rollout
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    # ------------------------------------------------------------------ #
    # Interacción con el entorno
    # ------------------------------------------------------------------ #
    def select_action(self, state, deterministic=False):
        """
        Entrenamiento (deterministic=False): muestrea, guarda en el buffer y devuelve la acción.
        Evaluación (deterministic=True): devuelve la media, sin guardar nada.
        """
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)

            if deterministic:
                return self.policy_old.actor_mean(state_t).cpu().numpy().flatten()

            action, logprob, value = self.policy_old.act(state_t)

        self.buffer.states.append(state_t.squeeze(0))
        self.buffer.actions.append(action.squeeze(0))
        self.buffer.logprobs.append(logprob.squeeze(0))
        self.buffer.state_values.append(value.squeeze(0))

        return action.cpu().numpy().flatten()

    def store_transition(self, reward, terminated, truncated=False, next_state=None):
        """
        Llamar después de env.step(), una vez por cada select_action().

        - terminated: fin real del episodio (el valor futuro es 0).
        - truncated: corte por límite de tiempo. Si se pasa next_state, se hace
          bootstrap con V(next_state) para no sesgar el aprendizaje.
        """
        if truncated and not terminated and next_state is not None:
            with torch.no_grad():
                ns = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).reshape(1, -1)
                reward = reward + self.gamma * self.policy.critic(ns).item()

        self.buffer.rewards.append(float(reward))
        self.buffer.is_terminals.append(bool(terminated or truncated))

    # ------------------------------------------------------------------ #
    # Ventajas (GAE)
    # ------------------------------------------------------------------ #
    def _compute_gae(self, rewards, values, dones, last_value):
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
    # Actualización PPO
    # ------------------------------------------------------------------ #
    def update(self, last_state=None):
        """
        last_state: estado actual del entorno al terminar el rollout. Se usa para
        hacer bootstrap si la última transición no fue terminal. Si es None, se
        asume valor futuro 0 en ese punto.
        Devuelve un dict con métricas promedio (útil para logging).
        """
        n = len(self.buffer)
        if n == 0:
            return None

        # 1. Buffer -> tensores
        states = torch.stack(self.buffer.states).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device)
        old_values = torch.stack(self.buffer.state_values).to(self.device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.buffer.is_terminals, dtype=torch.float32, device=self.device)

        # 2. Ventajas GAE y retornos
        last_value = 0.0
        if last_state is not None and dones[-1].item() == 0.0:
            with torch.no_grad():
                ls = torch.as_tensor(last_state, dtype=torch.float32, device=self.device).reshape(1, -1)
                last_value = self.policy.critic(ls).item()

        advantages, returns = self._compute_gae(rewards, old_values, dones, last_value)
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Optimización durante K_epochs
        mb_size = self.mini_batch_size or n
        pl_hist, vl_hist, ent_hist = [], [], []

        for _ in range(self.K_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb_size):
                idx = perm[start:start + mb_size]

                # a. Re-evaluar acciones con la política actual
                logprobs, values, entropy = self.policy.evaluate(states[idx], actions[idx])

                # b. Ratio r_t = pi_theta / pi_theta_old
                ratios = torch.exp(logprobs - old_logprobs[idx])

                # c. Surrogate losses
                adv = advantages[idx]
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv

                # d. Pérdida total
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.MseLoss(values, returns[idx])
                entropy_bonus = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                # e. Paso de optimización
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pl_hist.append(policy_loss.item())
                vl_hist.append(value_loss.item())
                ent_hist.append(entropy_bonus.item())

        # 4. Sincronizar policy_old y limpiar el buffer
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return {
            "policy_loss": float(np.mean(pl_hist)),
            "value_loss": float(np.mean(vl_hist)),
            "entropy": float(np.mean(ent_hist)),
        }

    # ------------------------------------------------------------------ #
    # Guardado / carga
    # ------------------------------------------------------------------ #
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy_old.load_state_dict(state_dict)