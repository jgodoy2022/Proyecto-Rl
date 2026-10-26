import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Crítico: Evalúa el estado devolviendo un escalar V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Actor: Red neuronal para la media de la acción continua
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        
        # Parámetro entrenable independiente para la desviación estándar (exploración)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def act(self, state):
        action_mean = self.actor_mean(state)
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_value = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action):
        # Utilizado durante el entrenamiento para re-evaluar estados guardados
        action_mean = self.actor_mean(state)
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards = [], [], []
        self.logprobs, self.state_values, self.is_terminals = [], [], []
    
    def clear(self):
        del self.states[:], self.actions[:], self.rewards[:]
        del self.logprobs[:], self.state_values[:], self.is_terminals[:]

class CustomPPO:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=80, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_log_std, 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob, state_value = self.policy_old.act(state)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_value)
        
        return action.numpy().flatten()

    def update(self):
        # 1. Calcular retornos de Monte Carlo o ventajas (GAE) a partir de self.buffer.rewards
        
        # 2. Convertir listas del buffer a tensores
        
        # 3. Bucle de optimización durante K_epochs:
        #    a. Evaluar acciones antiguas con self.policy.evaluate()
        #    b. Calcular el ratio de probabilidades r_t
        #    c. Calcular Surrogate Loss 1 y Surrogate Loss 2 (con torch.clamp)
        #    d. Calcular la pérdida total: -min(surr1, surr2) + 0.5 * MSELoss(values, returns) - 0.01 * entropy
        #    e. optimizer.zero_grad(), loss.backward(), optimizer.step()
        
        # 4. Actualizar policy_old con los pesos de policy y limpiar el buffer
        pass