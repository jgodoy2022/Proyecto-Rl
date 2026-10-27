"""
Inicialización y configuración del simulador (Sinergym + EnergyPlus).

make_env() devuelve un entorno con:
  - Acciones reescaladas a [-1, 1]: el agente siempre actúa en ese rango y el
    wrapper lo convierte a los setpoints reales (calefacción/refrigeración).
  - Observaciones opcionalmente normalizadas con media/varianza móviles
    (ObsProcessor). Las estadísticas se pueden guardar y reutilizar (congeladas)
    en evaluate.py.
  - La observación original (sin normalizar) se expone en info["raw_obs"], que
    es de donde utils/metrics.py extrae temperaturas y potencia.
"""

import os

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction

DEFAULT_ENV_ID = "Eplus-5zone-hot-continuous-v1"

class StrictClipAction(gym.ActionWrapper):
    # evita errores de validación recortando la acción a los límites exactos del simulador

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

class ObsProcessor(gym.Wrapper):
    """Normaliza observaciones (opcional) y agrega info["raw_obs"]."""

    def __init__(self, env, normalize=True, update=True, clip=10.0, epsilon=1e-8):
        super().__init__(env)
        shape = env.observation_space.shape
        self.normalize = normalize
        self.update = update          # False => estadísticas congeladas (evaluación)
        self.clip = clip
        self.epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        if normalize:
            self.observation_space = gym.spaces.Box(-clip, clip, shape=shape, dtype=np.float32)

    # -- estadísticas móviles (Welford, lote de tamaño 1) --
    def _update_stats(self, x):
        delta = x - self.mean
        tot = self.count + 1.0
        self.mean = self.mean + delta / tot
        m2 = self.var * self.count + np.square(delta) * self.count / tot
        self.var = m2 / tot
        self.count = tot

    def _process(self, obs):
        raw = np.asarray(obs, dtype=np.float64)
        if not self.normalize:
            return raw.astype(np.float32), raw
        if self.update:
            self._update_stats(raw)
        norm = (raw - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(norm, -self.clip, self.clip).astype(np.float32), raw

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, raw = self._process(obs)
        info = dict(info)
        info["raw_obs"] = raw
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs, raw = self._process(obs)
        info = dict(info)
        info["raw_obs"] = raw
        return obs, reward, terminated, truncated, info

    # -- persistencia --
    def save_stats(self, path):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    def load_stats(self, path):
        data = np.load(path)
        self.mean, self.var, self.count = data["mean"], data["var"], float(data["count"])


def make_env(env_id=DEFAULT_ENV_ID, normalize_obs=True, norm_stats_path=None, freeze_norm=False):
    """
    Abre el simulador. Cada llamada lanza una instancia de EnergyPlus, por lo que
    conviene crear un solo entorno a la vez (ver AGENTES.md, gestión de memoria).
    """
    import sinergym  # noqa: F401  (registra los entornos Eplus-* en gymnasium)

    env = gym.make(env_id)
    env = StrictClipAction(env) # filtra desbordamientos de punto flotante
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = ObsProcessor(env, normalize=normalize_obs, update=not freeze_norm)
    if norm_stats_path and os.path.exists(norm_stats_path):
        env.load_stats(norm_stats_path)
    return env


def find_processor(env):
    """Recorre la cadena de wrappers hasta encontrar el ObsProcessor."""
    current = env
    while current is not None:
        if isinstance(current, ObsProcessor):
            return current
        current = getattr(current, "env", None)
    return None


def get_obs_variables(env):
    """Nombres de las variables de observación de Sinergym (o None si no están disponibles)."""
    try:
        return list(env.get_wrapper_attr("observation_variables"))
    except Exception:
        return None


def describe_env(env):
    """Imprime dimensiones y variables: útil para verificar los nombres que usa metrics.py."""
    print(f"Observaciones: {env.observation_space.shape} | Acciones: {env.action_space.shape}")
    variables = get_obs_variables(env)
    if variables:
        print("Variables de observación:", ", ".join(variables))
    else:
        print("No se pudieron leer los nombres de las variables de observación.")