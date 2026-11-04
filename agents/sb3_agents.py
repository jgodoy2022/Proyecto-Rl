"""
Integración de agentes de Stable-Baselines3 (PPO, A2C) con la arquitectura del proyecto.
"""

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback

# Mapa de algoritmos soportados en SB3
SB3_CLASSES = {
    "ppo": PPO,
    "a2c": A2C,
}


class MetricsCallback(BaseCallback):
    """
    Callback para conectar el bucle de entrenamiento de Stable-Baselines3
    con el logger central del proyecto (utils/metrics.py).
    """

    def __init__(self, metrics_logger, verbose=0):
        super().__init__(verbose)
        self.metrics = metrics_logger

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        info = self.locals["infos"][0]
        action = self.locals["actions"][0]

        # Registrar métricas por paso
        self.metrics.log_step(reward, info, action)

        # Si el episodio termina, registrar el resumen anual
        dones = self.locals["dones"]
        if dones[0]:
            self.metrics.end_episode()

        return True


def build_sb3_agent(algo_name, env, tensorboard_log=None, seed=None, device="cpu", **hp):
    """
    Construye e inicializa una instancia de PPO o A2C en Stable-Baselines3.
    """
    if algo_name not in SB3_CLASSES:
        raise ValueError(f"Algoritmo SB3 no soportado: {algo_name}. Opciones: {list(SB3_CLASSES.keys())}")

    algo_cls = SB3_CLASSES[algo_name]

    kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "tensorboard_log": tensorboard_log,
        "seed": seed,
        "device": device,
    }

    if "lr" in hp:
        kwargs["learning_rate"] = hp["lr"]
    if "gamma" in hp:
        kwargs["gamma"] = hp["gamma"]

    # Hiperparámetros específicos
    if "k_epochs" in hp and algo_name == "ppo":
        kwargs["n_epochs"] = hp["k_epochs"]
    if "mini_batch_size" in hp:
        kwargs["batch_size"] = hp["mini_batch_size"]

    return algo_cls(**kwargs)


def load_sb3_agent(algo_name, path, device="cpu"):
    """
    Carga un modelo de Stable-Baselines3 (.zip) para evaluación.
    """
    if algo_name not in SB3_CLASSES:
        raise ValueError(f"Algoritmo SB3 no soportado: {algo_name}")

    algo_cls = SB3_CLASSES[algo_name]
    return algo_cls.load(path, device=device)