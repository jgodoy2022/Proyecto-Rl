"""
Registro de agentes. Los imports son perezosos para que `python train.py --help`
funcione sin cargar torch ni stable-baselines3.

Interfaz que deben cumplir los agentes custom (CustomPPO ya la cumple):
    select_action(state, deterministic=False) -> np.ndarray
    store_transition(reward, terminated, truncated, next_state)
    update(last_state=None) -> dict | None
    save(path) / load(path)
"""

import importlib
from pathlib import Path

CUSTOM_AGENTS = {
    "custom_ppo": ("custom_ppo", "CustomPPO"),
    "custom_a2c": ("custom_a2c", "CustomA2C"), 
}
SB3_ALGOS = {"sb3_ppo": "ppo", "sb3_a2c": "a2c", "sb3_sac": "sac"}

AGENT_CHOICES = sorted(list(CUSTOM_AGENTS) + list(SB3_ALGOS))


def is_custom(name):
    return name in CUSTOM_AGENTS


def model_file(models_dir, run_name, agent_name):
    """Ruta del modelo final: .pt para agentes custom, .zip para SB3."""
    ext = "pt" if is_custom(agent_name) else "zip"
    return Path(models_dir) / run_name / f"final.{ext}"


def _custom_class(name):
    module_name, class_name = CUSTOM_AGENTS[name]
    full_name = f"{__package__}.{module_name}"
    try:
        module = importlib.import_module(full_name)
    except ModuleNotFoundError as exc:
        if exc.name == full_name:
            raise ImportError(
                f"Falta agents/{module_name}.py. Impleméntalo con la interfaz descrita "
                f"en agents/__init__.py para poder usar '{name}'."
            ) from exc
        raise
    return getattr(module, class_name)


def build_agent(name, env, device="cpu", seed=None, tensorboard_log=None, **hp):
    """
    hp admite (todos opcionales): lr, gamma, k_epochs, mini_batch_size.
    Cada agente usa los que le aplican e ignora el resto.
    """
    hp = {k: v for k, v in hp.items() if v is not None}

    if name in SB3_ALGOS:
        from .sb3_agents import build_sb3_agent
        return build_sb3_agent(SB3_ALGOS[name], env, tensorboard_log=tensorboard_log,
                               seed=seed, device=device, **hp)

    cls = _custom_class(name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    kwargs = {"device": device}
    if "gamma" in hp:
        kwargs["gamma"] = hp["gamma"]
    if name == "custom_ppo":
        kwargs["K_epochs"] = hp.get("k_epochs", 10)
        kwargs["mini_batch_size"] = hp.get("mini_batch_size", 64)
        if "lr" in hp:
            kwargs["lr_actor"] = hp["lr"]
    return cls(state_dim, action_dim, **kwargs)


def load_agent(name, path, env, device="cpu"):
    """Carga un modelo guardado (el entorno debe tener los mismos espacios que en el entrenamiento)."""
    if name in SB3_ALGOS:
        from .sb3_agents import load_sb3_agent
        return load_sb3_agent(SB3_ALGOS[name], str(path), device=device)

    agent = build_agent(name, env, device=device)
    agent.load(str(path))
    return agent