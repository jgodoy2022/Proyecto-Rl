"""
Orquestador de entrenamiento: un agente por ejecución, elegido por línea de comandos.

Ejemplos:
    python train.py --agent custom_ppo
    python train.py --agent sb3_ppo --timesteps 350400
    python train.py --agent custom_ppo --run-name ppo_lr1e-4 --lr 1e-4

Salidas (aisladas por run):
    models/<run>/final.(pt|zip), checkpoints/, obs_norm.npz, config.json
    runs/<run>/  -> TensorBoard + step_metrics.csv + episode_metrics.csv
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from agents import AGENT_CHOICES, build_agent, is_custom
from utils.env_setup import (DEFAULT_ENV_ID, describe_env, find_processor,
                             get_obs_variables, make_env)
from utils.metrics import MetricsLogger


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento RL para control HVAC (Sinergym)")
    p.add_argument("--agent", required=True, choices=AGENT_CHOICES, help="Agente a entrenar")
    p.add_argument("--env-id", default=DEFAULT_ENV_ID)
    p.add_argument("--timesteps", type=int, default=35040,
                   help="Pasos totales (35040 = 1 año con pasos de 15 min)")
    p.add_argument("--run-name", default=None, help="Nombre del run (por defecto, el del agente)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-freq", type=int, default=5000, help="Pasos entre checkpoints")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--step-hours", type=float, default=0.25, help="Duración de un paso en horas")
    p.add_argument("--no-normalize-obs", action="store_true", help="Desactiva la normalización de observaciones")
    # Hiperparámetros comunes (si se omiten, cada agente usa los suyos)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--k-epochs", type=int, default=None)
    p.add_argument("--mini-batch-size", type=int, default=None)
    # Solo agentes custom
    p.add_argument("--update-every", type=int, default=2048, help="Pasos por rollout (agentes custom)")
    return p.parse_args()


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_norm(env, model_dir):
    processor = find_processor(env)
    if processor is not None and processor.normalize:
        processor.save_stats(model_dir / "obs_norm.npz")


def train_custom(agent, env, args, metrics, model_dir):
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    obs, _ = env.reset(seed=args.seed)
    for step in range(1, args.timesteps + 1):
        action = agent.select_action(obs)
        env_action = np.clip(action, -1.0, 1.0).astype(np.float32)   # el log-prob usa la acción sin recortar
        next_obs, reward, terminated, truncated, info = env.step(env_action)

        agent.store_transition(reward, terminated, truncated, next_obs)
        metrics.log_step(reward, info, env_action)
        obs = next_obs

        if terminated or truncated:
            metrics.end_episode()
            if step < args.timesteps:
                obs, _ = env.reset()

        if step % args.update_every == 0 or step == args.timesteps:
            stats = agent.update(last_state=obs)
            metrics.log_scalars(stats, step, "train")

        if step % args.save_freq == 0:
            agent.save(str(ckpt_dir / f"step_{step}.pt"))
            save_norm(env, model_dir)

    agent.save(str(model_dir / "final.pt"))


def train_sb3(agent, args, metrics, model_dir, run_dir):
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from agents.sb3_agents import MetricsCallback

    callbacks = CallbackList([
        MetricsCallback(metrics),
        CheckpointCallback(save_freq=args.save_freq, save_path=str(model_dir / "checkpoints"),
                           name_prefix="step"),
    ])
    agent.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name="sb3")
    agent.save(str(model_dir / "final"))   # SB3 agrega la extensión .zip


def main():
    args = parse_args()
    run = args.run_name or args.agent
    run_dir = Path(args.runs_dir) / run
    model_dir = Path(args.models_dir) / run
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Entrenando '{args.agent}' (run: {run}) durante {args.timesteps} pasos ---")
    set_seed(args.seed)

    env = make_env(args.env_id, normalize_obs=not args.no_normalize_obs)
    describe_env(env)
    metrics = MetricsLogger(str(run_dir), get_obs_variables(env), step_hours=args.step_hours)

    with open(model_dir / "config.json", "w") as f:
        json.dump({"agent": args.agent, "env_id": args.env_id, "seed": args.seed,
                   "timesteps": args.timesteps, "normalize_obs": not args.no_normalize_obs}, f, indent=2)

    hp = dict(lr=args.lr, gamma=args.gamma, k_epochs=args.k_epochs, mini_batch_size=args.mini_batch_size)
    try:
        agent = build_agent(args.agent, env, device=args.device, seed=args.seed,
                            tensorboard_log=str(run_dir), **hp)
        if is_custom(args.agent):
            train_custom(agent, env, args, metrics, model_dir)
        else:
            train_sb3(agent, args, metrics, model_dir, run_dir)
        save_norm(env, model_dir)
        print(f"\n¡Entrenamiento completado! Modelo en {model_dir}/")
    finally:
        metrics.close()
        env.close()


if __name__ == "__main__":
    main()