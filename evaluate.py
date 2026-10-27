"""
Evaluación de modelos entrenados: inferencia determinista + curvas en TensorBoard.

Ejemplos:
    python evaluate.py --agents custom_ppo sb3_ppo
    python evaluate.py --agents custom_ppo:ppo_lr1e-4 sb3_ppo      # agente:run_name
    tensorboard --logdir runs/eval

Cada agente se evalúa por separado (un simulador abierto a la vez) y escribe en
runs/eval/<run_name>/. TensorBoard superpone las curvas de los distintos runs
(recompensa acumulada, temperatura, potencia, energía acumulada, confort).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from agents import AGENT_CHOICES, is_custom, load_agent, model_file
from utils.env_setup import DEFAULT_ENV_ID, get_obs_variables, make_env
from utils.metrics import MetricsLogger


def parse_args():
    p = argparse.ArgumentParser(description="Evaluación de agentes HVAC")
    p.add_argument("--agents", nargs="+", required=True,
                   help=f"Agentes a evaluar ({', '.join(AGENT_CHOICES)}); usa agente:run_name si el run tiene otro nombre")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--env-id", default=DEFAULT_ENV_ID, help="Solo se usa si el run no tiene config.json")
    p.add_argument("--step-hours", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def evaluate_agent(spec, args):
    name, _, run = spec.partition(":")
    run = run or name
    if name not in AGENT_CHOICES:
        raise SystemExit(f"Agente desconocido: {name}. Opciones: {AGENT_CHOICES}")

    model_dir = Path(args.models_dir) / run
    path = model_file(args.models_dir, run, name)
    if not path.exists():
        raise SystemExit(f"No existe el modelo {path}. ¿Entrenaste con train.py --agent {name}?")

    config = {"env_id": args.env_id, "normalize_obs": True}
    if (model_dir / "config.json").exists():
        config.update(json.loads((model_dir / "config.json").read_text()))

    # Mismo entorno que en el entrenamiento, con estadísticas de normalización congeladas
    env = make_env(config["env_id"], normalize_obs=config["normalize_obs"],
                   norm_stats_path=str(model_dir / "obs_norm.npz"), freeze_norm=True)
    metrics = MetricsLogger(str(Path(args.runs_dir) / "eval" / run), get_obs_variables(env),
                            step_hours=args.step_hours, tb_steps=True)
    summaries = []
    try:
        agent = load_agent(name, path, env, device=args.device)
        print(f"\n--- Evaluando {run} ({args.episodes} episodio(s)) ---")

        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = False
            while not done:
                if is_custom(name):
                    action = agent.select_action(obs, deterministic=True)
                else:
                    action, _ = agent.predict(obs, deterministic=True)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                metrics.log_step(reward, info, action)
                done = terminated or truncated
            summaries.append(metrics.end_episode())
    finally:
        metrics.close()
        env.close()

    keys = ["return", "energy_kwh", "mean_temperature", "violation_degC_h", "violation_steps_pct"]
    return {"run": run, **{k: float(np.nanmean([s[k] for s in summaries])) for k in keys}}


def main():
    args = parse_args()
    results = [evaluate_agent(spec, args) for spec in args.agents]

    out = Path(args.runs_dir) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Resumen (promedio por episodio) ===")
    print(f"{'run':<22}{'retorno':>12}{'energía kWh':>14}{'T media °C':>12}{'viol °C·h':>12}")
    for r in results:
        print(f"{r['run']:<22}{r['return']:>12.2f}{r['energy_kwh']:>14.1f}"
              f"{r['mean_temperature']:>12.2f}{r['violation_degC_h']:>12.1f}")
    print(f"\nGráficos: tensorboard --logdir {out}")


if __name__ == "__main__":
    main()