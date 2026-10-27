"""
Registro de métricas clave: recompensa, temperatura, consumo energético y confort.

Guarda en `log_dir`:
  - step_metrics.csv     una fila por paso de simulación
  - episode_metrics.csv  una fila por episodio (resumen)
  - eventos de TensorBoard (si tensorboard está instalado)

Las temperaturas y la potencia se leen de info["raw_obs"] (observación sin
normalizar, la agrega utils/env_setup.ObsProcessor) y de claves de `info` de
Sinergym. Si alguna no existe en tu versión, el valor queda como NaN y se omite
sin romper el entrenamiento.
"""

import csv
import os

import numpy as np

# Nombres esperados (Sinergym 3.x, edificio 5Zone). Se buscan por coincidencia parcial.
TEMP_KEY = "air_temperature"
OUTDOOR_KEY = "outdoor_temperature"
POWER_OBS_KEY = "HVAC_electricity_demand_rate"
# Claves habituales de info en Sinergym
INFO_POWER = "total_power_demand"
INFO_VIOLATION = "total_temperature_violation"


def _nansum(values):
    arr = np.asarray(values, dtype=np.float64)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nansum(arr))


def _nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    return float("nan") if arr.size == 0 or np.all(np.isnan(arr)) else float(np.nanmean(arr))


class MetricsLogger:
    def __init__(self, log_dir, observation_variables=None, step_hours=0.25,
                 tensorboard=True, tb_steps=False, verbose=True):
        """
        step_hours: duración de un paso en horas (0.25 = 15 min, 35 040 pasos/año).
        tb_steps:   si True, escribe curvas paso a paso en TensorBoard (para evaluación).
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.step_hours = step_hours
        self.tb_steps = tb_steps
        self.verbose = verbose

        names = observation_variables or []
        self.idx_temp = [i for i, n in enumerate(names) if TEMP_KEY in n]
        self.idx_out = next((i for i, n in enumerate(names) if n == OUTDOOR_KEY), None)
        self.idx_power = next((i for i, n in enumerate(names) if n == POWER_OBS_KEY), None)

        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                print("[metrics] TensorBoard no disponible: se guardan solo los CSV.")

        self._step_file = open(os.path.join(log_dir, "step_metrics.csv"), "w", newline="")
        self._step_csv = csv.writer(self._step_file)
        self._step_header_written = False
        self._episode_file = open(os.path.join(log_dir, "episode_metrics.csv"), "w", newline="")
        self._episode_csv = None

        self.global_step = 0
        self.episode = 0
        self._reset_episode()

    # ------------------------------------------------------------------ #
    def _reset_episode(self):
        self.rewards, self.temps, self.outdoor, self.power, self.violation = [], [], [], [], []
        self.ep_step = 0
        self._cum_reward = 0.0
        self._cum_energy_kwh = 0.0

    def _extract(self, info):
        raw = info.get("raw_obs")
        temp = out = power = float("nan")
        if raw is not None:
            if self.idx_temp:
                temp = float(np.mean(raw[self.idx_temp]))
            if self.idx_out is not None:
                out = float(raw[self.idx_out])
            if self.idx_power is not None:
                power = float(raw[self.idx_power])
        if info.get(INFO_POWER) is not None:
            power = float(info[INFO_POWER])
        violation = info.get(INFO_VIOLATION)
        violation = float("nan") if violation is None else float(violation)
        return temp, out, power, violation

    # ------------------------------------------------------------------ #
    def log_step(self, reward, info, action=None):
        temp, out, power, violation = self._extract(info)
        reward = float(reward)
        self.global_step += 1
        self.ep_step += 1

        self.rewards.append(reward)
        self.temps.append(temp)
        self.outdoor.append(out)
        self.power.append(power)
        self.violation.append(violation)

        self._cum_reward += reward
        if not np.isnan(power):
            self._cum_energy_kwh += power * self.step_hours / 1000.0

        action = [] if action is None else np.asarray(action, dtype=np.float64).flatten().tolist()
        if not self._step_header_written:
            header = ["global_step", "episode", "step", "reward", "temperature",
                      "outdoor_temperature", "power_w", "comfort_violation"]
            header += [f"action_{i}" for i in range(len(action))]
            self._step_csv.writerow(header)
            self._step_header_written = True
        self._step_csv.writerow([self.global_step, self.episode, self.ep_step, reward,
                                 temp, out, power, violation] + action)

        if self.writer is not None and self.tb_steps:
            g = self.global_step
            self.writer.add_scalar("step/reward", reward, g)
            self.writer.add_scalar("step/cumulative_reward", self._cum_reward, g)
            self.writer.add_scalar("step/cumulative_energy_kwh", self._cum_energy_kwh, g)
            for tag, value in (("temperature", temp), ("outdoor_temperature", out),
                               ("power_w", power), ("comfort_violation", violation)):
                if not np.isnan(value):
                    self.writer.add_scalar(f"step/{tag}", value, g)

    def end_episode(self):
        """Cierra el episodio actual, escribe su resumen y lo devuelve (None si estaba vacío)."""
        if not self.rewards:
            return None

        viol = np.asarray(self.violation, dtype=np.float64)
        valid = viol[~np.isnan(viol)]
        summary = {
            "episode": self.episode,
            "steps": len(self.rewards),
            "return": float(np.sum(self.rewards)),
            "energy_kwh": _nansum(self.power) * self.step_hours / 1000.0,
            "mean_temperature": _nanmean(self.temps),
            "mean_outdoor_temperature": _nanmean(self.outdoor),
            "violation_degC_h": _nansum(self.violation) * self.step_hours,
            "violation_steps_pct": float(100.0 * np.mean(valid > 0)) if valid.size else float("nan"),
        }

        if self._episode_csv is None:
            self._episode_csv = csv.DictWriter(self._episode_file, fieldnames=list(summary.keys()))
            self._episode_csv.writeheader()
        self._episode_csv.writerow(summary)
        self._episode_file.flush()
        self._step_file.flush()

        if self.writer is not None:
            for key, value in summary.items():
                if key not in ("episode", "steps") and not np.isnan(value):
                    self.writer.add_scalar(f"episode/{key}", value, self.global_step)
            self.writer.flush()

        if self.verbose:
            print(f"[episodio {self.episode}] pasos={summary['steps']} "
                  f"retorno={summary['return']:.2f} energía={summary['energy_kwh']:.1f} kWh "
                  f"T_media={summary['mean_temperature']:.2f} °C "
                  f"violación={summary['violation_degC_h']:.1f} °C·h")

        self.episode += 1
        self._reset_episode()
        return summary

    def log_scalars(self, values, step, prefix="train"):
        """Métricas de entrenamiento arbitrarias (pérdidas, entropía...)."""
        if self.writer is None or not values:
            return
        for key, value in values.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def close(self):
        self.end_episode()
        if self.writer is not None:
            self.writer.close()
        self._step_file.close()
        self._episode_file.close()