import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_run_metrics(run_folder, runs_dir="runs", freq="W"):
    run_path = Path(runs_dir) / run_folder

    if not run_path.exists():
        print(f"❌ La carpeta '{run_path}' no existe.")
        return

    step_csv = run_path / "step_metrics.csv"
    ep_csv = run_path / "episode_metrics.csv"

    if not step_csv.exists() and not ep_csv.exists():
        print(f"❌ No se encontraron archivos CSV dentro de: {run_path}")
        return

    # Estilo gráfico
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    # Creamos figura con dimensiones amplias
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    freq_label = "Semana" if freq == "W" else "Día"
    fig.suptitle(
        f"Análisis Métricas (Promedio por {freq_label}): {run_folder}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Configuración de los puntos para que se vean delgados y elegantes
    ms = 3  # tamaño del punto (markersize)
    lw = 1.0  # grosor de la línea (linewidth)

    # -------------------------------------------------------------
    # 1. FILA SUPERIOR: MÉTRICAS POR STEP AGRUPADAS
    # -------------------------------------------------------------
    if step_csv.exists():
        print(f"📈 Leyendo steps desde: {step_csv}")
        df_step = pd.read_csv(step_csv)

        # 15 min/paso -> 96 pasos/día -> 672 pasos/semana
        window_size = 672 if freq == "W" else 96
        df_resampled = df_step.groupby(df_step.index // window_size).mean()
        x_step = df_resampled.index

        # (1.1) Recompensa Media por Periodo
        if "reward" in df_resampled.columns:
            axes[0, 0].plot(
                x_step,
                df_resampled["reward"],
                color="tab:blue",
                marker="o",
                markersize=ms,
                linewidth=lw,
                alpha=0.85,
                label=f"Recompensa ({freq_label})",
            )
            axes[0, 0].set_title("Recompensa Promedio por Semana", pad=8)
            axes[0, 0].set_xlabel("Semanas Transcurridas", labelpad=6)
            axes[0, 0].set_ylabel("Recompensa Promedio")
            axes[0, 0].legend(loc="lower right", fontsize=9)

        # (1.2) Potencia Promedio (W) por Periodo
        if "power_w" in df_resampled.columns:
            axes[0, 1].plot(
                x_step,
                df_resampled["power_w"],
                color="tab:red",
                marker="o",
                markersize=ms,
                linewidth=lw,
                alpha=0.85,
                label=f"Potencia W ({freq_label})",
            )
            axes[0, 1].set_title("Potencia Térmica Promedio por Semana", pad=8)
            axes[0, 1].set_xlabel("Semanas Transcurridas", labelpad=6)
            axes[0, 1].set_ylabel("Watts (W)")
            axes[0, 1].legend(loc="upper right", fontsize=9)

        # (1.3) Incomodidad Térmica Promedio (°C) por Periodo
        if "comfort_violation" in df_resampled.columns:
            axes[0, 2].plot(
                x_step,
                df_resampled["comfort_violation"],
                color="tab:purple",
                marker="o",
                markersize=ms,
                linewidth=lw,
                alpha=0.85,
                label=f"Violación °C ({freq_label})",
            )
            axes[0, 2].set_title(
                "Incomodidad Térmica Promedio por Semana", pad=8
            )
            axes[0, 2].set_xlabel("Semanas Transcurridas", labelpad=6)
            axes[0, 2].set_ylabel("Desviación (°C)")
            axes[0, 2].legend(loc="upper right", fontsize=9)

    # -------------------------------------------------------------
    # 2. FILA INFERIOR: MÉTRICAS POR EPISODIO
    # -------------------------------------------------------------
    if ep_csv.exists():
        print(f"📊 Leyendo episodios desde: {ep_csv}")
        df_ep = pd.read_csv(ep_csv)
        x_ep = df_ep["episode"] if "episode" in df_ep.columns else df_ep.index

        if "return" in df_ep.columns:
            axes[1, 0].plot(
                x_ep,
                df_ep["return"],
                marker="o",
                markersize=5,
                linewidth=1.8,
                color="tab:green",
                label="Retorno Total",
            )
            axes[1, 0].set_title("Retorno por Episodio", pad=8)
            axes[1, 0].set_xlabel("Episodio (Año)", labelpad=6)
            axes[1, 0].set_ylabel("Recompensa")
            axes[1, 0].legend(fontsize=9)

        if "energy_kwh" in df_ep.columns:
            axes[1, 1].plot(
                x_ep,
                df_ep["energy_kwh"],
                marker="^",
                markersize=5,
                linewidth=1.8,
                color="tab:red",
                label="Energía Año (kWh)",
            )
            axes[1, 1].set_title("Consumo Energético por Episodio", pad=8)
            axes[1, 1].set_xlabel("Episodio (Año)", labelpad=6)
            axes[1, 1].set_ylabel("kWh / Episodio")
            axes[1, 1].legend(fontsize=9)

        if "violation_degC_h" in df_ep.columns:
            axes[1, 2].plot(
                x_ep,
                df_ep["violation_degC_h"],
                marker="s",
                markersize=5,
                linewidth=1.8,
                color="tab:purple",
                label="Violación Año (°C·h)",
            )
            axes[1, 2].set_title("Incomodidad Térmica por Episodio", pad=8)
            axes[1, 2].set_xlabel("Episodio (Año)", labelpad=6)
            axes[1, 2].set_ylabel("°C·h / Episodio")
            axes[1, 2].legend(fontsize=9)

    # Ajuste preciso de espacios para evitar choque de títulos y etiquetas de ejes
    plt.subplots_adjust(
        left=0.06, right=0.97, top=0.91, bottom=0.08, hspace=0.38, wspace=0.25
    )

    output_png = run_path / f"plot_summary_{freq_label.lower()}_fixed.png"
    plt.savefig(output_png, dpi=300)
    print(f"\n✅ Gráfica generada en: {output_png}\n")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Graficador de 6 paneles agrupados para HVAC"
    )
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        help="Nombre de la carpeta de la ejecución dentro de runs/",
    )
    parser.add_argument(
        "--freq",
        choices=["W", "D"],
        default="W",
        help="Agrupación para la fila superior: 'W' (Semana) o 'D' (Día)",
    )
    args = parser.parse_args()

    runs_path = Path("runs")

    if not args.run:
        folders = []
        for p in runs_path.rglob("*"):
            if p.is_dir() and (
                (p / "step_metrics.csv").exists()
                or (p / "episode_metrics.csv").exists()
            ):
                rel_path = p.relative_to(runs_path)
                folders.append(str(rel_path))

        folders = sorted(list(set(folders)))

        if not folders:
            print("❌ No se encontraron subcarpetas válidas en runs/")
            return

        print("\n--- Carpetas encontradas en runs/ ---")
        for idx, name in enumerate(folders, start=1):
            print(f"  [{idx}] {name}")
        print("-------------------------------------")

        try:
            choice = int(input(f"Selecciona una opción (1-{len(folders)}): "))
            selected_run = folders[choice - 1]
        except (ValueError, IndexError):
            print("Opción inválida.")
            return
    else:
        selected_run = args.run

    plot_run_metrics(selected_run, freq=args.freq)


if __name__ == "__main__":
    main()