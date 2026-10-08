import gymnasium as gym
import sinergym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

print("--- Iniciando entrenamiento HVAC RL con PPO ---")

# 1. Crear el entorno de 5 zonas
env = gym.make('Eplus-5zone-hot-continuous-v1')

# 2. Configurar carpetas de guardado
log_dir = "./tensorboard_logs/"
model_dir = "./models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 3. Callback para guardar el modelo periódicamente
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=model_dir,
    name_prefix='ppo_hvac_model'
)

# 4. Inicializar el agente PPO con Stable-Baselines3
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=0.0003,
    tensorboard_log=log_dir
)

# 5. Ejecutar el entrenamiento (35,040 pasos equivalen a 1 año de simulación)
TOTAL_TIMESTEPS = 35040
print(f"Entrenando durante {TOTAL_TIMESTEPS} pasos de tiempo...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback
)

# 6. Guardar el modelo final
model.save(f"{model_dir}/ppo_hvac_final")
print("\n¡Entrenamiento completado! Modelo guardado en ./models/ppo_hvac_final.zip")

env.close()