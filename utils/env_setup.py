import gymnasium as gym
import sinergym

print("--- Verificando instalación de Sinergym y EnergyPlus ---")

try:
    # Cargar el entorno de 5 zonas
    env = gym.make('Eplus-5zone-hot-continuous-v1')
    obs, info = env.reset()
    
    print("\n¡ÉXITO! El entorno se ha iniciado correctamente.")
    print(f"Espacio de observaciones: {env.observation_space.shape}")
    print(f"Espacio de acciones: {env.action_space.shape}")
    
    # Probar 3 pasos del simulador
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Paso {step + 1} - Recompensa: {reward:.4f}")

    env.close()
    print("\n--- Entorno cerrado correctamente ---")

except Exception as e:
    print(f"\nERROR: Ocurrió un problema al cargar el entorno:\n{e}")