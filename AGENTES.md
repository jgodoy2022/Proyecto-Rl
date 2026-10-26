# Estructura del Proyecto e Implementación

Este documento describe la arquitectura del repositorio, la organización de los agentes de aprendizaje por refuerzo y el flujo de trabajo para el entrenamiento y la evaluación de los modelos.

---

## 1. Arquitectura de Directorios

El proyecto se organiza en módulos especializados para separar la lógica de los algoritmos, la configuración del simulador y las herramientas de análisis:

*   **agents/**: Contiene las implementaciones específicas de los agentes de aprendizaje por refuerzo.
    *   `custom_ppo.py`: Implementación personalizada del algoritmo Proximal Policy Optimization.
    *   `custom_a2c.py`: Implementación personalizada del algoritmo Advantage Actor-Critic.
    *   `sb3_agents.py`: Agentes e integraciones basadas en la librería Stable-Baselines3.
*   **utils/**: Funciones auxiliares y de soporte para el entorno.
    *   Inicialización y configuración del simulador.
    *   Funciones de registro (logging) para el almacenamiento de métricas clave: recompensas, consumo energético y evolución de la temperatura.
*   **train.py**: Orquestador principal del proyecto encargado de gestionar el ciclo de entrenamiento.
*   **evaluate.py**: Módulo especializado en la carga de modelos guardados, pruebas de rendimiento y generación de gráficos comparativos.

---

## 2. Flujo de Entrenamiento (train.py)

El archivo `train.py` actúa como el punto de entrada (main) para el entrenamiento. El diseño requiere la selección del agente mediante argumentos por línea de comandos (parsing).

### Justificación Técnica del Diseño

Se ha optado de manera deliberada por la ejecución individual de agentes a través de argumentos por las siguientes razones:

1.  **Gestión de Memoria:** Previene la saturación de los recursos de hardware (RAM/VRAM) al evitar la carga simultánea de múltiples instancias de simulación y redes neuronales.
2.  **Aislamiento de Logs:** Asegura que los datos dirigidos a TensorBoard se almacenen en directorios independientes, evitando la superposición o mezcla de métricas entre distintos algoritmos.

### Estrategias de Ejecución

El sistema es flexible y permite dos enfoques según la disponibilidad de hardware:

*   **Ejecución Secuencial:** Correr los agentes de manera consecutiva en la misma terminal para pruebas controladas o con recursos limitados.
*   **Ejecución Paralela:** Abrir múltiples terminales independientes para entrenar diferentes agentes simultáneamente, siempre que el sistema cuente con la capacidad de procesamiento necesaria.

---

## 3. Evaluación y Métricas (evaluate.py)

Una vez completadas las fases de entrenamiento, el archivo `evaluate.py` se encarga de procesar los resultados sin interferir con el entorno de simulación activo. Sus funciones principales son:

*   Cargar los pesos de los modelos previamente guardados en el directorio de salida.
*   Ejecutar pruebas de inferencia sobre el simulador para validar la política aprendida.
*   Generar las gráficas de rendimiento enfocadas en la eficiencia del agente (recompensas acumuladas), el impacto ambiental (temperatura) y el costo operativo (consumo energético).
