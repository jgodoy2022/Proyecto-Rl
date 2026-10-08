# Control Dinámico de Clima y Eficiencia Energética en Edificios con Aprendizaje por Refuerzo (RL)

Proyecto de Aprendizaje por Refuerzo para la optimización y control de sistemas HVAC (Heating, Ventilation, and Air Conditioning) en edificaciones. El objetivo es equilibrar la comodidad térmica de los ocupantes con la reducción del consumo energético.

---

## Integrantes del Equipo
* Natalia Contreras
* Joaquín Godoy
* Gabriela Zambrano

## Requisitos Previos

Tener instalado en tu computadora:

1. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**
2. **[Git](https://git-scm.com/)**

---

## Instalación y Configuración

### Paso 1: Clonar el Repositorio
```bash
git clone [https://github.com/TU_USUARIO/Proyecto-Rl.git]
cd Proyecto-Rl
```

### Paso 2: Iniciar Docker Desktop

### Paso 3: Desplegar el Contenedor de Sinergym

## Windows (PowerShell): 
```bash
docker run -it -p 6006:6006 -v ${PWD}:/workspace -w /workspace sailugr/sinergym:latest bash
```

## macOS / Linux:
```bash
docker run -it -p 6006:6006 -v $(pwd):/workspace -w /workspace sailugr/sinergym:latest bash
```

### Paso 4: Instalar Dependencias Internas de Visualización
```bash
pip install tensorboard
```
