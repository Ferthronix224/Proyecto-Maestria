# Diseño automático de detectores de puntos de interés mediante evolución gramatical

Este proyecto implementa un **método de evolución gramatical** para diseñar detectores de puntos de interés en imágenes.  
La idea principal es utilizar **gramáticas libres de contexto** para generar filtros de procesamiento de imágenes que se optimizan con **Evolución Diferencial (DE)**.  

El proyecto está acelerado en **GPU mediante CuPy y cuCIM**, lo cual permite trabajar con grandes volúmenes de imágenes de manera eficiente.

---

## 🚀 Características

- **GPU Acceleration** con [CuPy](https://cupy.dev/) y [cuCIM](https://docs.rapids.ai/api/cucim/stable/).
- **Evolución Diferencial** como algoritmo evolutivo.
- **Gramáticas libres de contexto** para mapear genotipos a fenotipos (filtros).
- Implementación de múltiples **filtros**:
  - Gaussian Blur (σ=1, σ=2)  
  - Laplacian y Laplacian of Gaussian  
  - Derivadas Gaussianas (X, Y)  
  - Operaciones aritméticas: raíz cuadrada, cuadrado, logaritmo, absoluto, multiplicar por 0.5  
  - Filtros de media, mediana y ecualización de histograma
- Métrica de desempeño: **Tasa de repetibilidad** entre puntos de interés en imágenes transformadas.
- Soporte para **transformaciones geométricas**:
  - Rotación  
  - Traslación  
  - Escalado  

---

## 📂 Estructura del Proyecto

├── Filters.py # Filtros de imagen implementados con CuPy/cuCIM  
├── Fitness.py # Métrica de repetibilidad  
├── Genotype.py # Evolución Diferencial (mutación, cruce y selección)  
├── MP.py # Mapping Process (genotipo -> fenotipo)  
├── Process.py # Pipeline principal de evaluación  
├── Transformations.py # Transformaciones geométricas de interés  
├── main.py # Script principal de ejecución  
├── img/ # Carpeta de imágenes (originales y transformadas)  
└── README.md # Documentación del proyecto  


---

## ⚙️ Requisitos

- Python 3.10+
- [CuPy](https://cupy.dev/) (compatible con CUDA 12.4 en este proyecto)
- [cuCIM](https://docs.rapids.ai/api/cucim/stable/)
- NumPy
- scikit-image

Instalación en Linux con CUDA:

```bash
pip install cupy-cuda12x cucim-cu12 scikit-image numpy
```

▶️ Ejecución

1. Prepara las imágenes en la carpeta img/:

    - img/originals/ → imágenes originales.

    - img/rotated/, img/translated/, img/scale/ → imágenes transformadas.

2. Corre el script principal:

```bash
python main.py
```


El script entrenará un detector evolutivo para cada transformación (rotación, traslación, escalado).

📊 Ejemplo de salida
```bash
rotated/90
Generation 1
Best Solution: ft.Gau1(ft.Sqrt(img))
Best Fitness: 62.35%
Time: 15.23 segundos

Generation 12
Best Solution: ft.Lap(ft.Gau2(img))
Best Fitness: 74.91%
Time: 184.54 segundos
```
