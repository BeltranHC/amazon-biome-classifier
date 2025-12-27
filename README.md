# Transfer Learning con ResNet-18 para Clasificación de Biomas de la Amazonía Peruana

**Universidad Nacional del Altiplano - Ciencia de Datos I**

## Descripción del Proyecto

Este proyecto implementa un sistema de clasificación de imágenes de biomas amazónicos utilizando **Transfer Learning** con la arquitectura **ResNet-18** pre-entrenada en ImageNet. El objetivo es clasificar automáticamente imágenes satelitales en 14 categorías de ecosistemas de la Amazonía Peruana.

## Objetivo

Desarrollar un clasificador de imágenes que pueda identificar diferentes tipos de biomas amazónicos peruanos utilizando técnicas de aprendizaje profundo, aprovechando el conocimiento pre-entrenado de redes neuronales convolucionales.

## Dataset

**BiomePeruvianAmazon2023** - Dataset de Kaggle que contiene imágenes satelitales de 14 clases de ecosistemas amazónicos:

- Aguajales
- Bamboo-dominated forests
- Floodplain forests
- Gallery forests  
- Highland grasslands
- Lowland evergreen forests
- Montane cloud forests
- Palm swamp forests
- Paramos
- Puna grasslands
- Riparian forests
- Seasonally flooded forests
- Shrublands
- Wetlands

## Tecnologías Utilizadas

- **Python 3.x**
- **PyTorch** - Framework de deep learning
- **torchvision** - Modelos pre-entrenados y transformaciones
- **NumPy** - Operaciones numéricas
- **Matplotlib & Seaborn** - Visualización de datos
- **scikit-learn** - Métricas de evaluación
- **kagglehub** - Descarga del dataset

## Arquitectura del Modelo

- **Modelo Base**: ResNet-18 pre-entrenada en ImageNet (1000 clases)
- **Técnica**: Transfer Learning con fine-tuning de la capa clasificadora
- **Modificaciones**:
  - Capas convolucionales congeladas (extracción de características)
  - Capa fully-connected personalizada con Dropout (0.3)
  - Capa de salida adaptada a 14 clases

## Estructura del Proyecto

```
Proyecto Transfer Learning
├── TransferLearning_PeruvianAmazon_Colab.ipynb  # Notebook principal (Google Colab)
├── distribucion_clases.png                      # Gráfico de distribución de datos
├── muestras_dataset.png                         # Muestras visuales del dataset
├── curvas_aprendizaje.png                       # Curvas de loss y accuracy
├── matriz_confusion.png                         # Matriz de confusión
├── predicciones_ejemplo.png                     # Ejemplos de predicciones
├── modelo_biomas_amazonia.pth                   # Modelo entrenado guardado
├── index.html                                   # Interfaz web de demostración
├── servidor.py                                  # Servidor backend para inferencia
├── ciencia_de_datos.pdf                         # Artículo/reporte del proyecto
└── README.md                                    # Este archivo
```

## Cómo Ejecutar

### Opción 1: Google Colab (Recomendado)
1. Abrir `TransferLearning_PeruvianAmazon_Colab.ipynb` en Google Colab
2. Ejecutar todas las celdas secuencialmente
3. El notebook descargará automáticamente el dataset desde Kaggle

### Opción 2: Ejecución Local
```bash
# Instalar dependencias
pip install torch torchvision kagglehub matplotlib seaborn scikit-learn

# Ejecutar el notebook con Jupyter
jupyter notebook TransferLearning_PeruvianAmazon_Colab.ipynb
```

### Opción 3: Aplicación Web
```bash
# Ejecutar el servidor de inferencia
python servidor.py

# Abrir index.html en el navegador
```

## Resultados

El modelo logra clasificar efectivamente los diferentes biomas amazónicos con las siguientes métricas:

- **Data Augmentation**: Aplicación de transformaciones aleatorias (flip, rotación, color jitter)
- **Optimizador**: Adam con learning rate de 0.001
- **Scheduler**: StepLR con reducción de learning rate
- **Épocas**: 10 épocas de entrenamiento
- **Split**: 80% entrenamiento, 20% validación

## Visualizaciones Generadas

1. **Distribución de Clases**: Muestra el balance del dataset
2. **Muestras del Dataset**: Ejemplos visuales de cada bioma
3. **Curvas de Aprendizaje**: Evolución del loss y accuracy
4. **Matriz de Confusión**: Análisis de errores de clasificación
5. **Predicciones de Ejemplo**: Comparación predicción vs. valor real

## Metodología

1. **Preparación de Datos**: Carga y preprocesamiento de imágenes (224x224)
2. **Aumento de Datos**: Transformaciones para mejorar generalización
3. **Transfer Learning**: Uso de ResNet-18 pre-entrenada
4. **Entrenamiento**: Fine-tuning de la capa clasificadora
5. **Evaluación**: Métricas de precision, recall y F1-score
6. **Visualización**: Generación de gráficos y matrices

## Referencias

- Dataset: [BiomePeruvianAmazon2023 - Kaggle](https://www.kaggle.com/datasets/earthshot/biomeperuvianamazon2023)
- Arquitectura: [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- Framework: [PyTorch](https://pytorch.org/)

## Autor

**JuniDev**  
Universidad Nacional del Altiplano  
Curso: Ciencia de Datos I  
Trabajo Final - 8vo Semestre

---

*Proyecto desarrollado con fines académicos para el estudio de técnicas de Transfer Learning aplicadas a la clasificación de ecosistemas amazónicos peruanos.*
