# Clasificador de Perros y Gatos usando CNN

Este proyecto implementa una Red Neuronal Convolucional (CNN) para clasificar imágenes de perros y gatos usando TensorFlow y Keras.

## Descripción
El modelo utiliza una arquitectura CNN con múltiples capas convolucionales, normalización por lotes y regularización L2 para prevenir el sobreajuste. El proyecto alcanza una precisión del 89.29% en el conjunto de prueba.

## Estructura del Modelo
- 4 bloques convolucionales con:
  - Capas Conv2D
  - Batch Normalization
  - Activación ReLU
  - MaxPooling2D
- Capa Flatten
- Dropout (0.5)
- Capa Dense final con activación sigmoid

## Dataset
- Dataset de 30,000 imágenes de perros y gatos (150x150 píxeles)
- Split de datos:
  - Entrenamiento: 70%
  - Validación: 15%
  - Prueba: 15%

## Técnicas de Optimización
- Data Augmentation:
  - Rotación
  - Desplazamiento horizontal
  - Zoom
  - Volteo horizontal
- Regularización L2 (lambda=0.001)
- Batch Normalization
- Dropout (50%)
- Early Stopping
- Learning Rate Reduction
- Checkpoints para guardar el mejor modelo

## Requisitos
```python
tensorflow
keras
numpy
pandas
matplotlib
kagglehub
```

## Estructura del Proyecto
```
.
├── model_cat.keras         # Modelo entrenado
├── best_dogs_cats_model.keras   # Mejor modelo guardado
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

## Resultados
- Precisión en entrenamiento: 89.25%
- Precisión en prueba: 89.29%

## Uso
1. Clonar el repositorio
2. Instalar dependencias:
```bash
pip install -r requirements.txt
```
3. Ejecutar el notebook o script principal
4. Para predicciones:
```python
from keras.models import load_model
model = load_model('best_dogs_cats_model.keras')
# Preparar imagen y hacer predicción
```

## Mejoras Futuras
- Implementar transfer learning con modelos pre-entrenados
- Aumentar el tamaño del dataset
- Probar diferentes arquitecturas de CNN
- Implementar interfaz de usuario para predicciones

## Autor
Eduard Manuel Giraldo Martinez
