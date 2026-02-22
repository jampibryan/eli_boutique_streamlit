# Predicción de Ventas — Boutique IA + Dashboard

## Descripción

Este proyecto utiliza Streamlit y Machine Learning para predecir ventas de una tienda de ropa. Los datos se obtienen desde una API REST desarrollada en Laravel. El sistema permite visualizar predicciones y análisis de ventas en un dashboard interactivo.

## Componentes

- **API Laravel**: Proporciona datos históricos de ventas en formato JSON.
- **training.py**: Script para entrenar el modelo de predicción usando TensorFlow.
- **app.py**: Aplicación Streamlit para mostrar el dashboard y realizar predicciones.
- **Archivos de modelo**: 
  - `modelo_ventas.h5`
  - `scaler_X.pkl`
  - `y_mean.pkl`
  - `y_std.pkl`

## Flujo de trabajo

1. **Entrenamiento**
   - Ejecuta `training.py` para entrenar el modelo con los datos de la API.
   - Se generan los archivos del modelo y escaladores.

2. **Visualización**
   - Ejecuta `app.py` con Streamlit.
   - La app consume la API y muestra predicciones de ventas para los próximos meses.

## Requisitos

- Python 3.8+
- TensorFlow
- scikit-learn
- pandas
- joblib
- streamlit
- matplotlib
- requests

Instala dependencias con:
```
pip install -r requirements.txt
```

## Configuración

1. Asegúrate de que la API Laravel esté corriendo en:
   ```
   http://127.0.0.1:8000/api/obtener-datos-ventas
   ```
2. Ejecuta el entrenamiento:
   ```
   python training.py
   ```
3. Inicia el dashboard:
   ```
   streamlit run app.py
   ```

## Preguntas frecuentes

- **¿Cuándo debo ejecutar el entrenamiento?**
  - Solo cuando tengas nuevos datos de ventas o quieras actualizar el modelo.

- **¿Qué datos utiliza el modelo?**
  - producto_id, año, mes, día, cantidad_vendida.

- **¿Puedo usar ventas.csv?**
  - No, el sistema ahora consume solo la API.

## Autores

- Bryan Amaya
- Karen Perez
