# 🛍️ Predicción de Ventas — Eli Boutique IA + Dashboard

## ¿Qué es esta aplicación?

Es un **dashboard interactivo** desarrollado con **Streamlit** que predice las ventas de una tienda de ropa para el **próximo mes**, utilizando un modelo de **Inteligencia Artificial (Red Neuronal Artificial)** entrenado con datos históricos reales obtenidos desde una **API REST** desarrollada en **Laravel (PHP)**.

El sistema no solo predice **cantidades de unidades**, sino que también proyecta **ingresos, costos y beneficio neto** en soles (S/) por producto, usando precios y costos reales de la tienda.

### ¿Cómo funciona la predicción?

1. El modelo de IA analiza los datos históricos de ventas y calcula una **tendencia** para cada producto.
2. Esa tendencia se aplica sobre el **promedio mensual histórico real** de cada producto.
3. La variación se limita a **±15%**, lo que significa que algunos productos suben y otros bajan, reflejando un comportamiento realista del mercado.

---

## Modelo de Inteligencia Artificial

| Aspecto | Detalle |
|---|---|
| **Tipo de modelo** | Red Neuronal Artificial (ANN) |
| **Framework** | TensorFlow / Keras |
| **Arquitectura** | 3 capas Dense (64 → 32 → 1 neurona) |
| **Función de activación** | ReLU (capas ocultas), Linear (salida) |
| **Optimizador** | Adam |
| **Función de pérdida** | Mean Squared Error (MSE) |
| **Normalización** | StandardScaler (scikit-learn) |
| **Early Stopping** | Sí, con paciencia de 5 épocas |
| **Épocas máximas** | 50 |
| **Tamaño de lote** | 16 |
| **Control de predicción** | Variación ±15% sobre promedio histórico |

### Variables de entrada (features):
- `producto_id` — Identificador del producto
- `año` — Año de la venta
- `mes` — Mes de la venta
- `día` — Día de la venta

### Variable objetivo (target):
- `cantidad_vendida` — Cantidad de unidades vendidas

---

## Origen de los datos

Los datos se obtienen en tiempo real desde una **API REST** desarrollada en Laravel.

**Endpoint:**
```
http://127.0.0.1:8000/api/obtener-datos-ventas
```

La API devuelve un JSON con los campos:
`venta_id`, `producto_id`, `producto_nombre`, `cantidad_vendida`, `año`, `mes`, `dia`

> No se usan archivos CSV. Todo viene de la base de datos del sistema de la tienda.

---

## Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `app.py` | Dashboard Streamlit (predicción, gráficos, finanzas, recomendaciones, ayuda) |
| `training.py` | Script para entrenar el modelo de IA (consume la API) |
| `modelo_ventas.h5` | Modelo de red neuronal entrenado |
| `scaler_X.pkl` | Escalador de características (StandardScaler) |
| `y_mean.pkl` | Media de la variable objetivo (para desnormalizar) |
| `y_std.pkl` | Desviación estándar de la variable objetivo |
| `requirements.txt` | Dependencias del proyecto |
| `logo_eli_boutique.png` | Logo de la tienda |
| `README.md` | Este archivo de documentación |

---

## Funcionalidades del dashboard

### 📊 Tab: Predicción
- **KPIs históricos**: Transacciones, unidades vendidas, producto más vendido, productos únicos.
- **Predicción individual**: Selecciona un producto en el sidebar y ve su estimación vs promedio histórico.

### 📈 Tab: Gráficos
- **Predicción vs Promedio histórico**: Gráfico de barras agrupadas por producto.
- **Mapa de calor de predicción**: Barras con escala de color Viridis.
- **Top 5 productos**: Los 5 con mayor demanda estimada.
- **Gráfico circular**: Distribución estimación vs promedio del producto seleccionado.

### 📋 Tab: Resumen
- **KPIs de unidades**: Total estimado, promedio por producto, mayor/menor demanda.
- **Proyección financiera**: Ingresos, costos, beneficio neto y margen de beneficio en soles (S/).
- **Tabla completa**: Cada producto con unidades, precio unitario, ingreso, costo y beneficio.
- **Gráfico financiero**: Barras agrupadas de ingreso vs costo vs beneficio por producto.

### 💡 Tab: Recomendaciones
- **Alta demanda**: Productos que necesitan stock suficiente.
- **Baja demanda**: Productos que podrían beneficiarse de promociones.
- **Consejos generales**: Tips para mejorar la gestión del negocio.

### ❓ Tab: Ayuda
- Explicación de la app, conceptos de IA, significado de cada indicador (incluidos los financieros).
- Tabla técnica del modelo y tecnologías usadas.

### Otras funcionalidades
- **Filtro de producto en sidebar**: Selección rápida desde el panel lateral.
- **Mes dinámico**: El mes de predicción se calcula automáticamente (siempre el próximo mes).
- **Precios reales por producto**: Cada producto tiene su precio y costo real en soles.
- **Cache de predicción**: Se calcula una sola vez por sesión (session_state).

---

## Requisitos previos

- **Python 3.8** o superior
- **API Laravel** corriendo en `http://127.0.0.1:8000`

### Dependencias (requirements.txt)

```
setuptools
wheel
streamlit
tensorflow
numpy
pandas
matplotlib
requests
Pillow
joblib
scikit-learn
plotly
```

---

## ¿Cómo correr la aplicación?

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Asegúrate de que la API Laravel esté corriendo

```
http://127.0.0.1:8000/api/obtener-datos-ventas
```

### 3. Entrenar el modelo (solo la primera vez o cuando haya datos nuevos)

```bash
python training.py
```

Esto genera: `modelo_ventas.h5`, `scaler_X.pkl`, `y_mean.pkl`, `y_std.pkl`

### 4. Iniciar el dashboard

```bash
streamlit run app.py
```

Se abrirá en tu navegador en `http://localhost:8501`

---

## ¿Cuándo re-entrenar el modelo?

| Situación | ¿Re-entrenar? |
|---|---|
| Se registró una venta nueva | ❌ No es necesario |
| Pasó un mes completo con datos nuevos | ✅ Recomendado |
| Las predicciones ya no reflejan la realidad | ✅ Sí |
| Se agregaron productos nuevos al catálogo | ✅ Sí |
| Se modificaron precios en `app.py` | ❌ No (precios no afectan el modelo) |

---

## Precios y costos por producto

Los precios y costos están definidos en el diccionario `PRECIOS_PRODUCTO` dentro de `app.py`. Si se agrega un producto nuevo a la tienda, se debe añadir al diccionario. Si no se encuentra, usa un precio por defecto de S/ 60.00.

---

## Tecnologías utilizadas

| Componente | Tecnología |
|---|---|
| Dashboard | Streamlit |
| Backend / API | Laravel (PHP) |
| Modelo de IA | TensorFlow / Keras |
| Gráficos interactivos | Plotly |
| Procesamiento de datos | Pandas, NumPy |
| Normalización | scikit-learn (StandardScaler) |
| Serialización | Joblib |
| Imágenes | Pillow |

---

## Autores

- **Bryan Amaya**
- **Karen Perez**
