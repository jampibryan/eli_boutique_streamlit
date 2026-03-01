# =============================================
# IMPORTACIONES
# =============================================
import streamlit as st
import requests
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# =============================================
# CONFIGURACIÓN DE PÁGINA (debe ir primero)
# =============================================
st.set_page_config(
    page_title="Boutique IA Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CARGA DE MODELO Y ESCALADORES
# =============================================
modelo = tf.keras.models.load_model('modelo_ventas.h5')
scaler = joblib.load('scaler_X.pkl')
y_mean = joblib.load('y_mean.pkl')
y_std = joblib.load('y_std.pkl')

# =============================================
# URL DE LA API LARAVEL
# =============================================
LARAVEL_API_URL = "http://127.0.0.1:8000/api/obtener-datos-ventas"

# =============================================
# FUNCIONES DE DATOS
# =============================================
def cargar_datos_api():
    """Consume la API Laravel y retorna un DataFrame."""
    try:
        response = requests.get(LARAVEL_API_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['producto_id'] = pd.to_numeric(df['producto_id'], errors='coerce')
            df['producto_nombre'] = df['producto_nombre'].astype(str)
            return df
        else:
            st.error("Error al obtener datos desde la API.")
    except Exception as e:
        st.error(f"No se pudo conectar con la API: {e}")
    return None

# =============================================
# FUNCIONES DE PREDICCIÓN
# =============================================
# Calcular automáticamente el primer día del mes siguiente al actual
hoy = datetime.now()
if hoy.month == 12:
    primer_dia = datetime(hoy.year + 1, 1, 1)
else:
    primer_dia = datetime(hoy.year, hoy.month + 1, 1)

# Nombres de los 3 meses de predicción
meses_nombres = []
for i in range(3):
    mes = primer_dia.month + i
    anio = primer_dia.year
    if mes > 12:
        mes -= 12
        anio += 1
    meses_es = {1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',
                 7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'}
    meses_nombres.append(meses_es[mes])

def generar_fechas_3_meses(productos):
    """Genera fechas para los próximos 3 meses por producto."""
    fechas = []
    for i in range(3):
        for j in range(30):
            fecha = primer_dia + timedelta(days=i * 30 + j)
            for producto in productos:
                producto_id = producto['producto_id']
                fechas.append([producto_id, fecha.year, fecha.month, fecha.day])
    return np.array(fechas)

def predecir_ventas_3_meses(productos):
    """Predice ventas para los próximos 3 meses."""
    fechas_3_meses = generar_fechas_3_meses(productos)
    fechas_3_meses_scaled = scaler.transform(fechas_3_meses)
    predicciones_scaled = modelo.predict(fechas_3_meses_scaled)
    predicciones = predicciones_scaled * y_std + y_mean
    predicciones = np.clip(predicciones, 0, 500)
    predicciones = np.round(predicciones).astype(int)
    df_predicciones = pd.DataFrame({
        'producto_id': fechas_3_meses[:, 0],
        'cantidad_estimada': predicciones.flatten()
    })
    df_producto = df_predicciones.groupby('producto_id')['cantidad_estimada'].sum().reset_index()
    producto_nombres = {p['producto_id']: p['producto_nombre'] for p in productos}
    df_producto['producto_nombre'] = df_producto['producto_id'].map(producto_nombres)
    df_producto = df_producto.sort_values(by='cantidad_estimada', ascending=False).reset_index(drop=True)
    return df_producto

def generar_valores_aleatorios(producto_nombre):
    """Genera porcentajes consistentes para distribución mensual."""
    hash_producto = hashlib.md5(producto_nombre.encode()).hexdigest()
    random.seed(int(hash_producto, 16))
    porcentaje_enero = random.uniform(0.3, 0.5)
    porcentaje_febrero = random.uniform(0.25, 0.4)
    porcentaje_marzo = 1 - (porcentaje_enero + porcentaje_febrero)
    return porcentaje_enero, porcentaje_febrero, porcentaje_marzo

def asignar_estimaciones_mensuales(df_producto):
    """Asigna estimaciones mensuales dinámicas según meses_nombres."""
    for index, row in df_producto.iterrows():
        total_estimado = row['cantidad_estimada']
        producto_nombre = row['producto_nombre']
        p1, p2, p3 = generar_valores_aleatorios(producto_nombre)
        if p3 < 0:
            p2 = 1 - p1
            p3 = 0
        total = p1 + p2 + p3
        if total != 1:
            p1 /= total
            p2 /= total
            p3 /= total
        mes1 = int(total_estimado * p1)
        mes2 = int(total_estimado * p2)
        mes3 = total_estimado - (mes1 + mes2)
        df_producto.at[index, meses_nombres[0]] = mes1
        df_producto.at[index, meses_nombres[1]] = mes2
        df_producto.at[index, meses_nombres[2]] = mes3
    return df_producto

# =============================================
# LOGO Y PORTADA
# =============================================
col_logo, col_titulo = st.columns([1, 5])
with col_logo:
    try:
        logo = Image.open("logo_eli_boutique.png")
        st.image(logo, width=120)
    except:
        st.markdown("## 🛍️")
with col_titulo:
    st.markdown("<h1 style='color:#4CAF50;'>Boutique IA — Predicción de Ventas</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#888;'>Dashboard interactivo para análisis y predicción de ventas</h4>", unsafe_allow_html=True)

st.markdown("---")

# =============================================
# CARGAR DATOS
# =============================================
df = cargar_datos_api()
if df is None:
    st.stop()
productos = df.to_dict('records')

# Predicción (solo una vez, guardada en session_state)
if 'df_predicciones' not in st.session_state:
    df_pred = predecir_ventas_3_meses(productos)
    df_pred = asignar_estimaciones_mensuales(df_pred)
    st.session_state.df_predicciones = df_pred

# =============================================
# SIDEBAR — Filtros y desarrolladores
# =============================================
st.sidebar.markdown(
    "<div style='text-align: center;'>"
    "<img src='data:image/png;base64,{}' width='100'>"
    "<p style='margin-top: 5px; font-weight: bold;'>Eli Boutique</p>"
    "</div>".format(
        __import__('base64').b64encode(open("logo_eli_boutique.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Filtros")

productos_unicos = df['producto_nombre'].unique()
producto_seleccionado = st.sidebar.selectbox("Seleccionar producto", productos_unicos)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👩‍💻 Desarrolladores")
st.sidebar.markdown("- Bryan Amaya\n- Karen Perez")

# Filtrar predicción del producto seleccionado
df_sel = st.session_state.df_predicciones[
    st.session_state.df_predicciones['producto_nombre'] == producto_seleccionado
]

# =============================================
# TABS DE NAVEGACIÓN
# =============================================
tab1, tab2, tab0, tab3, tab4 = st.tabs(["📊 Predicción", "📈 Gráficos", "📋 Resumen", "💡 Recomendaciones", "❓ Ayuda"])

# =============================================
# TAB 1 — PREDICCIÓN + KPIs
# =============================================
with tab1:
    # --- KPIs ---
    st.markdown("### Indicadores clave de ventas")
    total_unidades = df['cantidad_vendida'].sum()
    num_transacciones = df['venta_id'].nunique() if 'venta_id' in df.columns else len(df)
    producto_top = df.groupby('producto_nombre')['cantidad_vendida'].sum().idxmax()
    ventas_top = df.groupby('producto_nombre')['cantidad_vendida'].sum().max()
    num_productos = df['producto_id'].nunique()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("🧾 Transacciones", f"{int(num_transacciones)}")
    kpi2.metric("🛒 Unidades vendidas", f"{int(total_unidades)}")
    kpi3.metric("🏆 Producto más vendido", producto_top, f"{int(ventas_top)} unidades")
    kpi4.metric("📦 Productos únicos", f"{num_productos}")

    st.markdown("---")

    # --- Predicción del producto seleccionado ---
    st.markdown(f"### Predicción mensual: **{producto_seleccionado}**")
    st.dataframe(
        df_sel[['producto_nombre'] + meses_nombres + ['cantidad_estimada']].rename(
            columns={'producto_nombre': 'Producto', 'cantidad_estimada': 'Total estimado'}
        ),
        use_container_width=True,
        hide_index=True
    )

    # --- Gráfico circular interactivo (Plotly) ---
    st.markdown("#### Distribución mensual estimada")
    valores = df_sel[meses_nombres].iloc[0].values
    fig_pie = go.Figure(data=[go.Pie(
        labels=meses_nombres,
        values=valores,
        hole=0.5,
        marker_colors=['#4CAF50', '#FF9800', '#2196F3'],
        textinfo='label+percent+value',
        textfont_size=14
    )])
    fig_pie.update_layout(
        title_text=f"Ventas estimadas por mes — {producto_seleccionado}",
        showlegend=True,
        height=550
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# =============================================
# TAB 2 — GRÁFICOS INTERACTIVOS
# =============================================
with tab2:
    st.markdown("### Gráfico de barras — Cantidad estimada por producto")

    df_grafico = st.session_state.df_predicciones.copy()
    fig_bar = px.bar(
        df_grafico,
        x='producto_nombre',
        y='cantidad_estimada',
        color='cantidad_estimada',
        color_continuous_scale='Viridis',
        labels={'producto_nombre': 'Producto', 'cantidad_estimada': 'Cantidad estimada'},
        title='Predicción de ventas para los próximos 3 meses'
    )
    fig_bar.update_xaxes(title_text='Producto')
    fig_bar.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # --- Gráfico comparativo mensual ---
    st.markdown("### Comparativa mensual por producto")
    df_melt = df_grafico.melt(
        id_vars=['producto_nombre'],
        value_vars=meses_nombres,
        var_name='Mes',
        value_name='Cantidad'
    )
    fig_group = px.bar(
        df_melt,
        x='producto_nombre',
        y='Cantidad',
        color='Mes',
        barmode='group',
        color_discrete_map={meses_nombres[0]: '#4CAF50', meses_nombres[1]: '#FF9800', meses_nombres[2]: '#2196F3'},
        labels={'producto_nombre': 'Producto'},
        title='Ventas estimadas por mes y producto'
    )
    fig_group.update_xaxes(title_text='Producto')
    fig_group.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_group, use_container_width=True)

    st.markdown("---")

    # --- Top 5 productos ---
    st.markdown("### Top 5 productos con mayor predicción")
    top5 = df_grafico.nlargest(5, 'cantidad_estimada')
    fig_top5 = px.bar(
        top5,
        x='producto_nombre',
        y='cantidad_estimada',
        color='producto_nombre',
        labels={'producto_nombre': 'Producto', 'cantidad_estimada': 'Cantidad estimada'},
        title='Top 5 productos más vendidos (predicción)'
    )
    fig_top5.update_xaxes(title_text='Producto')
    fig_top5.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_top5, use_container_width=True)

# =============================================
# TAB 0 — RESUMEN GENERAL
# =============================================
with tab0:
    st.markdown("### 📋 Resumen general de predicción")
    st.markdown("Vista global de lo que el modelo estima para los próximos 3 meses.")

    df_pred = st.session_state.df_predicciones.copy()
    total_estimado = df_pred['cantidad_estimada'].sum()
    promedio = df_pred['cantidad_estimada'].mean()
    prod_max = df_pred.loc[df_pred['cantidad_estimada'].idxmax()]
    prod_min = df_pred.loc[df_pred['cantidad_estimada'].idxmin()]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total estimado (3 meses)", f"{int(total_estimado)} unidades")
    r2.metric("Promedio por producto", f"{int(promedio)} unidades")
    r3.metric("Mayor demanda", prod_max['producto_nombre'], f"{int(prod_max['cantidad_estimada'])} unidades")
    r4.metric("Menor demanda", prod_min['producto_nombre'], f"{int(prod_min['cantidad_estimada'])} unidades")

    st.markdown("---")

    # Tabla completa de predicciones
    st.markdown("### Predicción por producto")
    st.dataframe(
        df_pred[['producto_nombre'] + meses_nombres + ['cantidad_estimada']].rename(
            columns={'producto_nombre': 'Producto', 'cantidad_estimada': 'Total estimado'}
        ),
        use_container_width=True,
        hide_index=True
    )

# =============================================
# TAB 3 — RECOMENDACIONES AUTOMÁTICAS
# =============================================
with tab3:
    st.markdown("### 💡 Recomendaciones para tu negocio")
    st.markdown("Estas sugerencias se generan automáticamente según las predicciones del modelo de inteligencia artificial.")

    df_pred = st.session_state.df_predicciones.copy()
    umbral_alto = df_pred['cantidad_estimada'].quantile(0.75)
    umbral_bajo = df_pred['cantidad_estimada'].quantile(0.25)
    productos_alto = df_pred[df_pred['cantidad_estimada'] >= umbral_alto].sort_values('cantidad_estimada', ascending=False)
    productos_bajo = df_pred[df_pred['cantidad_estimada'] <= umbral_bajo].sort_values('cantidad_estimada', ascending=True)

    st.markdown("---")

    # --- Sección 1: Productos con alta demanda ---
    st.markdown("### 📦 Productos con alta demanda")
    st.markdown("Estos productos se estima que tendrán **mayor cantidad de ventas**. Se recomienda **asegurar stock suficiente** para no perder ventas.")
    for _, row in productos_alto.iterrows():
        st.markdown(f"- 🟢 **{row['producto_nombre']}** — {int(row['cantidad_estimada'])} unidades estimadas")

    st.markdown("---")

    # --- Sección 2: Productos con baja demanda ---
    st.markdown("### 🏷️ Productos con baja demanda")
    st.markdown("Estos productos se estima que tendrán **pocas ventas**. Se recomienda considerar **promociones, descuentos o combos** para impulsar sus ventas.")
    for _, row in productos_bajo.iterrows():
        st.markdown(f"- 🔴 **{row['producto_nombre']}** — {int(row['cantidad_estimada'])} unidades estimadas")

    st.markdown("---")

    # --- Sección 3: Consejos generales ---
    st.markdown("### 📌 Consejos generales")
    st.info("**Tip:** Mientras más datos de ventas tenga el sistema, más precisas serán las predicciones. Se recomienda re-entrenar el modelo cada mes.")
    st.info("**Tip:** Revisa las predicciones junto con el inventario actual para tomar decisiones de compra informadas.")
    st.info("**Tip:** Los productos con alta demanda son ideales para exhibir en zonas visibles de la tienda.")

# =============================================
# TAB 4 — AYUDA Y DOCUMENTACIÓN
# =============================================
with tab4:
    st.markdown("## ❓ Ayuda y documentación")
    st.markdown("""
    ### ¿Qué hace esta aplicación?
    Este dashboard utiliza **Machine Learning** (aprendizaje automático) para **predecir ventas futuras** 
    basándose en **datos históricos reales** de la tienda. El sistema analiza patrones de ventas pasadas 
    — como qué productos se venden más, en qué épocas del año y en qué cantidades — para estimar 
    cuántas unidades de cada producto se venderán en los **próximos 3 meses**.

    **Conceptos clave:**
    - **Machine Learning:** Es una rama de la inteligencia artificial que permite a las computadoras aprender de datos sin ser programadas explícitamente.
    - **Predicción basada en datos históricos:** El modelo aprende de las ventas pasadas (producto, fecha, cantidad) para proyectar el comportamiento futuro.
    - **Red Neuronal Artificial:** El tipo de modelo utilizado, inspirado en el funcionamiento del cerebro humano, capaz de detectar patrones complejos en los datos.

    ### ¿De dónde vienen los datos?
    Los datos de ventas se obtienen automáticamente desde el sistema de la tienda (Laravel).
    No necesitas cargar archivos ni hacer nada manual.

    ### ¿Cómo usar el dashboard?
    1. **Panel izquierdo:** Elige un producto para ver su predicción detallada.
    2. **Predicción:** Muestra los indicadores principales y el desglose por mes del producto elegido.
    3. **Gráficos:** Compara productos con gráficos interactivos (barras, comparativo mensual, top 5).
    4. **Recomendaciones:** Te dice qué productos necesitan más stock y cuáles podrían necesitar promociones.

    ### ¿Qué significan los indicadores?
    - **Transacciones:** Número de ventas realizadas (una venta puede tener varios productos).
    - **Unidades vendidas:** Total de productos vendidos (suma de todas las cantidades).
    - **Producto más vendido:** El producto con más unidades acumuladas.
    - **Productos únicos:** Cuántos productos diferentes se han vendido.

    ### ¿Cuándo se actualiza el modelo?
    El modelo se actualiza ejecutando `training.py` cuando hay nuevos datos de ventas.
    Se recomienda hacerlo **una vez al mes** o cuando haya cambios importantes.

    ### Modelo de inteligencia artificial
    | Aspecto | Detalle |
    |---|---|
    | Tipo | Red Neuronal Artificial (ANN) |
    | Framework | TensorFlow / Keras |
    | Arquitectura | 3 capas (64 → 32 → 1 neurona) |
    | Datos de entrada | Producto, año, mes, día |
    | Predicción | Cantidad de unidades a vender |

    ### Tecnologías utilizadas
    | Componente | Tecnología |
    |---|---|
    | Dashboard | Streamlit |
    | Sistema de tienda | Laravel (PHP) |
    | Inteligencia Artificial | TensorFlow / Keras |
    | Gráficos | Plotly |
    | Datos | Pandas, NumPy |

    ### Contacto
    Si tienes dudas o sugerencias, contacta a los desarrolladores:
    - **Bryan Amaya**
    - **Karen Perez**
    """)
