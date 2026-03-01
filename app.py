#
# Esta aplicación Streamlit consume una API REST desarrollada en Laravel
# para obtener datos históricos de ventas de una tienda de ropa.
#
# Endpoint API: http://127.0.0.1:8000/api/obtener-datos-ventas
#
# La API devuelve JSON con:
# venta_id, producto_id, producto_nombre, cantidad_vendida, año, mes, dia
#
# Estos datos se usan para predecir ventas futuras utilizando un modelo
# de TensorFlow previamente entrenado.
#
# Predicción: próximo mes (1 mes) con techo basado en promedio histórico.
#

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
# PRECIOS Y COSTOS REALES POR PRODUCTO (en Soles S/)
# =============================================
PRECIOS_PRODUCTO = {
    'Camiseta básica algodón': {'precio': 35.00, 'costo': 14.00},
    'Polo clásico cuello pique': {'precio': 55.00, 'costo': 22.00},
    'Polo manga larga': {'precio': 50.00, 'costo': 20.00},
    'Polo tipo golf': {'precio': 60.00, 'costo': 24.00},
    'Sudadera con capucha': {'precio': 75.00, 'costo': 30.00},
    'Chaqueta bomber': {'precio': 120.00, 'costo': 48.00},
    'Abrigo trench': {'precio': 150.00, 'costo': 60.00},
    'Jeans skinny ajustados': {'precio': 90.00, 'costo': 36.00},
    'Leggings deportivos': {'precio': 45.00, 'costo': 18.00},
    'Conjunto deportivo': {'precio': 85.00, 'costo': 34.00},
    'Bermudas de lino': {'precio': 55.00, 'costo': 22.00},
    'Falda plisada': {'precio': 65.00, 'costo': 26.00},
    'Vestido casual': {'precio': 80.00, 'costo': 32.00},
    'Blusa elegante': {'precio': 70.00, 'costo': 28.00},
    'Chaleco acolchado': {'precio': 95.00, 'costo': 38.00},
    'Pantalon cargo': {'precio': 85.00, 'costo': 34.00},
    'Short deportivo': {'precio': 40.00, 'costo': 16.00},
    'Camisa formal': {'precio': 75.00, 'costo': 30.00},
    'Jogger urbano': {'precio': 70.00, 'costo': 28.00},
}
PRECIO_DEFAULT = {'precio': 60.00, 'costo': 24.00}

def obtener_precio(nombre_producto):
    """Retorna precio y costo de un producto. Usa default si no está en el diccionario."""
    return PRECIOS_PRODUCTO.get(nombre_producto, PRECIO_DEFAULT)

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
            df['cantidad_vendida'] = pd.to_numeric(df['cantidad_vendida'], errors='coerce')
            df['producto_nombre'] = df['producto_nombre'].astype(str)
            return df
        else:
            st.error("Error al obtener datos desde la API.")
    except Exception as e:
        st.error(f"No se pudo conectar con la API: {e}")
    return None

# =============================================
# FUNCIONES DE PREDICCIÓN (1 MES)
# =============================================
meses_es = {1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',
             7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'}

# Calcular automáticamente el primer día del mes siguiente
hoy = datetime.now()
if hoy.month == 12:
    primer_dia = datetime(hoy.year + 1, 1, 1)
else:
    primer_dia = datetime(hoy.year, hoy.month + 1, 1)

mes_prediccion = meses_es[primer_dia.month]
anio_prediccion = primer_dia.year

def generar_fechas_1_mes(productos):
    """Genera fechas para el próximo mes por producto."""
    fechas = []
    for j in range(30):
        fecha = primer_dia + timedelta(days=j)
        for producto in productos:
            producto_id = producto['producto_id']
            fechas.append([producto_id, fecha.year, fecha.month, fecha.day])
    return np.array(fechas)

def calcular_promedio_mensual_por_producto(df):
    """Calcula el promedio mensual histórico por producto."""
    df_agrupado = df.groupby(['producto_id', 'producto_nombre', 'mes'])['cantidad_vendida'].sum().reset_index()
    promedio = df_agrupado.groupby(['producto_id', 'producto_nombre'])['cantidad_vendida'].mean().reset_index()
    promedio.rename(columns={'cantidad_vendida': 'promedio_mensual'}, inplace=True)
    return promedio

def predecir_ventas_1_mes(productos, df_historico):
    """
    Predice ventas para el próximo mes.
    Usa el modelo de IA para determinar la TENDENCIA (subir o bajar) de cada producto
    respecto a su promedio histórico. Así algunos productos crecen y otros decrecen,
    lo cual es más realista que asumir crecimiento para todos.
    """
    fechas_1_mes = generar_fechas_1_mes(productos)
    fechas_1_mes_scaled = scaler.transform(fechas_1_mes)
    predicciones_scaled = modelo.predict(fechas_1_mes_scaled)
    predicciones = predicciones_scaled * y_std + y_mean
    predicciones = np.clip(predicciones, 0, None)
    predicciones = np.round(predicciones).astype(int)

    df_predicciones = pd.DataFrame({
        'producto_id': fechas_1_mes[:, 0],
        'cantidad_estimada': predicciones.flatten()
    })
    df_producto = df_predicciones.groupby('producto_id')['cantidad_estimada'].sum().reset_index()

    # Mapear nombres
    producto_nombres = {p['producto_id']: p['producto_nombre'] for p in productos}
    df_producto['producto_nombre'] = df_producto['producto_id'].map(producto_nombres)

    # Obtener promedio histórico
    promedio_hist = calcular_promedio_mensual_por_producto(df_historico)
    df_producto = df_producto.merge(promedio_hist[['producto_id', 'promedio_mensual']], on='producto_id', how='left')

    # -------------------------------------------------------
    # Usar el modelo como FACTOR DE VARIACIÓN sobre el promedio histórico.
    # Paso 1: Normalizar las predicciones del modelo a un factor relativo.
    #   - Si un producto tiene predicción alta vs otros → factor > 1 (sube)
    #   - Si un producto tiene predicción baja vs otros → factor < 1 (baja)
    # Paso 2: Aplicar ese factor al promedio histórico real.
    # Paso 3: Limitar la variación a ±20% para que sea realista.
    # -------------------------------------------------------
    media_pred = df_producto['cantidad_estimada'].mean()
    if media_pred > 0:
        df_producto['factor_modelo'] = df_producto['cantidad_estimada'] / media_pred
    else:
        df_producto['factor_modelo'] = 1.0

    # Limitar variación: entre -15% y +15% del promedio histórico
    df_producto['factor_modelo'] = df_producto['factor_modelo'].clip(0.85, 1.15)

    # Aplicar factor al promedio histórico
    df_producto['cantidad_estimada'] = (
        df_producto['promedio_mensual'] * df_producto['factor_modelo']
    ).round().astype(int)

    # Asegurar mínimo 1 unidad
    df_producto['cantidad_estimada'] = df_producto['cantidad_estimada'].clip(lower=1)

    df_producto = df_producto.drop(columns=['factor_modelo'])
    df_producto = df_producto.sort_values(by='cantidad_estimada', ascending=False).reset_index(drop=True)
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
    st.markdown(f"<h4 style='color:#888;'>Predicción para <b>{mes_prediccion} {anio_prediccion}</b></h4>", unsafe_allow_html=True)

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
    df_pred = predecir_ventas_1_mes(productos, df)
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

st.sidebar.markdown("### �👩‍💻 Desarrolladores")
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
    # --- KPIs históricos ---
    st.markdown("### Indicadores clave de ventas (datos históricos)")
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
    st.markdown(f"### Predicción para {mes_prediccion} {anio_prediccion}: **{producto_seleccionado}**")

    if not df_sel.empty:
        cant_estimada = int(df_sel['cantidad_estimada'].values[0])
        prom_hist = int(df_sel['promedio_mensual'].values[0]) if 'promedio_mensual' in df_sel.columns else 0

        col_pred1, col_pred2 = st.columns(2)
        col_pred1.metric(f"📦 Estimación {mes_prediccion}", f"{cant_estimada} unidades")
        col_pred2.metric("📊 Promedio mensual histórico", f"{prom_hist} unidades")

# =============================================
# TAB 2 — GRÁFICOS INTERACTIVOS
# =============================================
with tab2:
    st.markdown(f"### Cantidad estimada por producto — {mes_prediccion} {anio_prediccion}")

    df_grafico = st.session_state.df_predicciones.copy()

    # --- Gráfico de barras: Predicción vs Promedio histórico ---
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=df_grafico['producto_nombre'],
        y=df_grafico['cantidad_estimada'],
        name=f'Estimación {mes_prediccion}',
        marker_color='#4CAF50'
    ))
    fig_comp.add_trace(go.Bar(
        x=df_grafico['producto_nombre'],
        y=df_grafico['promedio_mensual'],
        name='Promedio mensual histórico',
        marker_color='#FF9800'
    ))
    fig_comp.update_layout(
        barmode='group',
        xaxis_tickangle=-45,
        height=500,
        title='Predicción vs Promedio histórico por producto',
        xaxis_title='Producto',
        yaxis_title='Unidades'
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # --- Gráfico de barras simple ---
    fig_bar = px.bar(
        df_grafico,
        x='producto_nombre',
        y='cantidad_estimada',
        color='cantidad_estimada',
        color_continuous_scale='Viridis',
        labels={'producto_nombre': 'Producto', 'cantidad_estimada': 'Cantidad estimada'},
        title=f'Predicción de ventas — {mes_prediccion} {anio_prediccion}'
    )
    fig_bar.update_xaxes(title_text='Producto')
    fig_bar.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_bar, use_container_width=True)

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
        title='Top 5 productos con mayor demanda estimada'
    )
    fig_top5.update_xaxes(title_text='Producto')
    fig_top5.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_top5, use_container_width=True)

    st.markdown("---")

    # --- Gráfico circular del producto seleccionado ---
    st.markdown(f"### Distribución: {producto_seleccionado}")
    if not df_sel.empty:
        cant_est = int(df_sel['cantidad_estimada'].values[0])
        prom_hist = int(df_sel['promedio_mensual'].values[0])
        fig_pie = go.Figure(data=[go.Pie(
            labels=[f'Estimación {mes_prediccion}', 'Promedio histórico'],
            values=[cant_est, prom_hist],
            hole=0.5,
            marker_colors=['#4CAF50', '#FF9800'],
            textinfo='label+percent+value',
            textfont_size=14
        )])
        fig_pie.update_layout(
            title_text=f"{producto_seleccionado} — Estimación vs Promedio",
            showlegend=True,
            height=450
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# =============================================
# TAB 0 — RESUMEN GENERAL
# =============================================
with tab0:
    st.markdown(f"### 📋 Resumen general — {mes_prediccion} {anio_prediccion}")
    st.markdown(f"Vista global de lo que el modelo estima para **{mes_prediccion} {anio_prediccion}**.")

    df_pred = st.session_state.df_predicciones.copy()
    total_estimado = df_pred['cantidad_estimada'].sum()
    promedio = df_pred['cantidad_estimada'].mean()
    prod_max = df_pred.loc[df_pred['cantidad_estimada'].idxmax()]
    prod_min = df_pred.loc[df_pred['cantidad_estimada'].idxmin()]

    # --- KPIs de unidades ---
    st.markdown("#### 📦 Unidades")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(f"Total estimado ({mes_prediccion})", f"{int(total_estimado)} unidades")
    r2.metric("Promedio por producto", f"{int(promedio)} unidades")
    r3.metric("Mayor demanda", prod_max['producto_nombre'], f"{int(prod_max['cantidad_estimada'])} unidades")
    r4.metric("Menor demanda", prod_min['producto_nombre'], f"{int(prod_min['cantidad_estimada'])} unidades")

    st.markdown("---")

    # --- KPIs financieros ---
    df_tabla = df_pred[['producto_nombre', 'cantidad_estimada', 'promedio_mensual']].copy()
    df_tabla['precio'] = df_tabla['producto_nombre'].apply(lambda x: obtener_precio(x)['precio'])
    df_tabla['costo'] = df_tabla['producto_nombre'].apply(lambda x: obtener_precio(x)['costo'])
    df_tabla['ingreso_est'] = (df_tabla['cantidad_estimada'] * df_tabla['precio']).round(2)
    df_tabla['costo_est'] = (df_tabla['cantidad_estimada'] * df_tabla['costo']).round(2)
    df_tabla['beneficio_est'] = (df_tabla['ingreso_est'] - df_tabla['costo_est']).round(2)

    ingreso_total = df_tabla['ingreso_est'].sum()
    costo_total = df_tabla['costo_est'].sum()
    beneficio_total = ingreso_total - costo_total
    margen_pct = (beneficio_total / ingreso_total * 100) if ingreso_total > 0 else 0

    st.markdown("#### 💰 Proyección financiera")
    st.caption("Calculada con precios y costos reales por producto de la tienda.")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Ingresos estimados", f"S/ {ingreso_total:,.2f}")
    f2.metric("Costos estimados", f"S/ {costo_total:,.2f}")
    f3.metric("Beneficio neto", f"S/ {beneficio_total:,.2f}")
    f4.metric("Margen de beneficio", f"{margen_pct:.1f}%")

    st.markdown("---")

    # --- Tabla completa con finanzas ---
    st.markdown("### Predicción por producto")
    st.dataframe(
        df_tabla[['producto_nombre', 'cantidad_estimada', 'precio', 'ingreso_est', 'costo_est', 'beneficio_est']].rename(
            columns={
                'producto_nombre': 'Producto',
                'cantidad_estimada': f'Unidades {mes_prediccion}',
                'precio': 'Precio unit. (S/)',
                'ingreso_est': 'Ingreso (S/)',
                'costo_est': 'Costo (S/)',
                'beneficio_est': 'Beneficio (S/)'
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --- Gráfico financiero ---
    st.markdown("### Ingresos vs Costos vs Beneficio por producto")
    fig_fin = go.Figure()
    fig_fin.add_trace(go.Bar(x=df_tabla['producto_nombre'], y=df_tabla['ingreso_est'], name='Ingreso', marker_color='#4CAF50'))
    fig_fin.add_trace(go.Bar(x=df_tabla['producto_nombre'], y=df_tabla['costo_est'], name='Costo', marker_color='#F44336'))
    fig_fin.add_trace(go.Bar(x=df_tabla['producto_nombre'], y=df_tabla['beneficio_est'], name='Beneficio', marker_color='#2196F3'))
    fig_fin.update_layout(barmode='group', xaxis_tickangle=-45, height=500, xaxis_title='Producto', yaxis_title='Monto (S/)')
    st.plotly_chart(fig_fin, use_container_width=True)

# =============================================
# TAB 3 — RECOMENDACIONES AUTOMÁTICAS
# =============================================
with tab3:
    st.markdown("### 💡 Recomendaciones para tu negocio")
    st.markdown(f"Sugerencias automáticas basadas en la predicción de **{mes_prediccion} {anio_prediccion}**.")

    df_pred = st.session_state.df_predicciones.copy()
    umbral_alto = df_pred['cantidad_estimada'].quantile(0.75)
    umbral_bajo = df_pred['cantidad_estimada'].quantile(0.25)
    productos_alto = df_pred[df_pred['cantidad_estimada'] >= umbral_alto].sort_values('cantidad_estimada', ascending=False)
    productos_bajo = df_pred[df_pred['cantidad_estimada'] <= umbral_bajo].sort_values('cantidad_estimada', ascending=True)

    st.markdown("---")

    # --- Productos con alta demanda ---
    st.markdown("### 📦 Productos con alta demanda")
    st.markdown("Se recomienda **asegurar stock suficiente** para estos productos.")
    for _, row in productos_alto.iterrows():
        st.markdown(f"- 🟢 **{row['producto_nombre']}** — {int(row['cantidad_estimada'])} unidades estimadas (promedio histórico: {int(row['promedio_mensual'])})")

    st.markdown("---")

    # --- Productos con baja demanda ---
    st.markdown("### 🏷️ Productos con baja demanda")
    st.markdown("Considera **promociones, descuentos o combos** para impulsar sus ventas.")
    for _, row in productos_bajo.iterrows():
        st.markdown(f"- 🔴 **{row['producto_nombre']}** — {int(row['cantidad_estimada'])} unidades estimadas (promedio histórico: {int(row['promedio_mensual'])})")

    st.markdown("---")

    # --- Consejos ---
    st.markdown("### 📌 Consejos generales")
    st.info("**Tip:** Mientras más datos de ventas tenga el sistema, más precisas serán las predicciones. Se recomienda re-entrenar el modelo cada mes.")
    st.info("**Tip:** Revisa las predicciones junto con el inventario actual para tomar decisiones de compra informadas.")
    st.info("**Tip:** Los productos con alta demanda son ideales para exhibir en zonas visibles de la tienda.")

# =============================================
# TAB 4 — AYUDA Y DOCUMENTACIÓN
# =============================================
with tab4:
    st.markdown("## ❓ Ayuda y documentación")
    st.markdown(f"""
    ### ¿Qué hace esta aplicación?
    Este dashboard utiliza **Machine Learning** (aprendizaje automático) para **predecir ventas futuras**
    basándose en **datos históricos reales** de la tienda. El sistema analiza patrones de ventas pasadas
    — como qué productos se venden más, en qué épocas del año y en qué cantidades — para estimar
    cuántas unidades de cada producto se venderán en el **próximo mes ({mes_prediccion} {anio_prediccion})**.

    **¿Cómo se controlan las predicciones?**
    El modelo calcula una **tendencia** para cada producto (si sube o baja respecto a su promedio).
    Luego aplica esa tendencia al **promedio mensual histórico real**, con una variación máxima
    de ±15%. Así algunos productos suben y otros bajan, reflejando un comportamiento realista.

    **Conceptos clave:**
    - **Machine Learning:** Rama de la inteligencia artificial que permite a las computadoras aprender de datos.
    - **Predicción basada en datos históricos:** El modelo aprende de las ventas pasadas para proyectar el futuro.
    - **Red Neuronal Artificial:** Modelo inspirado en el cerebro humano, capaz de detectar patrones en los datos.

    ### ¿De dónde vienen los datos?
    Los datos de ventas se obtienen automáticamente desde el sistema de la tienda (Laravel).
    No necesitas cargar archivos ni hacer nada manual.

    ### ¿Cómo usar el dashboard?
    1. **Panel izquierdo:** Elige un producto para ver su predicción detallada.
    2. **Predicción:** Indicadores principales y estimación del producto elegido.
    3. **Gráficos:** Compara productos con gráficos interactivos (predicción vs histórico, barras, top 5).
    4. **Resumen:** Vista global de lo que se espera para el próximo mes.
    5. **Recomendaciones:** Sugerencias de stock y promociones basadas en las predicciones.

    ### ¿Qué significan los indicadores?
    - **Transacciones:** Número de ventas realizadas (una venta puede tener varios productos).
    - **Unidades vendidas:** Total de productos vendidos (suma de todas las cantidades).
    - **Producto más vendido:** El producto con más unidades acumuladas.
    - **Productos únicos:** Cuántos productos diferentes se han vendido.
    - **Estimación:** Cantidad que el modelo predice que se venderá el próximo mes.
    - **Promedio mensual histórico:** Cuántas unidades se venden en promedio por mes según datos reales.

    ### ¿Qué significan los indicadores financieros?
    - **Ingresos estimados:** Dinero que se espera recibir (unidades × precio de venta de cada producto). En soles (S/).
    - **Costos estimados:** Lo que cuesta adquirir esas unidades (unidades × costo de cada producto).
    - **Beneficio neto:** La ganancia esperada (ingresos − costos).
    - **Margen de beneficio:** Porcentaje de ganancia respecto al ingreso total.
    - Cada producto tiene su **precio y costo real** asignado según los valores de la tienda.

    ### ¿Cuándo se actualiza el modelo?
    Ejecuta `training.py` cuando haya nuevos datos de ventas.
    Se recomienda hacerlo **una vez al mes**.

    ### Modelo de inteligencia artificial
    | Aspecto | Detalle |
    |---|---|
    | Tipo | Red Neuronal Artificial (ANN) |
    | Framework | TensorFlow / Keras |
    | Arquitectura | 3 capas (64 → 32 → 1 neurona) |
    | Datos de entrada | Producto, año, mes, día |
    | Predicción | Unidades a vender en el próximo mes |
    | Control | Variación ±15% sobre promedio histórico |

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
