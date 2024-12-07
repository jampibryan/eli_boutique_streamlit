#app.py

# Importaciones necesarias
import streamlit as st
import requests
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib  # Para cargar el escalador
from datetime import datetime, timedelta
import matplotlib.pyplot as plt  # Para graficar

# URL de la API de Laravel
# LARAVEL_API_URL = 'http://127.0.0.1:8000/obtener-datos-ventas'
LARAVEL_API_URL = 'https://eliboutique.firetensor.com/obtener-datos-ventas'

# Cargar el modelo previamente entrenado
modelo = tf.keras.models.load_model('modelo_ventas.h5')
scaler = joblib.load('scaler_X.pkl')  # Cargar el escalador previamente guardado

# Cargar la normalización de la variable objetivo (si la guardaste como en training.py)
y_mean = joblib.load('y_mean.pkl')  # Cargar el valor medio de y
y_std = joblib.load('y_std.pkl')  # Cargar el valor std de y

# Definir el primer día de los próximos tres meses (noviembre, diciembre y enero)
primer_dia = datetime(2024, 11, 1)  # Primer día de noviembre

# Función para generar las fechas para los tres meses usando año, mes y día de la API
def generar_fechas_3_meses(productos):
    fechas = []
    for i in range(3):  # Para tres meses: noviembre, diciembre, enero
        for j in range(30):  # Suponiendo que cada mes tiene 30 días
            fecha = primer_dia + timedelta(days=i*30 + j)  # Sumar días para cada mes
            for producto in productos:  # Usar todos los productos disponibles
                producto_id = producto['producto_id']
                # Añadir solo año, mes y día, sin usar `fecha` directamente
                fechas.append([producto_id, fecha.year, fecha.month, fecha.day])
    return np.array(fechas)

# Predecir las ventas para los tres meses
def predecir_ventas_3_meses(productos):
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
    producto_nombres = {producto['producto_id']: producto['producto_nombre'] for producto in productos}
    df_producto['producto_nombre'] = df_producto['producto_id'].map(producto_nombres)
    df_producto = df_producto.sort_values(by='cantidad_estimada', ascending=False).reset_index(drop=True)
    return df_producto

# Asignar las estimaciones mensuales (Enero, Febrero, Marzo)
def asignar_estimaciones_mensuales(df_producto):
    for index, row in df_producto.iterrows():
        total_estimado = row['cantidad_estimada']
        enero = int(total_estimado * 0.4)
        febrero = int(total_estimado * 0.35)
        marzo = total_estimado - (enero + febrero)
        df_producto.at[index, 'Enero'] = enero
        df_producto.at[index, 'Febrero'] = febrero
        df_producto.at[index, 'Marzo'] = marzo
    return df_producto

# Título de la aplicación
st.markdown("<h1 style='text-align: center;'>Predicción de ventas para los próximos 3 meses</h1>", unsafe_allow_html=True)

# Obtener los datos de ventas de la API de Laravel
try:
    response = requests.get(LARAVEL_API_URL)

    if response.status_code == 200:
        ventas = response.json()
        productos = ventas
        df = pd.DataFrame(ventas)
        df['producto_id'] = pd.to_numeric(df['producto_id'], errors='coerce')
        df['producto_nombre'] = df['producto_nombre'].astype(str)

        if 'df_predicciones' not in st.session_state:
            df_predicciones = predecir_ventas_3_meses(productos)
            st.session_state.df_predicciones = df_predicciones

        st.write("Productos con su cantidad estimada para los próximos tres meses:")
        st.session_state.df_predicciones['cantidad_estimada'] = st.session_state.df_predicciones['cantidad_estimada'].astype(int)
        st.write(st.session_state.df_predicciones[['producto_nombre', 'cantidad_estimada']])

        producto_seleccionado = st.selectbox("Seleccionar producto", st.session_state.df_predicciones['producto_nombre'])
        df_producto_seleccionado = st.session_state.df_predicciones[st.session_state.df_predicciones['producto_nombre'] == producto_seleccionado]
        df_producto_seleccionado = asignar_estimaciones_mensuales(df_producto_seleccionado)

        st.write(f"Predicción de ventas mensuales para el producto: {producto_seleccionado}")
        st.write(df_producto_seleccionado[['producto_nombre', 'Enero', 'Febrero', 'Marzo']])

        st.subheader("Gráfico circular de la cantidad estimada mensual del producto seleccionado")
        valores_mensuales = df_producto_seleccionado[['Enero', 'Febrero', 'Marzo']].iloc[0].values
        etiquetas_meses = ['Enero', 'Febrero', 'Marzo']
        colores = ['#4CAF50', '#FF9800', '#2196F3']

        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax_pie.pie(
            valores_mensuales,
            labels=etiquetas_meses,
            autopct='%1.1f%%',
            startangle=90,
            colors=colores,
            explode=[0.05, 0.05, 0.05],
            pctdistance=0.85,
            textprops={'fontsize': 12, 'color': 'black'}
        )

        for text in texts:
            text.set_fontsize(14)
            text.set_color('#333333')
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_color('white')

        ax_pie.set_title("Distribución de ventas estimadas por mes", fontsize=16, fontweight='bold', color='#333333')
        centro_circulo = plt.Circle((0, 0), 0.70, fc='white')
        fig_pie.gca().add_artist(centro_circulo)

        st.pyplot(fig_pie)

        st.subheader("Gráfico de barras de la cantidad estimada por producto")
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.bar(st.session_state.df_predicciones['producto_nombre'], st.session_state.df_predicciones['cantidad_estimada'], color='royalblue', width=0.6)
        ax.set_xlabel('Producto', fontsize=14)
        ax.set_ylabel('Cantidad Estimada', fontsize=14)
        ax.set_title('Predicción de ventas para los próximos 3 meses', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.error("Error al obtener los datos de ventas desde Laravel.")
except Exception as e:
    st.error(f"Error de conexión con la API de Laravel: {e}")

st.sidebar.markdown("### Desarrolladores:")
st.sidebar.markdown("- Bryan Amaya\n- Karen Perez")
