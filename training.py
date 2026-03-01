
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import joblib  # Para guardar el escalador y los valores de normalización

# --- Cargar datos desde la API Laravel ---
LARAVEL_API_URL = "http://127.0.0.1:8000/api/obtener-datos-ventas"

print("Obteniendo datos desde la API...")
try:
    response = requests.get(LARAVEL_API_URL, timeout=10)
    if response.status_code == 200:
        data = pd.DataFrame(response.json())
        print(f"Datos obtenidos: {len(data)} registros")
    else:
        print(f"Error al obtener datos: código {response.status_code}")
        exit()
except Exception as e:
    print(f"No se pudo conectar con la API: {e}")
    exit()

# Renombrar columnas si es necesario (la API usa 'dia', el CSV usaba 'día')
if 'dia' in data.columns:
    data.rename(columns={'dia': 'día'}, inplace=True)

# Asegurar tipos numéricos
data['producto_id'] = pd.to_numeric(data['producto_id'], errors='coerce')
data['año'] = pd.to_numeric(data['año'], errors='coerce')
data['mes'] = pd.to_numeric(data['mes'], errors='coerce')
data['día'] = pd.to_numeric(data['día'], errors='coerce')
data['cantidad_vendida'] = pd.to_numeric(data['cantidad_vendida'], errors='coerce')
data = data.dropna()

print(f"Meses encontrados: {sorted(data['mes'].unique())}")
print(f"Años encontrados: {sorted(data['año'].unique())}")

# Características (features) y variable objetivo (target)
X = data[['producto_id', 'año', 'mes', 'día']]  # Características relevantes
y = data['cantidad_vendida']  # Variable objetivo

# Normalizar las características (X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalizar la variable objetivo (y)
y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std  # Normalizar y

# Guardar los valores de normalización de y
joblib.dump(y_mean, 'y_mean.pkl')  # Guardar el valor medio de y
joblib.dump(y_std, 'y_std.pkl')    # Guardar la desviación estándar de y

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Crear un modelo de red neuronal
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

# Ajustar el número de épocas y el tamaño del lote
epochs = 50
batch_size = 16

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

# Evaluar el modelo en el conjunto de prueba
test_loss = model.evaluate(X_test, y_test)
print(f"Loss en el conjunto de prueba: {test_loss}")

# Realizar predicciones en el conjunto de prueba
y_pred_scaled = model.predict(X_test)

# Asegurarse de que y_pred sea unidimensional
y_pred = y_pred_scaled.flatten()  # Asegurar que las predicciones sean un vector 1D

# Desnormalizar las predicciones
y_pred = y_pred * y_std + y_mean  # Deshacer la normalización

# Desnormalizar y_test
y_test = y_test * y_std + y_mean  # Deshacer la normalización de los valores reales

# Calcular y mostrar las métricas adicionales
mae = mean_absolute_error(y_test, y_pred)  # Error absoluto medio
mse = mean_squared_error(y_test, y_pred)  # Error cuadrático medio

print(f"MAE (Error Absoluto Medio): {mae}")
print(f"MSE (Error Cuadrático Medio): {mse}")

# Guardar el modelo entrenado en un archivo .h5
model.save('modelo_ventas.h5')
print("Modelo guardado como 'modelo_ventas.h5'")

# Guardar el escalador para usarlo en las predicciones
joblib.dump(scaler, 'scaler_X.pkl')
print("Escalador guardado como 'scaler_X.pkl'")
