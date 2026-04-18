import os
import zipfile
import io
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from app.feature_extractor import get_color_histogram

MODEL_DIR = "models"

def train_model(zip_bytes, classifier_name):
    X = []
    y = []

    # 1. Descomprimir el archivo en memoria
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as z:
        for file_info in z.infolist():
            if file_info.filename.endswith(('.jpg', '.jpeg', '.png')):
                # El nombre de la carpeta es la etiqueta (ej: "perro/img1.jpg")
                parts = file_info.filename.split('/')
                if len(parts) > 1:
                    label = parts[-2]

                    # Leer imagen y extraer características
                    with z.open(file_info) as f:
                        img_bytes = f.read()
                        features = get_color_histogram(img_bytes)
                        X.append(features)
                        y.append(label)

    if not X:
        return {"error": "No se encontraron imágenes en el zip"}

    # 2. Preprocesamiento
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Entrenamiento (Regresión Logística según el taller)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y_encoded)

    # 4. Persistencia (Guardar en la carpeta /models)
    model_path = os.path.join(MODEL_DIR, f"{classifier_name}.pkl")
    data_to_save = {
        "model": model,
        "scaler": scaler,
        "le": le,
        "classes": le.classes_.tolist()
    }
    joblib.dump(data_to_save, model_path)

    return {"message": f"Modelo '{classifier_name}' entrenado", "classes": data_to_save["classes"]}

def predict_image(image_bytes, classifier_name):
    model_path = os.path.join(MODEL_DIR, f"{classifier_name}.pkl")
    if not os.path.exists(model_path):
        return {"error": "El clasificador no existe"}

    # Cargar modelo y herramientas
    data = joblib.load(model_path)
    model = data["model"]
    scaler = data["scaler"]
    le = data["le"]

    # Procesar imagen
    features = get_color_histogram(image_bytes).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predicción
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    label = le.inverse_transform(prediction)[0]
    confidence = float(np.max(probabilities))

    return {"label": label, "confidence": confidence}