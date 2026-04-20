from fastapi import FastAPI, UploadFile, File, Form
from app.model_service import train_model, predict_image
import os

app = FastAPI(title="Clasificador de Imágenes UCEVA")

# Crear carpeta de modelos si no existe
if not os.path.exists("models"):
    os.makedirs("models")

@app.get("/health")
def health_check():
    return {"status": "ready", "engine": "scikit-learn-logistic-regression"}

@app.post("/train")
async def train(classifier_name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    result = train_model(contents, classifier_name)
    return result

@app.post("/classify")
async def classify(classifier_name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_image(contents, classifier_name)
    return result