"""Serve the registered Food-11 champion model through FastAPI."""

import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = "models:/food11@champion"

CATEGORIES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if "SKIP_MODEL" not in os.environ:
        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        model = mlflow.pyfunc.load_model(MODEL_URI)
    try:
        yield
    finally:
        model = None


app = FastAPI(title="Food-11 Model API", lifespan=lifespan)


def preprocess(image: Image.Image) -> np.ndarray:
    resized = image.resize((128, 128), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    normalized = (pixels - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    logits = np.asarray(model.predict(preprocess(image))).squeeze()
    scores = np.exp(logits - logits.max())
    probabilities = scores / scores.sum()
    class_index = int(probabilities.argmax())

    return {
        "category": CATEGORIES[class_index],
        "confidence": float(probabilities[class_index]),
    }
