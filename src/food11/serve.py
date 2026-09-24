"""Serve the registered Food-11 champion model through FastAPI."""

import os
from io import BytesIO

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

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

preprocess = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(title="Food-11 Model API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    image_batch = preprocess(image).unsqueeze(0).numpy()
    prediction = model.predict(image_batch)

    logits = torch.as_tensor(np.asarray(prediction)).squeeze(0)
    probabilities = torch.softmax(logits, dim=0)
    confidence, class_index = torch.max(probabilities, dim=0)

    return {
        "category": CATEGORIES[class_index.item()],
        "confidence": confidence.item(),
    }