"""
Pakistani Politician Classifier — FastAPI Backend
Supports: EfficientNet-B3 (IMG_SIZE=300) and ResNet-50 (IMG_SIZE=224)
Endpoint: POST /predict
Returns:  top-N predictions with confidence scores
"""

import os
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import uvicorn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESNET_MODEL_PATH       = os.environ.get("RESNET_MODEL_PATH",       os.path.join(BASE_DIR, "src/models/resnet/resnet50_politician.pth"))
EFFICIENTNET_MODEL_PATH = os.environ.get("EFFICIENTNET_MODEL_PATH", os.path.join(BASE_DIR, "src/models/efficientnet/efficientnet_politician.pth"))
RESNET_CLASSES_PATH     = os.environ.get("RESNET_CLASSES_PATH",     os.path.join(BASE_DIR, "src/models/resnet/classes.json"))
EFFICIENTNET_CLASSES_PATH = os.environ.get("EFFICIENTNET_CLASSES_PATH", os.path.join(BASE_DIR, "src/models/efficientnet/classes.json"))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

TOP_N = int(os.environ.get("TOP_N", 5))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

resnet_transform       = get_transform(224)
efficientnet_transform = get_transform(300)

def load_classes(path: str) -> list:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    raise FileNotFoundError(f"Classes file not found: {path}")

def load_resnet(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    state = torch.load(RESNET_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def load_efficientnet(num_classes: int) -> nn.Module:
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    state = torch.load(EFFICIENTNET_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

app = FastAPI(
    title="Pakistani Politician Classifier",
    description="CNN classifier using EfficientNet-B3 and ResNet-50",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def root():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "API is running"}

print("[INFO] Loading class names...")
RESNET_CLASSES       = load_classes(RESNET_CLASSES_PATH)
EFFICIENTNET_CLASSES = load_classes(EFFICIENTNET_CLASSES_PATH)
print(f"[INFO] ResNet classes: {len(RESNET_CLASSES)}")
print(f"[INFO] EfficientNet classes: {len(EFFICIENTNET_CLASSES)}")

print(f"[INFO] Loading ResNet-50...")
resnet_model = load_resnet(len(RESNET_CLASSES))
print("[INFO] ResNet-50 ready.")

print(f"[INFO] Loading EfficientNet-B3...")
efficientnet_model = load_efficientnet(len(EFFICIENTNET_CLASSES))
print(f"[INFO] EfficientNet-B3 ready on {device}.")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "models": {
            "resnet50":        {"classes": len(RESNET_CLASSES),       "img_size": 224},
            "efficientnet_b3": {"classes": len(EFFICIENTNET_CLASSES), "img_size": 300},
        },
    }

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model: Optional[str] = Form("resnet50"),
    top_n: Optional[int] = Form(5),
):
    model = (model or "resnet50").lower().strip()
    if model not in ("resnet50", "efficientnet_b3", "efficientnet"):
        raise HTTPException(status_code=400, detail="model must be 'resnet50' or 'efficientnet_b3'")

    use_efficientnet = model in ("efficientnet_b3", "efficientnet")
    selected_model   = efficientnet_model if use_efficientnet else resnet_model
    selected_classes = EFFICIENTNET_CLASSES if use_efficientnet else RESNET_CLASSES
    transform        = efficientnet_transform if use_efficientnet else resnet_transform
    model_label      = "EfficientNet-B3" if use_efficientnet else "ResNet-50"

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = selected_model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    k = min(top_n, len(selected_classes))
    top_probs, top_idxs = torch.topk(probs, k=k)

    predictions = [
        {
            "rank":       rank + 1,
            "politician": selected_classes[idx.item()].replace("_", " ").title(),
            "confidence": round(prob.item() * 100, 2),
        }
        for rank, (prob, idx) in enumerate(zip(top_probs, top_idxs))
    ]

    return JSONResponse({"model": model_label, "top_n": k, "predictions": predictions})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
