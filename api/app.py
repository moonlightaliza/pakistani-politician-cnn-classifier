"""
Pakistani Politician Classifier — Flask REST API
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
from flask import Flask, request, jsonify

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.environ.get("MODEL_PATH",   "src/models/resnet/resnet50_politician.pth")
CLASSES_PATH = os.environ.get("CLASSES_PATH", "src/models/resnet/classes.json")
TOP_N        = int(os.environ.get("TOP_N", 5))
IMG_SIZE     = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ─────────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Load class names ───────────────────────────────────────────────────────────
def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH) as f:
            return json.load(f)
    # Fallback: read from dataset/train folder structure
    train_dir = os.environ.get("TRAIN_DIR", "dataset/train")
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    # Cache it
    os.makedirs(os.path.dirname(CLASSES_PATH), exist_ok=True)
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)
    return classes

# ── Load model ─────────────────────────────────────────────────────────────────
def load_model(num_classes: int):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# ── App init ───────────────────────────────────────────────────────────────────
app = Flask(__name__)

print(f"[INFO] Loading class names...")
CLASS_NAMES = load_classes()
NUM_CLASSES = len(CLASS_NAMES)
print(f"[INFO] {NUM_CLASSES} classes loaded.")

print(f"[INFO] Loading model from {MODEL_PATH}...")
model = load_model(NUM_CLASSES)
print(f"[INFO] Model ready on {device}.")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "num_classes": NUM_CLASSES, "device": str(device)})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file found. Send as multipart/form-data with key 'image'."}), 400

    file = request.files["image"]
    top_n = int(request.form.get("top_n", TOP_N))

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    tensor = preprocess(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)                          # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)[0]       # (num_classes,)

    top_probs, top_idxs = torch.topk(probs, k=min(top_n, NUM_CLASSES))

    predictions = [
        {
            "rank":        rank + 1,
            "politician":  CLASS_NAMES[idx.item()],
            "confidence":  round(prob.item() * 100, 2),   # percentage
        }
        for rank, (prob, idx) in enumerate(zip(top_probs, top_idxs))
    ]

    return jsonify({
        "top_n":       top_n,
        "predictions": predictions,
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)