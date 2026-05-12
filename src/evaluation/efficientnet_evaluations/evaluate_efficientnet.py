"""
evaluate.py — EfficientNet-B3 Evaluation Script
Computes: accuracy, per-class precision/recall/F1, confusion matrix,
          top-k accuracy, and saves a full JSON report + PNG confusion matrix.


"""

import sys, os, json
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
VAL_DIR       = "dataset/val"
MODEL_PATH    = "src/models/efficientnet/efficientnet_politician.pth"
REPORT_PATH   = "src/evaluation/eval_report.json"
CM_IMAGE_PATH = "src/evaluation/confusion_matrix.png"
BATCH_SIZE    = 32
TOP_K         = 3
NUM_WORKERS   = 0
IMG_SIZE      = 300   # EfficientNet-B3 native size

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ── Val transform (no augmentation) ───────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, loader):
    """Run inference and return (all_preds, all_labels, all_probs)."""
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  [EVAL]", leave=False):
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs, axis=0),
    )


def plot_confusion_matrix(cm, class_names, save_path):
    fig_size = max(8, len(class_names))
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix — EfficientNet-B3", fontsize=14, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


def main():
    print(f"Using device: {device}\n")

    # ── Dataset ────────────────────────────────────────────────────────────────
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)
    class_names = val_dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Val samples: {len(val_dataset)}\n")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = load_model(num_classes)

    # ── Predictions ────────────────────────────────────────────────────────────
    preds, labels, probs = collect_predictions(model, val_loader)

    # ── Metrics ────────────────────────────────────────────────────────────────
    overall_acc = 100.0 * (preds == labels).mean()
    top_k_acc   = 100.0 * top_k_accuracy_score(labels, probs, k=min(TOP_K, num_classes))
    report_dict = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    print(f"Overall Accuracy : {overall_acc:.2f}%")
    print(f"Top-{min(TOP_K,num_classes)} Accuracy : {top_k_acc:.2f}%\n")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    # ── Per-class summary ──────────────────────────────────────────────────────
    per_class = {}
    for cls in class_names:
        m = report_dict[cls]
        per_class[cls] = {
            "precision": round(m["precision"], 4),
            "recall":    round(m["recall"],    4),
            "f1-score":  round(m["f1-score"],  4),
            "support":   int(m["support"]),
        }

    # ── Save JSON report ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report = {
        "model":                           "EfficientNet-B3",
        "overall_accuracy_pct":            round(overall_acc, 4),
        f"top_{min(TOP_K,num_classes)}_accuracy_pct": round(top_k_acc, 4),
        "macro_avg":    {k: round(v, 4) for k, v in report_dict["macro avg"].items() if k != "support"},
        "weighted_avg": {k: round(v, 4) for k, v in report_dict["weighted avg"].items() if k != "support"},
        "per_class":    per_class,
        "confusion_matrix": cm.tolist(),
        "class_names":  class_names,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved → {REPORT_PATH}")

    # ── Confusion matrix plot ──────────────────────────────────────────────────
    plot_confusion_matrix(cm, class_names, CM_IMAGE_PATH)


if __name__ == "__main__":
    main()