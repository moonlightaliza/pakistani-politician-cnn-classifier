"""
ResNet-50 Training Script — Phase A + Phase B with MLflow tracking
"""

import sys, os, json
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow
import mlflow.pytorch

from mlops.mlflow.config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME
from src.augmentation.augment import get_transforms


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"

SAVE_PATH = "src/models/resnet/resnet50_politician.pth"
LOG_PATH  = "src/models/resnet/training_log.json"

PHASE_A_EPOCHS = 5
PHASE_B_EPOCHS = 20

BATCH_SIZE = 32
LR_HEAD = 0.001
LR_FINETUNE = 0.0001

NUM_WORKERS = 0
SEED = 42

torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ─────────────────────────────────────────────
# Epoch Runner
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    model.train() if phase == "train" else model.eval()

    total_loss, correct = 0.0, 0
    ctx = torch.enable_grad() if phase == "train" else torch.no_grad()

    with ctx:
        for images, labels in tqdm(loader, desc=phase.upper(), leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if phase == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    return (
        total_loss / len(loader.dataset),
        100 * correct / len(loader.dataset)
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():

    print(f"\nUsing device: {device}\n")

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Data
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms("train"))
    val_ds   = datasets.ImageFolder(VAL_DIR, transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    num_classes = len(train_ds.classes)

    print(f"Classes: {num_classes} | Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def log_history(tl, ta, vl, va):
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        with open(LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)


    # ─────────────────────────────
    # MLflow Run
    # ─────────────────────────────
    with mlflow.start_run(run_name="ResNet50_Politician"):

        mlflow.log_params({
            "model": "ResNet50",
            "batch_size": BATCH_SIZE,
            "lr_head": LR_HEAD,
            "lr_finetune": LR_FINETUNE,
            "epochs_A": PHASE_A_EPOCHS,
            "epochs_B": PHASE_B_EPOCHS,
            "num_classes": num_classes,
            "device": str(device),
        })

        # ───────── Phase A ─────────
        print("\nPHASE A (Head Training)\n")

        optimizer = optim.Adam(model.fc.parameters(), lr=LR_HEAD)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        for epoch in range(PHASE_A_EPOCHS):

            tl, ta = run_epoch(model, train_loader, criterion, optimizer, "train")
            vl, va = run_epoch(model, val_loader, criterion, None, "val")

            scheduler.step()

            log_history(tl, ta, vl, va)

            mlflow.log_metric("A_train_loss", tl, step=epoch)
            mlflow.log_metric("A_train_acc", ta, step=epoch)
            mlflow.log_metric("A_val_loss", vl, step=epoch)
            mlflow.log_metric("A_val_acc", va, step=epoch)

            if va > best_val_acc:
                best_val_acc = va
                torch.save(model.state_dict(), SAVE_PATH)
                mlflow.log_artifact(SAVE_PATH)

        # ───────── Phase B ─────────
        print("\nPHASE B (Fine-tuning)\n")

        for p in model.parameters():
            p.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE_B_EPOCHS)

        for epoch in range(PHASE_B_EPOCHS):

            global_epoch = PHASE_A_EPOCHS + epoch

            tl, ta = run_epoch(model, train_loader, criterion, optimizer, "train")
            vl, va = run_epoch(model, val_loader, criterion, None, "val")

            scheduler.step()

            log_history(tl, ta, vl, va)

            mlflow.log_metric("B_train_loss", tl, step=global_epoch)
            mlflow.log_metric("B_train_acc", ta, step=global_epoch)
            mlflow.log_metric("B_val_loss", vl, step=global_epoch)
            mlflow.log_metric("B_val_acc", va, step=global_epoch)

            if va > best_val_acc:
                best_val_acc = va
                torch.save(model.state_dict(), SAVE_PATH)
                mlflow.log_artifact(SAVE_PATH)

        # ───────── Final ─────────
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_artifact(LOG_PATH)

        mlflow.pytorch.log_model(model, "resnet50_model")

    print("\nTraining complete")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved → {SAVE_PATH}")
    print(f"Logs saved  → {LOG_PATH}")


if __name__ == "__main__":
    main()