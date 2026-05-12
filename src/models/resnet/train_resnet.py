"""
ResNet-50 Training Script — Phase 2 + MLflow Tracking
Two-phase training:
  Phase A: Train head only (frozen backbone) — 5 epochs
  Phase B: Fine-tune full network (unfrozen) — 20 epochs

Run from project root:
    python3 src/models/resnet/train_resnet.py
"""

import sys, os, json
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# MLflow tracking (keep this)
import mlflow
import mlflow.pytorch

from mlops.mlflow.config import *
from src.augmentation.augment import get_transforms

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_DIR  = "dataset/train"
VAL_DIR    = "dataset/val"

SAVE_PATH  = "src/models/resnet/resnet50_politician.pth"
LOG_PATH   = "src/models/resnet/training_log.json"

PHASE_A_EPOCHS = 5
PHASE_B_EPOCHS = 20

BATCH_SIZE     = 32
LR_HEAD        = 0.001
LR_FINETUNE    = 0.0001

NUM_WORKERS    = 0
SEED           = 42

EXPERIMENT_NAME = "Pakistani Politician CNN Classifier"

torch.manual_seed(SEED)

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    is_train = phase == "train"

    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, labels in tqdm(loader, desc=f"  [{phase.upper()}]", leave=False):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc  = 100 * correct / len(loader.dataset)

    return epoch_loss, epoch_acc


def main():

    print(f"Using device: {device}\n")

    # ── MLflow Setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=get_transforms("train")
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=get_transforms("val")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    num_classes = len(train_dataset.classes)

    print(
        f"Classes: {num_classes} | "
        f"Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)}\n"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    )

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # ── Training History ──────────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    def _log(t_loss, t_acc, v_loss, v_acc):

        history["train_loss"].append(round(t_loss, 6))
        history["val_loss"].append(round(v_loss, 6))
        history["train_acc"].append(round(t_acc, 4))
        history["val_acc"].append(round(v_acc, 4))

        with open(LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)

    # ── MLflow Run ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="ResNet50_Finetuning"):

        # ── Log Hyperparameters ───────────────────────────────────────────────
        mlflow.log_params({
            "model": "ResNet50",
            "phase_a_epochs": PHASE_A_EPOCHS,
            "phase_b_epochs": PHASE_B_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_head": LR_HEAD,
            "lr_finetune": LR_FINETUNE,
            "seed": SEED,
            "num_classes": num_classes,
            "device": str(device),
        })

        # ── Phase A ───────────────────────────────────────────────────────────
        print("=" * 55)
        print("PHASE A — Training head only (backbone frozen)")
        print("=" * 55)

        optimizer = optim.Adam(
            model.fc.parameters(),
            lr=LR_HEAD
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.5
        )

        for epoch in range(1, PHASE_A_EPOCHS + 1):

            t_loss, t_acc = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                "train"
            )

            v_loss, v_acc = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                "val"
            )

            scheduler.step()

            _log(t_loss, t_acc, v_loss, v_acc)

            # ── MLflow Metrics ───────────────────────────────────────────────
            mlflow.log_metric("phaseA_train_loss", t_loss, step=epoch)
            mlflow.log_metric("phaseA_train_acc", t_acc, step=epoch)

            mlflow.log_metric("phaseA_val_loss", v_loss, step=epoch)
            mlflow.log_metric("phaseA_val_acc", v_acc, step=epoch)

            print(
                f"Epoch {epoch:02d}/{PHASE_A_EPOCHS} | "
                f"Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | "
                f"Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%"
            )

            if v_acc > best_val_acc:

                best_val_acc = v_acc

                torch.save(model.state_dict(), SAVE_PATH)

                mlflow.log_artifact(SAVE_PATH)

                print(f"  ✓ Best model saved (val_acc={v_acc:.2f}%)")

        # ── Phase B ───────────────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("PHASE B — Full fine-tuning (backbone unfrozen)")
        print("=" * 55)

        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(
            model.parameters(),
            lr=LR_FINETUNE,
            weight_decay=1e-4
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=PHASE_B_EPOCHS
        )

        for epoch in range(1, PHASE_B_EPOCHS + 1):

            total_epoch = PHASE_A_EPOCHS + epoch

            t_loss, t_acc = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                "train"
            )

            v_loss, v_acc = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                "val"
            )

            scheduler.step()

            _log(t_loss, t_acc, v_loss, v_acc)

            # ── MLflow Metrics ───────────────────────────────────────────────
            mlflow.log_metric("phaseB_train_loss", t_loss, step=total_epoch)
            mlflow.log_metric("phaseB_train_acc", t_acc, step=total_epoch)

            mlflow.log_metric("phaseB_val_loss", v_loss, step=total_epoch)
            mlflow.log_metric("phaseB_val_acc", v_acc, step=total_epoch)

            print(
                f"Epoch {epoch:02d}/{PHASE_B_EPOCHS} | "
                f"Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | "
                f"Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%"
            )

            if v_acc > best_val_acc:

                best_val_acc = v_acc

                torch.save(model.state_dict(), SAVE_PATH)

                mlflow.log_artifact(SAVE_PATH)

                print(f"  ✓ Best model saved (val_acc={v_acc:.2f}%)")

        # ── Final Logs ────────────────────────────────────────────────────────
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        mlflow.log_artifact(LOG_PATH)

        mlflow.pytorch.log_model(
            model,
            artifact_path="resnet50-model"
        )

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved        → {SAVE_PATH}")
    print(f"Training log saved → {LOG_PATH}")


if __name__ == "__main__":
    main()