"""
EfficientNet-B3 Training Script + MLflow Tracking
Two-phase training:
  Phase A: Train classifier head only (frozen backbone)
  Phase B: Fine-tune full network (unfrozen)

Google Drive Structure:
  MyDrive/
  └── pakistani-politician-cnn-classifier/
      ├── train/
      ├── val/
      └── mlruns/

Run in Colab after mounting Google Drive.
"""

import os
import json
import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = "/content/drive/MyDrive/pakistani-politician-cnn-classifier"

TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VAL_DIR    = os.path.join(BASE_DIR, "val")

SAVE_PATH  = os.path.join(BASE_DIR, "efficientnet_b3_best.pth")
LOG_PATH   = os.path.join(BASE_DIR, "training_log.json")

# MLflow
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")

EXPERIMENT_NAME = "EfficientNet-B3-Politician-Classifier"

# Training
PHASE_A_EPOCHS = 5
PHASE_B_EPOCHS = 30

BATCH_SIZE     = 32
LR_HEAD        = 0.001
LR_FINETUNE    = 0.00005

NUM_WORKERS    = 2
SEED           = 42

IMG_SIZE       = 300

torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {device}\n")


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split="train"):

    if split == "train":
        return transforms.Compose([
            transforms.Resize((332, 332)),
            transforms.RandomCrop(IMG_SIZE),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),

            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
            ),

            transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.3
            ),

            transforms.RandomGrayscale(p=0.05),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            ),
        ])

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# EPOCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer=None, phase="train"):

    is_train = phase == "train"

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:

        for images, labels in tqdm(
            loader,
            desc=f"[{phase.upper()}]",
            leave=False
        ):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc  = 100 * correct / len(loader.dataset)

    return epoch_loss, epoch_acc


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    # ──────────────────────────────────────────────────────────────────────────
    # DATASETS
    # ──────────────────────────────────────────────────────────────────────────

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

    print(f"Classes : {num_classes}")
    print(f"Train   : {len(train_dataset)}")
    print(f"Val     : {len(val_dataset)}\n")


    # ──────────────────────────────────────────────────────────────────────────
    # MODEL
    # ──────────────────────────────────────────────────────────────────────────

    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
    )

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0


    # ──────────────────────────────────────────────────────────────────────────
    # HISTORY
    # ──────────────────────────────────────────────────────────────────────────

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    def save_history(
        train_loss,
        train_acc,
        val_loss,
        val_acc
    ):

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))

        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        with open(LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)


    # ══════════════════════════════════════════════════════════════════════════
    # MLFLOW SETUP
    # ══════════════════════════════════════════════════════════════════════════

    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        # ──────────────────────────────────────────────────────────────────────
        # LOG PARAMETERS
        # ──────────────────────────────────────────────────────────────────────

        mlflow.log_param("model", "EfficientNet-B3")
        mlflow.log_param("img_size", IMG_SIZE)

        mlflow.log_param("batch_size", BATCH_SIZE)

        mlflow.log_param("phase_a_epochs", PHASE_A_EPOCHS)
        mlflow.log_param("phase_b_epochs", PHASE_B_EPOCHS)

        mlflow.log_param("lr_head", LR_HEAD)
        mlflow.log_param("lr_finetune", LR_FINETUNE)

        mlflow.log_param("optimizer_phase_a", "Adam")
        mlflow.log_param("optimizer_phase_b", "Adam")

        mlflow.log_param("loss_function", "CrossEntropyLoss")
        mlflow.log_param("label_smoothing", 0.1)

        mlflow.log_param("num_classes", num_classes)


        # ══════════════════════════════════════════════════════════════════════
        # PHASE A
        # ══════════════════════════════════════════════════════════════════════

        print("=" * 60)
        print("PHASE A — Training classifier head only")
        print("=" * 60)

        optimizer = optim.Adam(
            model.classifier.parameters(),
            lr=LR_HEAD
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.5
        )

        for epoch in range(1, PHASE_A_EPOCHS + 1):

            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                "train"
            )

            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                "val"
            )

            scheduler.step()

            save_history(
                train_loss,
                train_acc,
                val_loss,
                val_acc
            )

            # MLflow logging
            mlflow.log_metric("phase_a_train_loss", train_loss, step=epoch)
            mlflow.log_metric("phase_a_train_acc", train_acc, step=epoch)

            mlflow.log_metric("phase_a_val_loss", val_loss, step=epoch)
            mlflow.log_metric("phase_a_val_acc", val_acc, step=epoch)

            print(
                f"Epoch {epoch:02d}/{PHASE_A_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                torch.save(model.state_dict(), SAVE_PATH)

                mlflow.log_artifact(SAVE_PATH)

                print(f"✓ Best model saved ({val_acc:.2f}%)")


        # ══════════════════════════════════════════════════════════════════════
        # PHASE B
        # ══════════════════════════════════════════════════════════════════════

        print("\n" + "=" * 60)
        print("PHASE B — Full fine-tuning")
        print("=" * 60)

        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(
            model.parameters(),
            lr=LR_FINETUNE,
            weight_decay=1e-4
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4
        )

        for epoch in range(1, PHASE_B_EPOCHS + 1):

            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                "train"
            )

            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                "val"
            )

            scheduler.step(val_acc)

            save_history(
                train_loss,
                train_acc,
                val_loss,
                val_acc
            )

            total_epoch = PHASE_A_EPOCHS + epoch

            # MLflow logging
            mlflow.log_metric("phase_b_train_loss", train_loss, step=total_epoch)
            mlflow.log_metric("phase_b_train_acc", train_acc, step=total_epoch)

            mlflow.log_metric("phase_b_val_loss", val_loss, step=total_epoch)
            mlflow.log_metric("phase_b_val_acc", val_acc, step=total_epoch)

            print(
                f"Epoch {epoch:02d}/{PHASE_B_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                torch.save(model.state_dict(), SAVE_PATH)

                mlflow.log_artifact(SAVE_PATH)

                print(f"✓ Best model saved ({val_acc:.2f}%)")


        # ──────────────────────────────────────────────────────────────────────
        # FINAL LOGGING
        # ──────────────────────────────────────────────────────────────────────

        mlflow.log_metric("best_val_accuracy", best_val_acc)

        mlflow.log_artifact(LOG_PATH)

        mlflow.pytorch.log_model(
            model,
            artifact_path="efficientnet_b3_model"
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        print(f"Best Validation Accuracy : {best_val_acc:.2f}%")
        print(f"Model Saved              : {SAVE_PATH}")
        print(f"Training Log Saved       : {LOG_PATH}")
        print(f"MLflow Tracking Saved    : {MLFLOW_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()