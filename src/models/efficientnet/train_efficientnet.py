"""
EfficientNet-B3 Training Script — Colab + Google Drive version
Two-phase training:
  Phase A: Train head only (frozen backbone) — 5 epochs
  Phase B: Fine-tune full network (unfrozen) — 30 epochs

Folder structure expected in Google Drive:
  My Drive/
  └── pakistani-politician-cnn-classifier/
      ├── train/
      └── val/

Results saved back to the same Drive folder.
"""

import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR   = "/content/drive/MyDrive/pakistani-politician-cnn-classifier"
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VAL_DIR    = os.path.join(BASE_DIR, "val")
SAVE_PATH  = os.path.join(BASE_DIR, "efficientnet_politician.pth")
LOG_PATH   = os.path.join(BASE_DIR, "training_log.json")

PHASE_A_EPOCHS = 5      # head only
PHASE_B_EPOCHS = 30     # full fine-tune (increased from 20)
BATCH_SIZE     = 32     # lower to 16 if GPU runs out of memory
LR_HEAD        = 0.001
LR_FINETUNE    = 0.00005  # lower than before for stable fine-tuning
NUM_WORKERS    = 2
SEED           = 42

torch.manual_seed(SEED)

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms (correct 300x300 native size for EfficientNet-B3) ───────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 300   # EfficientNet-B3 native size

def get_transforms(split: str = "train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((332, 332)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# Epoch runner
# ══════════════════════════════════════════════════════════════════════════════
def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    total_loss, correct = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc=f"  [{phase.upper()}]", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), 100 * correct / len(loader.dataset)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Using device: {device}\n")

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms("train"))
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=get_transforms("val"))
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    num_classes = len(train_dataset.classes)
    print(f"Classes : {num_classes}")
    print(f"Train   : {len(train_dataset)} images")
    print(f"Val     : {len(val_dataset)} images\n")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

    # Freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False

    # Lightweight head — simpler is better for small datasets
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    model = model.to(device)

    # Label smoothing helps generalization on small datasets
    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0.0

    # ── Training history ────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def _log(t_loss, t_acc, v_loss, v_acc):
        history["train_loss"].append(round(t_loss, 6))
        history["val_loss"].append(round(v_loss,   6))
        history["train_acc"].append(round(t_acc,   4))
        history["val_acc"].append(round(v_acc,     4))
        with open(LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)

    # ── Phase A: Head only ─────────────────────────────────────────────────────
    print("=" * 55)
    print("PHASE A — Training head only (backbone frozen)")
    print("=" * 55)

    optimizer = optim.Adam(model.classifier.parameters(), lr=LR_HEAD)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(1, PHASE_A_EPOCHS + 1):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, "train")
        v_loss, v_acc = run_epoch(model, val_loader,   criterion, None,      "val")
        scheduler.step()
        _log(t_loss, t_acc, v_loss, v_acc)
        print(f"Epoch {epoch:02d}/{PHASE_A_EPOCHS} | "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.2f}%")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Best model saved (val_acc={v_acc:.2f}%)")

    # ── Phase B: Full fine-tune ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PHASE B — Full fine-tuning (backbone unfrozen)")
    print("=" * 55)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    # ReduceLROnPlateau — drops LR when val acc stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    for epoch in range(1, PHASE_B_EPOCHS + 1):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, "train")
        v_loss, v_acc = run_epoch(model, val_loader,   criterion, None,      "val")
        scheduler.step(v_acc)   # plateau scheduler watches val acc
        _log(t_loss, t_acc, v_loss, v_acc)
        print(f"Epoch {epoch:02d}/{PHASE_B_EPOCHS} | "
              f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.2f}%")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Best model saved (val_acc={v_acc:.2f}%)")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved        → {SAVE_PATH}")
    print(f"Training log saved → {LOG_PATH}")


if __name__ == "__main__":
    main()