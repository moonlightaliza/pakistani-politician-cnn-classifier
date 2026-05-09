"""
Train/Val/Test Split Script — Phase 1
Splits dataset/raw/ into dataset/train/, dataset/val/, dataset/test/

Split ratio: 70% train | 15% val | 15% test
Stratified per politician (preserves class balance)

Run from project root:
    python src/data_collection/split_data.py
"""

import os
import shutil
import random

RAW_DIR   = "dataset/raw"
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"
TEST_DIR  = "dataset/test"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST gets the remainder

SEED = 42
random.seed(SEED)


def split_dataset():
    politicians = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ])

    summary = []

    for politician in politicians:
        src_folder = os.path.join(RAW_DIR, politician)
        images = sorted([
            f for f in os.listdir(src_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])

        random.shuffle(images)
        total   = len(images)
        n_train = int(total * TRAIN_RATIO)
        n_val   = int(total * VAL_RATIO)

        splits = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR:   images[n_train:n_train + n_val],
            TEST_DIR:  images[n_train + n_val:],
        }

        for dest_root, files in splits.items():
            dest_folder = os.path.join(dest_root, politician)
            os.makedirs(dest_folder, exist_ok=True)
            for fname in files:
                shutil.copy2(
                    os.path.join(src_folder, fname),
                    os.path.join(dest_folder, fname),
                )

        n_test = total - n_train - n_val
        summary.append((politician, total, n_train, n_val, n_test))
        print(f"  {politician:<30} total={total} | train={n_train} val={n_val} test={n_test}")

    print(f"\nDone! Dataset split into train/val/test.")
    return summary


if __name__ == "__main__":
    print(f"Splitting dataset (seed={SEED}) — 70/15/15\n")
    split_dataset()
