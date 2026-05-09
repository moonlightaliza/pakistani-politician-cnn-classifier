"""
Data Cleaning Script — Phase 1
Removes corrupt/unreadable images from dataset/raw/

Run from project root:
    python src/data_collection/clean_data.py
"""

import os
from PIL import Image


def clean_dataset(raw_dir: str = "dataset/raw") -> list:
    removed = []

    for politician in sorted(os.listdir(raw_dir)):
        folder = os.path.join(raw_dir, politician)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)

            # Remove non-image files
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                os.remove(fpath)
                removed.append(fpath)
                continue

            # Remove corrupt images
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception:
                os.remove(fpath)
                removed.append(fpath)

    return removed


if __name__ == "__main__":
    removed = clean_dataset()
    print(f"Removed {len(removed)} corrupt/invalid files.")
    for p in removed:
        print(f"  {p}")

    if not removed:
        print("Dataset is clean — no corrupt files found.")
