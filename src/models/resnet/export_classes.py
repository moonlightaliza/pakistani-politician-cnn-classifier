"""
Export class names from dataset/train/ into src/models/resnet/classes.json
Run ONCE from project root before building Docker:
    python3 src/models/resnet/export_classes.py
"""

import os, json

TRAIN_DIR    = "dataset/train"
CLASSES_PATH = "src/models/resnet/classes.json"

classes = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])

os.makedirs(os.path.dirname(CLASSES_PATH), exist_ok=True)
with open(CLASSES_PATH, "w") as f:
    json.dump(classes, f, indent=2)

print(f"Saved {len(classes)} classes to {CLASSES_PATH}")
for c in classes:
    print(f"  {c}")