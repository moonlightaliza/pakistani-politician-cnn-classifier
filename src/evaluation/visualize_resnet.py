"""
visualize.py — Evaluation Visualiser for ResNet-50
Reads:
  - src/evaluation/eval_report.json       (required — written by evaluate.py)
  - src/models/resnet/training_log.json   (optional — written by train_resnet.py)

Produces (saved to src/evaluation/):
  - per_class_metrics.png    — grouped bar chart: precision / recall / F1 per class
  - confusion_matrix.png     — heatmap rendered from eval_report.json
  - summary_card.png         — dark-theme summary card
  - training_curves.png      — loss + accuracy curves (ONLY if training_log.json exists)

Run from project root:
    python3 src/evaluation/visualize.py
"""

import sys, os, json
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Config ─────────────────────────────────────────────────────────────────────
LOG_PATH        = "src/models/resnet/training_log.json"
REPORT_PATH     = "src/evaluation/eval_report.json"
OUT_DIR         = "src/evaluation"

CURVES_OUT      = os.path.join(OUT_DIR, "training_curves.png")
PER_CLASS_OUT   = os.path.join(OUT_DIR, "per_class_metrics.png")
CONF_MAT_OUT    = os.path.join(OUT_DIR, "confusion_matrix.png")
SUMMARY_OUT     = os.path.join(OUT_DIR, "summary_card.png")

PHASE_A_EPOCHS  = 5      # must match train_resnet.py (only used if log exists)
PALETTE         = {
    "train":     "#4C9BE8",
    "val":       "#E87D4C",
    "phaseA":    "#C8E6C9",
    "phaseB":    "#FFF9C4",
    "precision": "#5B8FF9",
    "recall":    "#5AD8A6",
    "f1":        "#F6BD16",
}

os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Training Curves  (skipped gracefully if training_log.json is missing)
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(log: dict):
    train_loss = log["train_loss"]
    val_loss   = log["val_loss"]
    train_acc  = log["train_acc"]
    val_acc    = log["val_acc"]
    epochs     = list(range(1, len(train_loss) + 1))
    n_total    = len(epochs)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("ResNet-50 — Training Curves", fontsize=15, fontweight="bold", y=1.01)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.32)

    for col, (metric, y_train, y_val, ylabel) in enumerate([
        ("Loss",     train_loss, val_loss, "Cross-Entropy Loss"),
        ("Accuracy", train_acc,  val_acc,  "Accuracy (%)"),
    ]):
        ax = fig.add_subplot(gs[col])

        ax.axvspan(0.5, PHASE_A_EPOCHS + 0.5, alpha=0.18,
                   color=PALETTE["phaseA"], label="Phase A (head only)")
        ax.axvspan(PHASE_A_EPOCHS + 0.5, n_total + 0.5, alpha=0.18,
                   color=PALETTE["phaseB"], label="Phase B (full fine-tune)")
        ax.axvline(x=PHASE_A_EPOCHS + 0.5, color="#888", linestyle="--",
                   linewidth=1.0, alpha=0.7)
        ax.text(PHASE_A_EPOCHS + 0.7, min(y_train + y_val),
                "Phase B →", fontsize=8, color="#666", va="bottom")

        ax.plot(epochs, y_train, "o-", color=PALETTE["train"],
                linewidth=2, markersize=4, label="Train")
        ax.plot(epochs, y_val,   "s--", color=PALETTE["val"],
                linewidth=2, markersize=4, label="Val")

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel,  fontsize=11)
        ax.set_title(metric,   fontsize=12, fontweight="bold")
        ax.set_xlim(0.5, n_total + 0.5)
        ax.grid(True, alpha=0.3)

        handles = [
            mpatches.Patch(color=PALETTE["phaseA"], alpha=0.5, label="Phase A (head only)"),
            mpatches.Patch(color=PALETTE["phaseB"], alpha=0.5, label="Phase B (full fine-tune)"),
            plt.Line2D([0], [0], color=PALETTE["train"], marker="o", label="Train"),
            plt.Line2D([0], [0], color=PALETTE["val"],   marker="s", linestyle="--", label="Val"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="best")

    plt.tight_layout()
    fig.savefig(CURVES_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved  → {CURVES_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-Class Metrics Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_class_metrics(report: dict):
    per_class   = report["per_class"]
    classes     = list(per_class.keys())
    precisions  = [per_class[c]["precision"] for c in classes]
    recalls     = [per_class[c]["recall"]    for c in classes]
    f1s         = [per_class[c]["f1-score"]  for c in classes]

    # Prettify class labels: "ahmed_sharif_chaudhry" → "Ahmed Sharif\nChaudhry"
    def fmt(name):
        parts = name.replace("_", " ").title().split()
        # wrap after 2 words if > 2 words
        if len(parts) > 2:
            return " ".join(parts[:2]) + "\n" + " ".join(parts[2:])
        return " ".join(parts)

    labels = [fmt(c) for c in classes]

    x   = np.arange(len(classes))
    w   = 0.26

    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 1.5), 6))
    ax.bar(x - w, precisions, w, label="Precision", color=PALETTE["precision"])
    ax.bar(x,     recalls,    w, label="Recall",    color=PALETTE["recall"])
    ax.bar(x + w, f1s,        w, label="F1-Score",  color=PALETTE["f1"])

    macro = report["macro_avg"]
    ax.axhline(macro["f1-score"], color="#E87D4C", linestyle="--",
               linewidth=1.2, label=f"Macro F1 = {macro['f1-score']:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Per-Class Metrics — ResNet-50  |  "
        f"Overall Acc: {report['overall_accuracy_pct']:.2f}%",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PER_CLASS_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-class metrics saved → {PER_CLASS_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Confusion Matrix  (rendered directly from eval_report.json)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(report: dict):
    cm          = np.array(report["confusion_matrix"])
    class_names = report["class_names"]

    # Prettify labels
    def short(name):
        parts = name.replace("_", " ").title().split()
        return parts[0][0] + ". " + " ".join(parts[1:]) if len(parts) > 1 else parts[0]

    labels = [short(c) for c in class_names]
    n      = len(labels)

    # Row-normalise for colour, keep raw counts as text
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(n * 0.72 + 1.5, n * 0.72 + 1.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title("Confusion Matrix — ResNet-50", fontsize=13, fontweight="bold")

    # Annotate each cell with raw count (skip zeros to keep it clean)
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val > 0:
                ax.text(j, i, str(val),
                        ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > thresh else "black")

    plt.tight_layout()
    fig.savefig(CONF_MAT_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved  → {CONF_MAT_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Summary Card  (works without training log)
# ══════════════════════════════════════════════════════════════════════════════
def plot_summary_card(report: dict, log: dict | None = None):
    overall_acc = report["overall_accuracy_pct"]
    macro       = report["macro_avg"]
    weighted    = report["weighted_avg"]

    # top-k key (e.g. top_3_accuracy_pct)
    topk_key   = next((k for k in report if k.startswith("top_") and k.endswith("_accuracy_pct")), None)
    topk_val   = report[topk_key] if topk_key else None
    topk_label = topk_key.replace("_pct", "").replace("_", " ").title() if topk_key else ""

    # Training info — shown only if log is available
    training_line = ""
    if log:
        best_val_acc = max(log["val_acc"])
        n_epochs     = len(log["val_acc"])
        training_line = (f"Best Val Acc (training): {best_val_acc:.2f}%  |  "
                         f"Total Epochs: {n_epochs}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    fig.patch.set_facecolor("#1E1E2E")

    lines = [
        ("ResNet-50 — Evaluation Summary", 0.92, 16, "#CDD6F4", "bold"),
        ("Overall Accuracy",               0.78, 12, "#A6E3A1", "normal"),
        (f"{overall_acc:.2f}%",            0.70, 26, "#A6E3A1", "bold"),
    ]
    if topk_val is not None:
        lines += [
            (topk_label,          0.59, 12, "#89DCEB", "normal"),
            (f"{topk_val:.2f}%",  0.51, 20, "#89DCEB", "bold"),
        ]
    lines += [
        (f"Macro F1: {macro['f1-score']:.4f}  |  "
         f"Macro Precision: {macro['precision']:.4f}  |  "
         f"Macro Recall: {macro['recall']:.4f}",
         0.38, 10, "#CBA6F7", "normal"),
        (f"Weighted F1: {weighted['f1-score']:.4f}  |  "
         f"Weighted Precision: {weighted['precision']:.4f}  |  "
         f"Weighted Recall: {weighted['recall']:.4f}",
         0.28, 10, "#F5C2E7", "normal"),
    ]
    if training_line:
        lines.append((training_line, 0.16, 9, "#FAB387", "normal"))

    for text, y, size, color, weight in lines:
        ax.text(0.5, y, text, transform=ax.transAxes,
                fontsize=size, color=color, fontweight=weight,
                ha="center", va="center")

    plt.tight_layout()
    fig.savefig(SUMMARY_OUT, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Summary card saved      → {SUMMARY_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load evaluation report (required) ─────────────────────────────────────
    if not os.path.exists(REPORT_PATH):
        print(f"[ERROR] Evaluation report not found: {REPORT_PATH}")
        print("        Run evaluate.py first.")
        sys.exit(1)
    with open(REPORT_PATH) as f:
        report = json.load(f)

    # ── Load training log (optional) ───────────────────────────────────────────
    log = None
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log = json.load(f)
        print(f"  Training log found     → {LOG_PATH}")
    else:
        print(f"  [INFO] Training log not found ({LOG_PATH}) — skipping training curves.")
        print(f"         To generate training_curves.png, run train_resnet.py first.\n")

    print("Generating visualisations …\n")

    plot_per_class_metrics(report)
    plot_confusion_matrix(report)
    plot_summary_card(report, log)

    if log:
        plot_training_curves(log)

    print("\nDone. All plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()