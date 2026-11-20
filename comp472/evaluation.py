"""Shared evaluation helpers (metrics come from sklearn, which the rules allow)."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
)


def score_model(name, y_true, y_pred, fit_seconds, predict_seconds):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(
        f"{name:28s} accuracy={acc:.4f} macro-F1={f1:.4f} "
        f"fit={fit_seconds:.1f}s predict={predict_seconds:.1f}s"
    )
    return {
        "model": name, "accuracy": acc, "macro_precision": p,
        "macro_recall": r, "macro_f1": f1,
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds
    }


def print_train_gap(row):
    """Print training accuracy and the train - validation gap (over/underfitting)."""
    print(f"{'':28s} train accuracy={row['train_accuracy']:.4f} "
          f"(train - validation gap = {row['train_accuracy'] - row['accuracy']:.4f})")


def save_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cluster_purity(y_true, y_cluster):
    """Fraction of samples that belong to the majority class of their cluster."""
    total = 0
    for c in np.unique(y_cluster):
        _, counts = np.unique(y_true[y_cluster == c], return_counts=True)
        total += counts.max()
    return total / len(y_true)
