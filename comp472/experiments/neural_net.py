"""Experiment 4: Neural Network (PyTorch MLP).

Studies (rules: learning rate, hidden layers/size, dropout, weight decay, epochs):
one change at a time relative to a 256-64 base network. Per-epoch training loss
and validation accuracy are kept in extra["history"] (saved as nn_history.csv).
For the image task the input representation (z-score vs Hellinger) is also compared.
"""

import numpy as np

from ..config import QUICK_BEST
from ..evaluation import score_model
from ..models import run_neural_net
from .base import ExperimentResult

EPOCHS = 40
BASE = dict(learning_rate=1e-3, hidden=(256, 64), dropouts=(0.20, 0.10), weight_decay=1e-4)


class NeuralNetExperiment:
    model_name = "Neural Network"

    def __init__(self, quick=False, epochs=EPOCHS):
        self.quick = quick
        self.epochs = epochs

    def configs(self, prefix):
        """(label, settings). settings may contain "input": "zscore" (default) or
        "hellinger"; everything else is passed to run_neural_net."""
        if self.quick:
            return [("best settings", QUICK_BEST[prefix]["nn"])]
        return [
            ("256-64 lr=1e-3", BASE),
            ("lr=1e-2", dict(BASE, learning_rate=1e-2)),
            ("lr=1e-4", dict(BASE, learning_rate=1e-4)),
            ("512-256", dict(BASE, hidden=(512, 256))),
            ("1024-512", dict(BASE, hidden=(1024, 512))),
            ("dropout 0.5-0.3", dict(BASE, dropouts=(0.5, 0.3))),
            ("weight decay 1e-3", dict(BASE, weight_decay=1e-3)),
            # Hellinger features (image only; skipped for sentiment).
            ("hellinger input", dict(BASE, input="hellinger")),
            ("512-256 hellinger", dict(BASE, hidden=(512, 256), input="hellinger")),
        ] + self.regularization_grid() + [
            # No hidden layer = multinomial logistic regression (a linear model).
            ("linear lr=1e-3", dict(BASE, hidden=(), dropouts=())),
            ("linear lr=1e-3 wd=1e-2", dict(BASE, hidden=(), dropouts=(), weight_decay=1e-2)),
        ]

    @staticmethod
    def regularization_grid():
        """Combinations of the settings that individually reduced overfitting
        (all models peaked after 2-4 epochs on sentiment): lower learning rate,
        more dropout, stronger weight decay."""
        grid = []
        for lr in (1e-4, 3e-4):
            for dropouts in ((0.20, 0.10), (0.5, 0.3)):
                for wd in (1e-4, 1e-2):
                    label = f"lr={lr:g} dropout={dropouts[0]}-{dropouts[1]} wd={wd:g}"
                    grid.append((label, dict(BASE, learning_rate=lr, dropouts=dropouts,
                                             weight_decay=wd)))
        return grid

    @staticmethod
    def inputs(split, kind):
        """(train, validation, test) arrays for the chosen input representation,
        or None if that representation is not available for this dataset."""
        if kind == "zscore":
            return split.X_train_s, split.X_val_s, split.X_test_s
        if kind == "hellinger":
            if not split.has_hellinger:
                return None
            # Unit-norm rows have tiny entries (~1/sqrt(d)); multiplying by sqrt(d)
            # gives roughly unit-scale inputs for the network. A constant, not fitted.
            scale = np.float32(np.sqrt(split.X_train_h.shape[1]))
            return split.X_train_h * scale, split.X_val_h * scale, split.X_test_h * scale
        raise ValueError(f"Unknown neural-net input: {kind}")

    def run(self, split):
        """Returns an ExperimentResult, or None if PyTorch is not installed."""
        rows, history = [], []
        best = None  # (accuracy, label, val_pred, test_pred)
        for label, conf in self.configs(split.prefix):
            conf = dict(conf)
            arrays = self.inputs(split, conf.pop("input", "zscore"))
            if arrays is None:
                continue
            X_tr, X_va, X_te = arrays
            val_pred, test_pred, timings, info = run_neural_net(
                X_tr, split.y_train, X_va, split.y_val, X_te,
                epochs=self.epochs, verbose=False, **conf
            )
            if val_pred is None:
                break
            row = score_model(f"PyTorch MLP {label}", split.y_val, val_pred, *timings)
            row["train_accuracy"] = info["train_accuracy"]
            row["best_epoch"] = info["best_epoch"]
            row["dataset"] = split.prefix
            rows.append(row)
            print(f"{'':28s} train accuracy={info['train_accuracy']:.4f} "
                  f"best epoch={info['best_epoch']}/{self.epochs}")
            for epoch, train_loss, val_acc in info["history"]:
                history.append({
                    "dataset": split.prefix, "config": label, "epoch": epoch,
                    "train_loss": train_loss, "val_accuracy": val_acc
                })
            if best is None or row["accuracy"] > best[0]:
                best = (row["accuracy"], label, val_pred, test_pred)

        if best is None:
            return None
        _, label, val_pred, test_pred = best
        print(f"Best neural network: {label}")
        return ExperimentResult(self.model_name, rows, val_pred, label,
                                lambda: test_pred, extra={"history": history})
