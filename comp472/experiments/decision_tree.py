"""Experiment 2: Decision Tree.

Studies (rules: max depth, min samples split/leaf, splitting criterion):
  (a) max_depth in {3, 5, 10, 15, 20} with Gini, min_samples_leaf=5
  (b) at the best depth: min_samples_leaf in {1, 20} and entropy instead of Gini
Training accuracy is recorded for every model so the train - validation gap
shows underfitting (both low) vs overfitting (large gap).
"""

import time

from sklearn.metrics import accuracy_score

from ..config import QUICK_BEST, SEED
from ..evaluation import print_train_gap, score_model
from ..models import ScratchDecisionTree
from .base import ExperimentResult


class DecisionTreeExperiment:
    model_name = "Decision Tree"

    def __init__(self, quick=False):
        self.quick = quick

    def fit_one(self, split, depth, min_leaf=5, criterion="gini"):
        model = ScratchDecisionTree(
            max_depth=depth, min_samples_split=20, min_samples_leaf=min_leaf,
            max_features=None, random_state=SEED, criterion=criterion
        )
        start = time.time()
        model.fit(split.X_train_s, split.y_train)
        fit_time = time.time() - start
        start = time.time()
        pred = model.predict(split.X_val_s)
        predict_time = time.time() - start
        name = f"Scratch Decision Tree depth={depth} leaf={min_leaf} {criterion}"
        row = score_model(name, split.y_val, pred, fit_time, predict_time)
        row["train_accuracy"] = accuracy_score(split.y_train, model.predict(split.X_train_s))
        row["dataset"] = split.prefix
        print_train_gap(row)
        return row, model, pred

    def run(self, split):
        rows = []
        best = None  # (accuracy, depth, model, val_pred, name)

        def consider(depth, min_leaf=5, criterion="gini"):
            nonlocal best
            row, model, pred = self.fit_one(split, depth, min_leaf, criterion)
            rows.append(row)
            if best is None or row["accuracy"] > best[0]:
                best = (row["accuracy"], depth, model, pred, row["model"])

        if self.quick:
            qt = QUICK_BEST[split.prefix]["tree"]
            consider(qt["depth"], qt["min_leaf"], qt["criterion"])
        else:
            # (a) depth comparison (Gini, min_samples_leaf=5)
            for depth in [3, 5, 10, 15, 20]:
                consider(depth)
            # (b) at the best depth: minimum leaf size and splitting criterion
            best_depth = best[1]
            for min_leaf, criterion in [(1, "gini"), (20, "gini"), (5, "entropy")]:
                consider(best_depth, min_leaf, criterion)

        _, _, model, val_pred, name = best
        return ExperimentResult(self.model_name, rows, val_pred, name,
                                lambda: model.predict(split.X_test_s))
