"""Experiment 3: Random Forest.

Studies (rules: number of trees, max depth, features per split):
  (a) n_trees in {20, 40} (depth 14, sqrt features)
  (b) 20 trees with depth 8 / 20 and log2 features
Training accuracy is recorded so the overfitting can be compared with a single tree.
"""

import time

from sklearn.metrics import accuracy_score

from ..config import QUICK_BEST, SEED
from ..evaluation import print_train_gap, score_model
from ..models import ScratchRandomForest
from .base import ExperimentResult


class RandomForestExperiment:
    model_name = "Random Forest"

    def __init__(self, quick=False):
        self.quick = quick

    def fit_one(self, split, n_trees, depth=14, max_features="sqrt"):
        forest = ScratchRandomForest(
            n_estimators=n_trees, max_depth=depth,
            min_samples_split=20, min_samples_leaf=4,
            max_features=max_features, random_state=SEED
        )
        start = time.time()
        forest.fit(split.X_train_s, split.y_train)
        fit_time = time.time() - start
        start = time.time()
        pred = forest.predict(split.X_val_s)
        predict_time = time.time() - start
        name = f"Scratch Random Forest trees={n_trees} depth={depth} features={max_features}"
        row = score_model(name, split.y_val, pred, fit_time, predict_time)
        row["train_accuracy"] = accuracy_score(split.y_train, forest.predict(split.X_train_s))
        row["dataset"] = split.prefix
        print_train_gap(row)
        return row, forest, pred

    def run(self, split):
        rows = []
        best = None  # (accuracy, forest, val_pred, name)

        def consider(n_trees, depth=14, max_features="sqrt"):
            nonlocal best
            row, forest, pred = self.fit_one(split, n_trees, depth, max_features)
            rows.append(row)
            if best is None or row["accuracy"] > best[0]:
                best = (row["accuracy"], forest, pred, row["model"])

        if self.quick:
            qf = QUICK_BEST[split.prefix]["forest"]
            consider(qf["n_trees"], qf["depth"], qf["max_features"])
        else:
            # (a) number of trees
            for n_trees in [20, 40]:
                consider(n_trees)
            # (b) max depth and number of features per split (20 trees each,
            #     compare with "trees=20 depth=14 features=sqrt" above)
            for depth, max_features in [(8, "sqrt"), (20, "sqrt"), (14, "log2")]:
                consider(20, depth, max_features)

        _, forest, val_pred, name = best
        return ExperimentResult(self.model_name, rows, val_pred, name,
                                lambda: forest.predict(split.X_test_s))
