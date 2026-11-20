"""Experiment 1: k-NN.

Studies (rules: k, distance function, feature normalization, weighted voting):
  (a) k in {1, 3, 5, 10, 20} x voting scheme, with z-score + Euclidean
  (b) larger k in {30, 50, 100} with 1/d^2 voting
  (c) raw vs z-score features, Euclidean vs cosine distance
  (d) Hellinger features (image only: sqrt + L2-normalize of the sparse histograms)
"""

import time

from ..config import QUICK_BEST
from ..evaluation import score_model
from ..models import knn_predict
from .base import ExperimentResult

WEIGHT_OPTIONS = ["uniform", "distance", "distance_squared", "gaussian"]


class KNNExperiment:
    model_name = "kNN"

    def __init__(self, quick=False):
        self.quick = quick

    def configs(self, prefix):
        """List of (normalization, metric, k, weighting) to evaluate."""
        if self.quick:
            return [QUICK_BEST[prefix]["knn"]]
        # (a) k and voting weights, with z-score + Euclidean distance.
        #     With k=1 all schemes give the same answer, so only "uniform" is used.
        configs = [("zscore", "euclidean", 1, "uniform")] + [
            ("zscore", "euclidean", k, w)
            for k in [3, 5, 10, 20] for w in WEIGHT_OPTIONS
        ]
        # (b) larger k: sentiment accuracy was still rising at k=20.
        configs += [("zscore", "euclidean", k, "distance_squared") for k in [30, 50, 100]]
        # (c) feature normalization and distance function.
        configs += [
            (norm, metric, k, "distance")
            for norm, metric in [("none", "euclidean"), ("zscore", "cosine"), ("none", "cosine")]
            for k in [1, 5, 20]
        ]
        # (d) Hellinger features (sqrt + L2-normalize), only for non-negative data.
        #     Vectors are unit length, so cosine would rank neighbours identically.
        configs += [("hellinger", "euclidean", 1, "uniform")] + [
            ("hellinger", "euclidean", k, w)
            for k in [5, 10, 20] for w in ["distance", "distance_squared"]
        ]
        return configs

    def run(self, split):
        # Each entry: (train split, validation split, test, ALL training rows).
        data = {
            "zscore": (split.X_train_s, split.X_val_s, split.X_test_s, split.X_all_s),
            "none": (split.X_train, split.X_val, split.X_test, split.X),
        }
        if split.has_hellinger:
            data["hellinger"] = (split.X_train_h, split.X_val_h, split.X_test_h, split.X_all_h)
        rows = []
        best_acc, best_conf, best_pred = -1, None, None
        for norm, metric, k, weighted in self.configs(split.prefix):
            if norm not in data:  # e.g. Hellinger on the sentiment embeddings
                continue
            Xtr, Xva, _, _ = data[norm]
            start = time.time()
            pred = knn_predict(Xtr, split.y_train, Xva, k=k, weighted=weighted, metric=metric)
            elapsed = time.time() - start
            name = f"kNN k={k} weighted={weighted} norm={norm} metric={metric}"
            row = score_model(name, split.y_val, pred, 0.0, elapsed)
            row["dataset"] = split.prefix
            rows.append(row)
            if row["accuracy"] > best_acc:
                best_acc, best_conf, best_pred = row["accuracy"], (norm, metric, k, weighted), pred

        norm, metric, k, weighted = best_conf
        # The validation split was only needed to CHOOSE the settings.
        # The final test predictions use ALL training rows (train + validation).
        _, _, Xte, Xall = data[norm]
        test_pred = knn_predict(Xall, split.y, Xte, k=k, weighted=weighted, metric=metric)
        setting = f"k={k}, weighted={weighted}, norm={norm}, metric={metric}"
        print(f"Best kNN: {setting} (test predictions use all {len(split.y)} training rows).")
        return ExperimentResult(self.model_name, rows, best_pred, setting, lambda: test_pred)
