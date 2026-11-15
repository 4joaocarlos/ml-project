#!/usr/bin/env python3
"""
COMP 472 ML experiments for image and sentiment feature-vector datasets.

Expected ZIP contents:
  image_train.csv, image_test.csv
  sentiment_train.csv, sentiment_test.csv
  sample_submission.csv

Required algorithms:
  - k-NN (implemented with NumPy)
  - CART decision tree (implemented with NumPy)
  - random forest (ensemble of the custom CART trees)
  - feed-forward neural network (PyTorch)
  - k-means clustering (implemented with NumPy)

Version 2 additions:
  - k-NN: cosine distance, no-normalization vs z-score, larger k values
  - Decision Tree: training accuracy, entropy criterion, min_samples_leaf
  - Random Forest: training accuracy, max depth, features per split
  - Neural Network: learning-rate / architecture / dropout / weight-decay sweep,
    training accuracy, per-epoch history (nn_history.csv)
  - k-means: several k values, NMI and purity, cluster-vs-class table
  - Error analysis: per-class recall of all models, overlap of their mistakes

Run:
  python comp472_experiments.py --zip data.zip

Optional:
  python comp472_experiments.py --zip data.zip --quick   # only the QUICK_BEST settings
  python comp472_experiments.py --zip data.zip --skip-neural-net
"""

import argparse
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    adjusted_rand_score, normalized_mutual_info_score
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Best settings found by the full run (see validation_results.csv).
# --quick trains ONLY these, one model per algorithm, to make a fast submission.
# After each full run, copy the new best values here.
QUICK_BEST = {
    "image": {
        # (normalization, distance, k, voting weights)
        "knn": ("zscore", "euclidean", 1, "uniform"),
        "tree": {"depth": 20, "min_leaf": 5, "criterion": "gini"},
        "forest": {"n_trees": 40, "depth": 14, "max_features": "sqrt"},
        "nn": dict(learning_rate=1e-3, hidden=(256, 64), dropouts=(0.20, 0.10),
                   weight_decay=1e-4),
        "clusters": [50],
    },
    "sentiment": {
        "knn": ("zscore", "euclidean", 20, "distance_squared"),
        "tree": {"depth": 5, "min_leaf": 5, "criterion": "gini"},
        "forest": {"n_trees": 40, "depth": 14, "max_features": "sqrt"},
        "nn": dict(learning_rate=1e-3, hidden=(256, 64), dropouts=(0.20, 0.10),
                   weight_decay=1e-4),
        "clusters": [3],
    },
}


def extract_zip(zip_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Avoid extracting files outside the chosen directory.
            target = (output_dir / member).resolve()
            if output_dir.resolve() not in target.parents and target != output_dir.resolve():
                continue
            if member.lower().endswith(".csv"):
                zf.extract(member, output_dir)
    return output_dir


def find_csv(root, filename):
    matches = list(Path(root).rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename}. Check that the ZIP contains this exact CSV filename."
        )
    return matches[0]


def load_dataset(root, prefix):
    train_path = find_csv(root, f"{prefix}_train.csv")
    test_path = find_csv(root, f"{prefix}_test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required = {"id", "label"}
    if not required.issubset(train_df.columns):
        raise ValueError(f"{train_path.name} must contain id and label columns.")
    if "id" not in test_df.columns:
        raise ValueError(f"{test_path.name} must contain an id column.")

    feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
    if not feature_cols:
        feature_cols = [c for c in train_df.columns if c not in ("id", "label")]
    missing = sorted(set(feature_cols) - set(test_df.columns))
    if missing:
        raise ValueError(f"Test file is missing feature columns, e.g. {missing[:5]}")

    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    y = train_df["label"].to_numpy(dtype=np.int64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all() or not np.isfinite(X_test).all():
        raise ValueError(f"{prefix}: feature data contains NaN or infinite values.")

    return train_df, test_df, X, y, X_test, feature_cols


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


# ------------------------- k-NN from scratch -------------------------

def knn_predict(X_train, y_train, X_query, k=5, weighted=False, batch_size=128,
                metric="euclidean"):
    """Euclidean or cosine k-NN.

    weighted chooses how much each of the k neighbours counts in the vote:
      False or "uniform"   -> every neighbour has 1 vote
      True  or "distance"  -> vote = 1 / d
      "distance_squared"   -> vote = 1 / d^2  (the closest neighbours count much more)
      "gaussian"           -> vote = exp(-d^2 / (2 * sigma^2)),
                              sigma = mean distance of this sample's k neighbours

    metric:
      "euclidean" -> straight-line distance
      "cosine"    -> compares only the DIRECTION of the vectors, not their length
    """
    valid = (False, True, "uniform", "distance", "distance_squared", "gaussian")
    if weighted not in valid:
        raise ValueError(f"Unknown weighting scheme: {weighted}")
    if metric == "cosine":
        # Scale every vector to length 1. For unit vectors ||a-b||^2 = 2 - 2*cos(a, b),
        # so the Euclidean code below then ranks neighbours exactly by cosine similarity.
        X_train = X_train / np.maximum(np.linalg.norm(X_train, axis=1, keepdims=True), 1e-12)
        X_query = X_query / np.maximum(np.linalg.norm(X_query, axis=1, keepdims=True), 1e-12)
    elif metric != "euclidean":
        raise ValueError(f"Unknown metric: {metric}")

    predictions = []
    classes = np.unique(y_train)
    for start in range(0, len(X_query), batch_size):
        Q = X_query[start:start + batch_size]
        # Squared Euclidean distances, avoiding a large 3-D temporary.
        d2 = (
            np.sum(Q * Q, axis=1, keepdims=True)
            + np.sum(X_train * X_train, axis=1)[None, :]
            - 2.0 * Q @ X_train.T
        )
        np.maximum(d2, 0, out=d2)
        nearest = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        for row, indices in enumerate(nearest):
            distances = np.sqrt(d2[row, indices])
            labels = y_train[indices]
            if weighted is False or weighted == "uniform":
                weights = np.ones(len(indices), dtype=np.float64)
            elif weighted is True or weighted == "distance":
                weights = 1.0 / np.maximum(distances, 1e-8)
            elif weighted == "distance_squared":
                weights = 1.0 / np.maximum(distances ** 2, 1e-8)
            else:  # "gaussian"
                sigma = max(float(distances.mean()), 1e-8)
                weights = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
            totals = np.zeros(len(classes), dtype=np.float64)
            for label, weight in zip(labels, weights):
                totals[np.searchsorted(classes, label)] += weight
            predictions.append(classes[np.argmax(totals)])
    return np.asarray(predictions, dtype=np.int64)


# ---------------------- Decision tree from scratch ------------------

class TreeNode:
    __slots__ = ("prediction", "feature", "threshold", "left", "right")
    def __init__(self, prediction):
        self.prediction = prediction
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


class ScratchDecisionTree:
    """A compact CART-style classifier using Gini impurity or entropy."""
    def __init__(self, max_depth=12, min_samples_split=20, min_samples_leaf=5,
                 max_features=None, max_thresholds=24, random_state=42,
                 criterion="gini"):
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"Unknown criterion: {criterion}")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.criterion = criterion
        self.rng = np.random.default_rng(random_state)
        self.root = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs * probs)

    def _impurity(self, y):
        """Gini = 1 - sum(p^2)   |   Entropy = -sum(p * log2(p))"""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        if self.criterion == "entropy":
            return float(-np.sum(probs * np.log2(probs)))
        return 1.0 - np.sum(probs * probs)

    def _best_split(self, X, y):
        n, d = X.shape
        feature_count = d if self.max_features is None else min(self.max_features, d)
        features = self.rng.choice(d, size=feature_count, replace=False)
        parent_impurity = self._impurity(y)
        best_gain, best_feature, best_threshold = 0.0, None, None

        for feature in features:
            values = X[:, feature]
            unique = np.unique(values)
            if len(unique) < 2:
                continue
            if len(unique) > self.max_thresholds + 1:
                quantiles = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
                thresholds = np.unique(np.quantile(values, quantiles))
            else:
                thresholds = (unique[:-1] + unique[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values <= threshold
                n_left = int(left_mask.sum())
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                gain = parent_impurity
                gain -= (n_left / n) * self._impurity(y[left_mask])
                gain -= (n_right / n) * self._impurity(y[~left_mask])
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, int(feature), float(threshold)

        return best_feature, best_threshold

    def _grow(self, X, y, depth):
        values, counts = np.unique(y, return_counts=True)
        node = TreeNode(values[np.argmax(counts)])
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(values) == 1):
            return node

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return node

        mask = X[:, feature] <= threshold
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return node

        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._grow(np.asarray(X), np.asarray(y), 0)
        return self

    def predict(self, X):
        output = []
        for row in np.asarray(X):
            node = self.root
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            output.append(node.prediction)
        return np.asarray(output, dtype=np.int64)


class ScratchRandomForest:
    def __init__(self, n_estimators=30, max_depth=14, min_samples_split=20,
                 min_samples_leaf=4, max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _n_features(self, d):
        """Number of random features each split may look at."""
        mf = self.max_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return max(1, min(int(mf), d))

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        self.trees = []
        for i in range(self.n_estimators):
            indices = rng.integers(0, n, size=n)
            tree = ScratchDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self._n_features(d),
                random_state=self.random_state + i + 1
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
            print(f"    Random Forest tree {i + 1}/{self.n_estimators}", end="\r")
        print()
        return self

    def predict(self, X):
        votes = np.stack([tree.predict(X) for tree in self.trees], axis=0)
        predictions = []
        for column in votes.T:
            labels, counts = np.unique(column, return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions, dtype=np.int64)


# ----------------------- Neural network (PyTorch) -------------------

def run_neural_net(X_train, y_train, X_val, y_val, X_test, epochs=25,
                   batch_size=256, learning_rate=1e-3, hidden=(256, 64),
                   dropouts=(0.20, 0.10), weight_decay=1e-4, verbose=True):
    """Train one MLP. Returns (val_pred, test_pred, (fit_s, predict_s), info).
    info = best epoch, training accuracy and per-epoch history."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch not installed; skipping neural network. Install with: pip install torch")
        return None, None, None, None

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = int(max(y_train.max(), y_val.max())) + 1

    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            layers, prev = [], input_dim
            for size, p in zip(hidden, dropouts):
                layers += [nn.Linear(prev, size), nn.ReLU()]
                if p > 0:
                    layers.append(nn.Dropout(p))
                prev = size
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    model = MLP(X_train.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    start = time.time()
    best_state, best_acc, best_epoch = None, -1, 0
    history = []
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    for epoch in range(epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_val, pred)
        history.append((epoch + 1, total_loss / seen, acc))
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    Neural net epoch {epoch + 1}/{epochs}: validation accuracy={acc:.4f}")
    fit_seconds = time.time() - start

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    start = time.time()
    with torch.no_grad():
        val_pred = model(Xv).argmax(dim=1).cpu().numpy()
        Xt = torch.tensor(X_test, dtype=torch.float32, device=device)
        test_pred = model(Xt).argmax(dim=1).cpu().numpy()
    predict_seconds = time.time() - start
    with torch.no_grad():
        Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
        train_acc = accuracy_score(y_train, model(Xtr).argmax(dim=1).cpu().numpy())
    info = {"best_epoch": best_epoch, "train_accuracy": train_acc, "history": history}
    return (val_pred, test_pred, (fit_seconds, predict_seconds), info)


# ------------------------- k-means from scratch ----------------------

def scratch_kmeans(X, k, max_iter=100, random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    X_sq = np.sum(X * X, axis=1)

    def sq_dist(C):
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, for all points and centres at once
        d2 = X_sq[:, None] - 2.0 * (X @ C.T) + np.sum(C * C, axis=1)[None, :]
        return np.maximum(d2, 0.0)

    # k-means++ initialization: each new centre is picked with probability
    # proportional to its squared distance from the closest centre chosen so far.
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(len(X))]
    closest = sq_dist(centers[:1])[:, 0]
    for i in range(1, k):
        total = closest.sum()
        if total <= 0:
            centers[i] = X[rng.integers(len(X))]
        else:
            centers[i] = X[rng.choice(len(X), p=closest / total)]
        closest = np.minimum(closest, sq_dist(centers[i:i + 1])[:, 0])

    for _ in range(max_iter):
        labels = sq_dist(centers).argmin(axis=1)
        new_centers = centers.copy()
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
            else:
                new_centers[cluster] = X[rng.integers(len(X))]
        if np.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers
    labels = sq_dist(centers).argmin(axis=1)  # labels for the final centres
    return labels, centers


def cluster_purity(y_true, y_cluster):
    """Fraction of samples that belong to the majority class of their cluster."""
    total = 0
    for c in np.unique(y_cluster):
        _, counts = np.unique(y_true[y_cluster == c], return_counts=True)
        total += counts.max()
    return total / len(y_true)


def save_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compare_model_errors(prefix, y_val, candidate_preds, output_dir):
    """Per-class recall of every model, and how much their mistakes overlap."""
    classes = np.unique(y_val)
    recall = {}
    for name, pred in candidate_preds.items():
        recall[name] = [np.mean(pred[y_val == c] == c) for c in classes]
    recall_df = pd.DataFrame(recall, index=classes)
    recall_df["mean"] = recall_df.mean(axis=1)
    recall_df.sort_values("mean").to_csv(
        output_dir / f"{prefix}_per_class_recall.csv", index_label="class"
    )
    print(f"\nHardest {prefix} classes (lowest recall averaged over all models):")
    print(recall_df.sort_values("mean").head(5).round(3).to_string())

    wrong = {name: pred != y_val for name, pred in candidate_preds.items()}
    names = list(wrong)
    all_wrong = np.logical_and.reduce([wrong[n] for n in names])
    print(f"Validation samples misclassified by ALL models: {all_wrong.mean():.1%}")
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = int(np.sum(wrong[a] & wrong[b]))
            either = int(np.sum(wrong[a] | wrong[b]))
            rows.append({
                "model_a": a, "model_b": b,
                "jaccard_of_errors": both / max(either, 1),
                "pct_of_a_errors_fixed_by_b": np.sum(wrong[a] & ~wrong[b]) / max(wrong[a].sum(), 1),
                "pct_of_b_errors_fixed_by_a": np.sum(wrong[b] & ~wrong[a]) / max(wrong[b].sum(), 1),
            })
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_error_overlap.csv", index=False)


def run_experiments(root, output_dir, quick=False, skip_neural_net=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    test_predictions = {}
    nn_history_rows = []
    clustering_rows = []

    for prefix in ("image", "sentiment"):
        print("\n" + "=" * 72)
        print(f"DATASET: {prefix.upper()}")
        print("=" * 72)
        train_df, test_df, X, y, X_test, feature_cols = load_dataset(root, prefix)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=SEED, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        print(f"Rows: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")
        print(f"Features: {len(feature_cols)} | Classes: {len(np.unique(y))}")

        # ------------------------------ k-NN ------------------------------
        # Two versions of the data: z-score (standardized) and raw (no scaling).
        # Each entry: (train split, validation split, test, ALL training rows).
        knn_data = {
            "zscore": (X_train_s, X_val_s, X_test_s, scaler.transform(X).astype(np.float32)),
            "none": (X_train, X_val, X_test, X),
        }
        weight_options = ["uniform", "distance", "distance_squared", "gaussian"]
        if not quick:
            # (a) k and voting weights, with z-score + Euclidean distance.
            #     With k=1 all schemes give the same answer, so only "uniform" is used.
            knn_configs = [("zscore", "euclidean", 1, "uniform")] + [
                ("zscore", "euclidean", k_value, w)
                for k_value in [3, 5, 10, 20] for w in weight_options
            ]
            # (b) larger k: sentiment accuracy was still rising at k=20.
            knn_configs += [("zscore", "euclidean", k_value, "distance_squared")
                            for k_value in [30, 50, 100]]
            # (c) feature normalization and distance function.
            knn_configs += [
                (norm, metric, k_value, "distance")
                for norm, metric in [("none", "euclidean"), ("zscore", "cosine"), ("none", "cosine")]
                for k_value in [1, 5, 20]
            ]
        else:
            knn_configs = [QUICK_BEST[prefix]["knn"]]

        best_knn = (-1, None, None)  # (accuracy, settings, validation predictions)
        for norm, metric, k, weighted in knn_configs:
            Xtr_k, Xva_k, _, _ = knn_data[norm]
            start = time.time()
            pred = knn_predict(Xtr_k, y_train, Xva_k, k=k, weighted=weighted, metric=metric)
            elapsed = time.time() - start
            name = f"kNN k={k} weighted={weighted} norm={norm} metric={metric}"
            row = score_model(name, y_val, pred, 0.0, elapsed)
            row["dataset"] = prefix
            all_results.append(row)
            if row["accuracy"] > best_knn[0]:
                best_knn = (row["accuracy"], (norm, metric, k, weighted), pred)
        norm, metric, k, weighted = best_knn[1]
        # The validation split was only needed to CHOOSE the settings.
        # The final test predictions use ALL training rows (train + validation).
        _, _, Xte_k, Xall_k = knn_data[norm]
        knn_test_pred = knn_predict(Xall_k, y, Xte_k, k=k, weighted=weighted, metric=metric)
        print(f"Best kNN: k={k}, weighted={weighted}, norm={norm}, metric={metric} "
              f"(test predictions use all {len(y)} training rows).")
        test_predictions[f"{prefix}_knn"] = knn_test_pred

        # -------------------------- Decision tree --------------------------
        def fit_tree(depth, min_leaf=5, criterion="gini"):
            model = ScratchDecisionTree(
                max_depth=depth, min_samples_split=20, min_samples_leaf=min_leaf,
                max_features=None, random_state=SEED, criterion=criterion
            )
            start = time.time()
            model.fit(X_train_s, y_train)
            fit_time = time.time() - start
            start = time.time()
            pred = model.predict(X_val_s)
            predict_time = time.time() - start
            name = f"Scratch Decision Tree depth={depth} leaf={min_leaf} {criterion}"
            row = score_model(name, y_val, pred, fit_time, predict_time)
            row["train_accuracy"] = accuracy_score(y_train, model.predict(X_train_s))
            row["dataset"] = prefix
            all_results.append(row)
            print(f"{'':28s} train accuracy={row['train_accuracy']:.4f} "
                  f"(train - validation gap = {row['train_accuracy'] - row['accuracy']:.4f})")
            return row, model

        best_tree = (-1, None, None)  # (accuracy, depth, model)
        if quick:
            qt = QUICK_BEST[prefix]["tree"]
            row, model = fit_tree(qt["depth"], qt["min_leaf"], qt["criterion"])
            best_tree = (row["accuracy"], qt["depth"], model)
        else:
            # (a) depth comparison (Gini, min_samples_leaf=5)
            for depth in [3, 5, 10, 15, 20]:
                row, model = fit_tree(depth)
                if row["accuracy"] > best_tree[0]:
                    best_tree = (row["accuracy"], depth, model)
            # (b) at the best depth: minimum leaf size and splitting criterion
            best_depth = best_tree[1]
            for min_leaf, criterion in [(1, "gini"), (20, "gini"), (5, "entropy")]:
                row, model = fit_tree(best_depth, min_leaf, criterion)
                if row["accuracy"] > best_tree[0]:
                    best_tree = (row["accuracy"], best_depth, model)

        # -------------------------- Random forest --------------------------
        def fit_forest(n_trees, depth=14, max_features="sqrt"):
            forest = ScratchRandomForest(
                n_estimators=n_trees, max_depth=depth,
                min_samples_split=20, min_samples_leaf=4,
                max_features=max_features, random_state=SEED
            )
            start = time.time()
            forest.fit(X_train_s, y_train)
            fit_time = time.time() - start
            start = time.time()
            pred = forest.predict(X_val_s)
            predict_time = time.time() - start
            name = f"Scratch Random Forest trees={n_trees} depth={depth} features={max_features}"
            row = score_model(name, y_val, pred, fit_time, predict_time)
            row["train_accuracy"] = accuracy_score(y_train, forest.predict(X_train_s))
            row["dataset"] = prefix
            all_results.append(row)
            print(f"{'':28s} train accuracy={row['train_accuracy']:.4f} "
                  f"(train - validation gap = {row['train_accuracy'] - row['accuracy']:.4f})")
            return row, forest

        best_forest = (-1, None)
        if quick:
            qf = QUICK_BEST[prefix]["forest"]
            row, forest = fit_forest(qf["n_trees"], qf["depth"], qf["max_features"])
            best_forest = (row["accuracy"], forest)
        else:
            # (a) number of trees
            for n_trees in [20, 40]:
                row, forest = fit_forest(n_trees)
                if row["accuracy"] > best_forest[0]:
                    best_forest = (row["accuracy"], forest)
            # (b) max depth and number of features per split (20 trees each,
            #     compare with "trees=20 depth=14 features=sqrt" above)
            for depth, max_features in [(8, "sqrt"), (20, "sqrt"), (14, "log2")]:
                row, forest = fit_forest(20, depth, max_features)
                if row["accuracy"] > best_forest[0]:
                    best_forest = (row["accuracy"], forest)

        # ------------------------- Neural network -------------------------
        if not skip_neural_net:
            epochs = 40
            base = dict(learning_rate=1e-3, hidden=(256, 64), dropouts=(0.20, 0.10),
                        weight_decay=1e-4)
            if quick:
                nn_configs = [("best settings", QUICK_BEST[prefix]["nn"])]
            else:
                # One change at a time, compared with the base network.
                nn_configs = [
                    ("256-64 lr=1e-3", base),
                    ("lr=1e-2", dict(base, learning_rate=1e-2)),
                    ("lr=1e-4", dict(base, learning_rate=1e-4)),
                    ("512-256", dict(base, hidden=(512, 256))),
                    ("1024-512", dict(base, hidden=(1024, 512))),
                    ("dropout 0.5-0.3", dict(base, dropouts=(0.5, 0.3))),
                    ("weight decay 1e-3", dict(base, weight_decay=1e-3)),
                ]
            best_nn = None  # (accuracy, label, validation predictions, test predictions)
            for label, conf in nn_configs:
                nn_result = run_neural_net(
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                    epochs=epochs, verbose=False, **conf
                )
                if nn_result[0] is None:
                    break
                val_pred, test_pred, timings, info = nn_result
                row = score_model(f"PyTorch MLP {label}", y_val, val_pred, *timings)
                row["train_accuracy"] = info["train_accuracy"]
                row["best_epoch"] = info["best_epoch"]
                row["dataset"] = prefix
                all_results.append(row)
                print(f"{'':28s} train accuracy={info['train_accuracy']:.4f} "
                      f"best epoch={info['best_epoch']}/{epochs}")
                for epoch, train_loss, val_acc in info["history"]:
                    nn_history_rows.append({
                        "dataset": prefix, "config": label, "epoch": epoch,
                        "train_loss": train_loss, "val_accuracy": val_acc
                    })
                if best_nn is None or row["accuracy"] > best_nn[0]:
                    best_nn = (row["accuracy"], label, val_pred, test_pred)
            if best_nn is not None:
                nn_val_pred = best_nn[2]
                test_predictions[f"{prefix}_mlp"] = best_nn[3]
                print(f"Best neural network: {best_nn[1]}")

        # Save best validation model predictions and error analysis for the selected
        # best-performing model among the tested required methods.
        candidate_preds = {
            "kNN": best_knn[2],
            "Decision Tree": best_tree[2].predict(X_val_s),
            "Random Forest": best_forest[1].predict(X_val_s),
        }
        if not skip_neural_net and f"{prefix}_mlp" in test_predictions:
            candidate_preds["Neural Network"] = nn_val_pred
        best_name = max(
            candidate_preds,
            key=lambda model_name: accuracy_score(y_val, candidate_preds[model_name])
        )
        best_val_pred = candidate_preds[best_name]
        print(f"\nBest validation model for {prefix}: {best_name}")
        print(classification_report(y_val, best_val_pred, zero_division=0))
        save_confusion(
            y_val, best_val_pred, f"{prefix}: {best_name}",
            output_dir / f"{prefix}_confusion_matrix.png"
        )

        # Error analysis: most commonly confused true/predicted class pairs.
        cm = confusion_matrix(y_val, best_val_pred)
        np.fill_diagonal(cm, 0)
        pairs = []
        for true_class, pred_class in zip(*np.where(cm > 0)):
            pairs.append((int(cm[true_class, pred_class]), int(true_class), int(pred_class)))
        pairs.sort(reverse=True)
        pd.DataFrame(pairs[:20], columns=["count", "true_class", "predicted_class"]).to_csv(
            output_dir / f"{prefix}_most_confused_pairs.csv", index=False
        )
        # Error analysis across ALL models: hardest classes and shared mistakes.
        compare_model_errors(prefix, y_val, candidate_preds, output_dir)

        # Clustering experiment on a capped, reproducible sample for practical runtime.
        rng = np.random.default_rng(SEED)
        cluster_n = min(5000, len(X_train_s))
        chosen = rng.choice(len(X_train_s), size=cluster_n, replace=False)
        cluster_X, cluster_y = X_train_s[chosen], y_train[chosen]
        n_true = len(np.unique(y))
        if quick:
            cluster_ks = QUICK_BEST[prefix]["clusters"]
        elif prefix == "image":
            cluster_ks = [10, 25, 50, 100]
        else:
            cluster_ks = [2, 3, 5, 10]
        dataset_cluster_rows = []
        for n_clusters in cluster_ks:
            cluster_labels, centers = scratch_kmeans(
                cluster_X, k=n_clusters, max_iter=60, random_state=SEED
            )
            inertia = float(np.sum((cluster_X - centers[cluster_labels]) ** 2))
            r = {
                "dataset": prefix, "samples": cluster_n, "clusters": n_clusters,
                "adjusted_rand_index": adjusted_rand_score(cluster_y, cluster_labels),
                "nmi": normalized_mutual_info_score(cluster_y, cluster_labels),
                "purity": cluster_purity(cluster_y, cluster_labels),
                "inertia": inertia,
            }
            dataset_cluster_rows.append(r)
            print(f"k-means k={n_clusters:3d} ({cluster_n} samples): ARI={r['adjusted_rand_index']:.4f} "
                  f"NMI={r['nmi']:.4f} purity={r['purity']:.4f}")
            if n_clusters == n_true:
                # Which true classes end up in which cluster?
                pd.crosstab(
                    pd.Series(cluster_labels, name="cluster"),
                    pd.Series(cluster_y, name="true_class")
                ).to_csv(output_dir / f"{prefix}_cluster_vs_class_k{n_true}.csv")
        clustering_rows += dataset_cluster_rows
        pd.DataFrame(dataset_cluster_rows).to_csv(
            output_dir / f"{prefix}_clustering.csv", index=False
        )

        # Choose the best validation model's corresponding test predictions.
        # MLP is included if it was the best candidate.
        if best_name == "kNN":
            final_test = test_predictions[f"{prefix}_knn"]
        elif best_name == "Neural Network":
            final_test = test_predictions[f"{prefix}_mlp"]
        elif best_name == "Decision Tree":
            final_test = best_tree[2].predict(X_test_s)
        else:
            final_test = best_forest[1].predict(X_test_s)
        test_predictions[f"{prefix}_best"] = final_test
        print(f"Selected {best_name} for the generated {prefix} test predictions.")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "validation_results.csv", index=False)
    print("\nValidation results saved to:", output_dir / "validation_results.csv")
    if nn_history_rows:
        pd.DataFrame(nn_history_rows).to_csv(output_dir / "nn_history.csv", index=False)
    if clustering_rows:
        pd.DataFrame(clustering_rows).to_csv(output_dir / "clustering_results.csv", index=False)

    # Build Kaggle submission in the exact sample_submission.csv ID order.
    image_train, image_test, *_ = load_dataset(root, "image")
    sentiment_train, sentiment_test, *_ = load_dataset(root, "sentiment")
    image_pred_df = pd.DataFrame({
        "id": image_test["id"], "label": test_predictions["image_best"]
    })
    sentiment_pred_df = pd.DataFrame({
        "id": sentiment_test["id"], "label": test_predictions["sentiment_best"]
    })
    predictions = pd.concat([image_pred_df, sentiment_pred_df], ignore_index=True)

    sample_path = find_csv(root, "sample_submission.csv")
    sample = pd.read_csv(sample_path)
    if list(sample.columns)[:1] != ["id"]:
        raise ValueError("sample_submission.csv must contain an id column.")
    submission = sample[["id"]].merge(
        predictions, on="id", how="left", validate="one_to_one"
    )
    if len(submission) != 10000:
        raise ValueError(f"Expected 10,000 submission rows; found {len(submission)}.")
    if submission["label"].isna().any():
        missing = submission.loc[submission["label"].isna(), "id"].head().tolist()
        raise ValueError(f"Missing predictions for IDs such as: {missing}")
    submission["label"] = submission["label"].astype(int)
    image_mask = submission["id"].str.startswith("img_")
    sentiment_mask = submission["id"].str.startswith("text_")
    if not submission.loc[image_mask, "label"].between(0, 49).all():
        raise ValueError("Image predictions must be integers from 0 to 49.")
    if not submission.loc[sentiment_mask, "label"].between(0, 2).all():
        raise ValueError("Sentiment predictions must be integers from 0 to 2.")
    submission.to_csv(output_dir / "submission.csv", index=False)
    print("Kaggle submission saved to:", output_dir / "submission.csv")
    print("Other outputs: confusion matrices, class-confusion CSVs, clustering CSVs, "
          "per-class recall, error overlap, nn_history.csv.")


def main():
    parser = argparse.ArgumentParser(description="Run COMP 472 ML experiments.")
    parser.add_argument("--zip", required=True, help="Path to the competition ZIP file.")
    parser.add_argument("--workdir", default="comp472_data", help="ZIP extraction folder.")
    parser.add_argument("--output", default="comp472_results", help="Results output folder.")
    parser.add_argument("--quick", action="store_true",
                        help="Train only the best known settings (QUICK_BEST) for a fast submission.")
    parser.add_argument("--skip-neural-net", action="store_true", help="Skip PyTorch neural network.")
    args = parser.parse_args()

    if not os.path.isfile(args.zip):
        raise FileNotFoundError(f"ZIP file not found: {args.zip}")
    root = extract_zip(args.zip, args.workdir)
    run_experiments(root, args.output, quick=args.quick,
                    skip_neural_net=args.skip_neural_net)


if __name__ == "__main__":
    main()