"""CART decision tree, implemented from scratch with NumPy."""

import numpy as np


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
            # Candidate thresholds: quantiles when there are many distinct values,
            # otherwise the midpoints between consecutive values.
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
