"""Random forest: bootstrap-sampled ensemble of the scratch CART trees."""

import numpy as np

from .decision_tree import ScratchDecisionTree


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
            indices = rng.integers(0, n, size=n)  # bootstrap sample
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
        """Majority vote over all trees."""
        votes = np.stack([tree.predict(X) for tree in self.trees], axis=0)
        predictions = []
        for column in votes.T:
            labels, counts = np.unique(column, return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions, dtype=np.int64)
