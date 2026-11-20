"""Global settings: random seed and the best known settings for --quick runs."""

import random

import numpy as np

SEED = 42
DATASETS = ("image", "sentiment")

# Best settings found by the full run (see validation_results.csv).
# --quick trains ONLY these, one model per algorithm, to make a fast submission.
# After each full run, copy the new best values here.
QUICK_BEST = {
    "image": {
        # (normalization, distance, k, voting weights)
        # normalization: "zscore", "none" or "hellinger"
        "knn": ("hellinger", "euclidean", 1, "uniform"),  # val 0.572 (z-score: 0.434)
        "tree": {"depth": 20, "min_leaf": 5, "criterion": "gini"},
        "forest": {"n_trees": 40, "depth": 14, "max_features": "sqrt"},
        # "input": "zscore" or "hellinger" (sqrt + L2-normalized features)
        "nn": dict(learning_rate=1e-3, hidden=(512, 256), dropouts=(0.20, 0.10),
                   weight_decay=1e-4, input="zscore"),  # val 0.377
        "clusters": [50],
    },
    "sentiment": {
        "knn": ("zscore", "euclidean", 50, "distance_squared"),  # val 0.586
        "tree": {"depth": 5, "min_leaf": 5, "criterion": "gini"},
        "forest": {"n_trees": 40, "depth": 14, "max_features": "sqrt"},
        "nn": dict(learning_rate=3e-4, hidden=(256, 64), dropouts=(0.5, 0.3),
                   weight_decay=1e-2),  # val 0.681 (old lr=1e-3 setting: 0.665)
        "clusters": [3],
    },
}


def set_seeds(seed=SEED):
    """Seed Python and NumPy. PyTorch is seeded inside the neural-net training."""
    random.seed(seed)
    np.random.seed(seed)
