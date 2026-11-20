"""Unsupervised experiment: k-means on the (scaled) training features.

Rules questions this answers:
  - How closely do the clusters match the true labels? (ARI, NMI, purity)
  - How does changing the number of clusters affect the results?
  - Which classes are grouped together? (cluster-vs-class table at k = #classes)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..config import QUICK_BEST, SEED
from ..evaluation import cluster_purity
from ..models import scratch_kmeans

CLUSTER_KS = {"image": [10, 25, 50, 100], "sentiment": [2, 3, 5, 10]}


class ClusteringExperiment:
    def __init__(self, output_dir, quick=False, max_samples=5000, max_iter=60):
        self.output_dir = output_dir
        self.quick = quick
        self.max_samples = max_samples
        self.max_iter = max_iter

    def cluster_counts(self, prefix):
        return QUICK_BEST[prefix]["clusters"] if self.quick else CLUSTER_KS[prefix]

    def run(self, split):
        # Capped, reproducible sample for practical runtime.
        rng = np.random.default_rng(SEED)
        n = min(self.max_samples, len(split.X_train_s))
        chosen = rng.choice(len(split.X_train_s), size=n, replace=False)
        X, y = split.X_train_s[chosen], split.y_train[chosen]
        n_true = split.n_classes

        rows = []
        for k in self.cluster_counts(split.prefix):
            labels, centers = scratch_kmeans(X, k=k, max_iter=self.max_iter, random_state=SEED)
            r = {
                "dataset": split.prefix, "samples": n, "clusters": k,
                "adjusted_rand_index": adjusted_rand_score(y, labels),
                "nmi": normalized_mutual_info_score(y, labels),
                "purity": cluster_purity(y, labels),
                "inertia": float(np.sum((X - centers[labels]) ** 2)),
            }
            rows.append(r)
            print(f"k-means k={k:3d} ({n} samples): ARI={r['adjusted_rand_index']:.4f} "
                  f"NMI={r['nmi']:.4f} purity={r['purity']:.4f}")
            if k == n_true:
                # Which true classes end up in which cluster?
                pd.crosstab(
                    pd.Series(labels, name="cluster"),
                    pd.Series(y, name="true_class")
                ).to_csv(self.output_dir / f"{split.prefix}_cluster_vs_class_k{n_true}.csv")

        pd.DataFrame(rows).to_csv(self.output_dir / f"{split.prefix}_clustering.csv", index=False)
        return rows
