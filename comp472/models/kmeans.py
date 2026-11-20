"""k-means with k-means++ initialization, implemented from scratch with NumPy."""

import numpy as np


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
