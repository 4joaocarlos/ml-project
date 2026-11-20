"""k-Nearest Neighbours, implemented from scratch with NumPy."""

import numpy as np

WEIGHTING_SCHEMES = (False, True, "uniform", "distance", "distance_squared", "gaussian")


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
    if weighted not in WEIGHTING_SCHEMES:
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
