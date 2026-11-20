"""Loading the competition CSVs and building the train/validation split."""

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import SEED


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


def hellinger(X):
    """Hellinger transform for non-negative histogram features: sqrt, then L2-normalize.

    The image features are sparse (85% zeros), non-negative and unit-norm, i.e.
    histogram-like. sqrt() shrinks the few large bins so many small ones count too,
    and Euclidean distance on the result is the Hellinger distance between histograms.
    It is a fixed per-row formula: nothing is fitted, so nothing leaks between splits.
    """
    H = np.sqrt(np.maximum(X, 0.0))
    return (H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)).astype(np.float32)


@dataclass
class DatasetSplit:
    """Everything an experiment needs for one dataset.

    Validation methodology: a stratified 80/20 split of the labelled training data.
    The scaler is fitted on the 80% train part only, so no validation statistics leak.
    The `_s` arrays are z-score standardized; the others are the raw features.
    The `_h` arrays are Hellinger-transformed, or None if the features have negative
    values (sentiment embeddings), where sqrt is not defined.
    """
    prefix: str
    feature_cols: list
    test_ids: pd.Series
    X: np.ndarray          # all labelled rows (raw)
    y: np.ndarray
    X_test: np.ndarray     # Kaggle test rows (raw)
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    X_train_s: np.ndarray
    X_val_s: np.ndarray
    X_test_s: np.ndarray
    X_all_s: np.ndarray    # all labelled rows, scaled with the train-split scaler
    X_train_h: np.ndarray = None
    X_val_h: np.ndarray = None
    X_test_h: np.ndarray = None
    X_all_h: np.ndarray = None

    @property
    def n_classes(self):
        return len(np.unique(self.y))

    @property
    def has_hellinger(self):
        return self.X_train_h is not None


def prepare_split(root, prefix, test_size=0.20):
    _, test_df, X, y, X_test, feature_cols = load_dataset(root, prefix)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    h = {}
    if X.min() >= 0 and X_test.min() >= 0:
        h = dict(X_train_h=hellinger(X_train), X_val_h=hellinger(X_val),
                 X_test_h=hellinger(X_test), X_all_h=hellinger(X))
    return DatasetSplit(
        **h,
        prefix=prefix,
        feature_cols=feature_cols,
        test_ids=test_df["id"],
        X=X, y=y, X_test=X_test,
        X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
        X_train_s=X_train_s,
        X_val_s=scaler.transform(X_val).astype(np.float32),
        X_test_s=scaler.transform(X_test).astype(np.float32),
        X_all_s=scaler.transform(X).astype(np.float32),
    )
