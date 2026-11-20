"""Common result type for the classifier experiments."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ExperimentResult:
    """What a classifier experiment hands back to the runner.

    model_name   -> "kNN", "Decision Tree", ... (used in reports and model selection)
    rows         -> one validation_results.csv row per configuration tried
    val_pred     -> validation predictions of the best configuration
    best_setting -> human-readable description of the best configuration
    predict_test -> returns Kaggle test predictions for the best configuration
                    (a function, so slow models only predict if they are selected)
    extra        -> anything experiment-specific (e.g. the NN per-epoch history)
    """
    model_name: str
    rows: list
    val_pred: np.ndarray
    best_setting: str
    predict_test: Callable[[], np.ndarray]
    extra: dict = field(default_factory=dict)
