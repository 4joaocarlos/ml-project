"""One class per experiment required by the project rules."""

from .base import ExperimentResult
from .clustering import ClusteringExperiment
from .decision_tree import DecisionTreeExperiment
from .error_analysis import ErrorAnalysis
from .knn import KNNExperiment
from .neural_net import NeuralNetExperiment
from .random_forest import RandomForestExperiment

__all__ = [
    "ExperimentResult", "KNNExperiment", "DecisionTreeExperiment",
    "RandomForestExperiment", "NeuralNetExperiment", "ClusteringExperiment",
    "ErrorAnalysis",
]
