"""Runs every experiment on both datasets and writes all outputs."""

from pathlib import Path

import pandas as pd

from .config import DATASETS
from .data import prepare_split
from .experiments import (
    ClusteringExperiment, DecisionTreeExperiment, ErrorAnalysis, KNNExperiment,
    NeuralNetExperiment, RandomForestExperiment,
)
from .submission import write_submission


def run_experiments(root, output_dir, quick=False, skip_neural_net=False, skip_trees=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results, nn_history_rows, clustering_rows = [], [], []
    final_predictions = {}

    for prefix in DATASETS:
        print("\n" + "=" * 72)
        print(f"DATASET: {prefix.upper()}")
        print("=" * 72)
        split = prepare_split(root, prefix)
        print(f"Rows: train={len(split.X_train)}, validation={len(split.X_val)}, "
              f"test={len(split.X_test)}")
        print(f"Features: {len(split.feature_cols)} | Classes: {split.n_classes}")

        experiments = [KNNExperiment(quick)]
        if not skip_trees:  # the pure-Python trees are by far the slowest part
            experiments += [DecisionTreeExperiment(quick), RandomForestExperiment(quick)]
        if not skip_neural_net:
            experiments.append(NeuralNetExperiment(quick))

        results = []
        for experiment in experiments:
            result = experiment.run(split)
            if result is None:  # e.g. PyTorch not installed
                continue
            all_results += result.rows
            nn_history_rows += result.extra.get("history", [])
            results.append(result)

        best = ErrorAnalysis(output_dir).run(split, results)
        clustering_rows += ClusteringExperiment(output_dir, quick).run(split)

        final_predictions[prefix] = (split.test_ids, best.predict_test())
        print(f"Selected {best.model_name} for the generated {prefix} test predictions.")

    pd.DataFrame(all_results).to_csv(output_dir / "validation_results.csv", index=False)
    print("\nValidation results saved to:", output_dir / "validation_results.csv")
    if nn_history_rows:
        pd.DataFrame(nn_history_rows).to_csv(output_dir / "nn_history.csv", index=False)
    if clustering_rows:
        pd.DataFrame(clustering_rows).to_csv(output_dir / "clustering_results.csv", index=False)

    write_submission(root, output_dir, final_predictions)
    print("Other outputs: confusion matrices, class-confusion CSVs, clustering CSVs, "
          "per-class recall, error overlap, nn_history.csv.")
