"""Error analysis across the classifier experiments.

Rules questions this answers:
  - Which classes are most frequently confused?        -> {prefix}_most_confused_pairs.csv
  - Which classes are hardest / is Neutral harder?     -> {prefix}_per_class_recall.csv
  - Do multiple algorithms make the same mistakes?
    Does one model fix another's errors?               -> {prefix}_error_overlap.csv
Also picks the best model on validation accuracy (used for the Kaggle submission).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ..evaluation import save_confusion


class ErrorAnalysis:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, split, results):
        """results: list of ExperimentResult. Returns the best one on validation."""
        prefix, y_val = split.prefix, split.y_val
        # Ties go to the earlier experiment (kNN first), same as before the refactor.
        best = max(results, key=lambda r: accuracy_score(y_val, r.val_pred))
        print(f"\nBest validation model for {prefix}: {best.model_name}")
        print(classification_report(y_val, best.val_pred, zero_division=0))
        save_confusion(y_val, best.val_pred, f"{prefix}: {best.model_name}",
                       self.output_dir / f"{prefix}_confusion_matrix.png")

        self.most_confused_pairs(prefix, y_val, best.val_pred)
        self.compare_models(prefix, y_val, {r.model_name: r.val_pred for r in results})
        return best

    def most_confused_pairs(self, prefix, y_val, pred, top=20):
        cm = confusion_matrix(y_val, pred)
        np.fill_diagonal(cm, 0)
        pairs = [(int(cm[t, p]), int(t), int(p)) for t, p in zip(*np.where(cm > 0))]
        pairs.sort(reverse=True)
        pd.DataFrame(pairs[:top], columns=["count", "true_class", "predicted_class"]).to_csv(
            self.output_dir / f"{prefix}_most_confused_pairs.csv", index=False
        )

    def compare_models(self, prefix, y_val, candidate_preds):
        """Per-class recall of every model, and how much their mistakes overlap."""
        classes = np.unique(y_val)
        recall = {name: [np.mean(pred[y_val == c] == c) for c in classes]
                  for name, pred in candidate_preds.items()}
        recall_df = pd.DataFrame(recall, index=classes)
        recall_df["mean"] = recall_df.mean(axis=1)
        recall_df.sort_values("mean").to_csv(
            self.output_dir / f"{prefix}_per_class_recall.csv", index_label="class"
        )
        print(f"\nHardest {prefix} classes (lowest recall averaged over all models):")
        print(recall_df.sort_values("mean").head(5).round(3).to_string())

        wrong = {name: pred != y_val for name, pred in candidate_preds.items()}
        names = list(wrong)
        all_wrong = np.logical_and.reduce([wrong[n] for n in names])
        print(f"Validation samples misclassified by ALL models: {all_wrong.mean():.1%}")
        rows = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = int(np.sum(wrong[a] & wrong[b]))
                either = int(np.sum(wrong[a] | wrong[b]))
                rows.append({
                    "model_a": a, "model_b": b,
                    "jaccard_of_errors": both / max(either, 1),
                    "pct_of_a_errors_fixed_by_b": np.sum(wrong[a] & ~wrong[b]) / max(wrong[a].sum(), 1),
                    "pct_of_b_errors_fixed_by_a": np.sum(wrong[b] & ~wrong[a]) / max(wrong[b].sum(), 1),
                })
        pd.DataFrame(rows).to_csv(self.output_dir / f"{prefix}_error_overlap.csv", index=False)
