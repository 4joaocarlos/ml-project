"""Builds submission.csv in sample_submission.csv ID order, with the rules' checks."""

import pandas as pd

from .data import find_csv


def write_submission(root, output_dir, predictions_by_dataset):
    """predictions_by_dataset: {"image": (test_ids, labels), "sentiment": (test_ids, labels)}"""
    predictions = pd.concat(
        [pd.DataFrame({"id": ids, "label": labels})
         for ids, labels in predictions_by_dataset.values()],
        ignore_index=True,
    )

    sample = pd.read_csv(find_csv(root, "sample_submission.csv"))
    if list(sample.columns)[:1] != ["id"]:
        raise ValueError("sample_submission.csv must contain an id column.")
    submission = sample[["id"]].merge(predictions, on="id", how="left", validate="one_to_one")

    if len(submission) != 10000:
        raise ValueError(f"Expected 10,000 submission rows; found {len(submission)}.")
    if submission["label"].isna().any():
        missing = submission.loc[submission["label"].isna(), "id"].head().tolist()
        raise ValueError(f"Missing predictions for IDs such as: {missing}")
    submission["label"] = submission["label"].astype(int)
    image_mask = submission["id"].str.startswith("img_")
    sentiment_mask = submission["id"].str.startswith("text_")
    if not submission.loc[image_mask, "label"].between(0, 49).all():
        raise ValueError("Image predictions must be integers from 0 to 49.")
    if not submission.loc[sentiment_mask, "label"].between(0, 2).all():
        raise ValueError("Sentiment predictions must be integers from 0 to 2.")

    path = output_dir / "submission.csv"
    submission.to_csv(path, index=False)
    print("Kaggle submission saved to:", path)
    return path
