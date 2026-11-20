#!/usr/bin/env python3
"""
COMP 472 ML experiments for image and sentiment feature-vector datasets.

Entry point only - the code lives in the comp472/ package
(see comp472/__init__.py for the file layout).

Expected ZIP contents:
  image_train.csv, image_test.csv
  sentiment_train.csv, sentiment_test.csv
  sample_submission.csv

Run:
  python3 comp472_experiments.py --zip data.zip

Optional:
  python3 comp472_experiments.py --zip data.zip --quick   # only the QUICK_BEST settings
  python3 comp472_experiments.py --zip data.zip --skip-neural-net
  python3 comp472_experiments.py --zip data.zip --skip-trees      # fast tuning (no tree/forest)
"""

import argparse
import os

from comp472.config import set_seeds
from comp472.data import extract_zip
from comp472.runner import run_experiments


def main():
    parser = argparse.ArgumentParser(description="Run COMP 472 ML experiments.")
    parser.add_argument("--zip", required=True, help="Path to the competition ZIP file.")
    parser.add_argument("--workdir", default="comp472_data", help="ZIP extraction folder.")
    parser.add_argument("--output", default="comp472_results", help="Results output folder.")
    parser.add_argument("--quick", action="store_true",
                        help="Train only the best known settings (QUICK_BEST) for a fast submission.")
    parser.add_argument("--skip-neural-net", action="store_true", help="Skip PyTorch neural network.")
    parser.add_argument("--skip-trees", action="store_true",
                        help="Skip Decision Tree and Random Forest (fast tuning runs).")
    args = parser.parse_args()

    if not os.path.isfile(args.zip):
        raise FileNotFoundError(f"ZIP file not found: {args.zip}")
    set_seeds()
    root = extract_zip(args.zip, args.workdir)
    run_experiments(root, args.output, quick=args.quick,
                    skip_neural_net=args.skip_neural_net, skip_trees=args.skip_trees)


if __name__ == "__main__":
    main()
