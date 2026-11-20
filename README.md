# COMP 472 Project — Image & Sentiment Classification on Feature Vectors

**Team:** COMP472_GroupXX
**Members:** _Name 1 (ID)_, _Name 2 (ID)_, _Name 3 (ID)_

This project compares several AI algorithms on two classification tasks that use pre-extracted feature vectors:

| Task | Classes | Train / Test | Features |
|---|---|---|---|
| Visual object recognition | 50 | 20,000 / 5,000 | 500 (sparse, non-negative, unit-norm) |
| Sentiment analysis (frozen BERT, mean pooling) | 3 (Negative / Neutral / Positive) | 12,000 / 5,000 | 768 |

We implemented and evaluated:

- **k-Nearest Neighbours**, from scratch in NumPy
- **Decision Tree (CART)**, from scratch in NumPy
- **Random Forest**, built from our own decision trees
- **Feed-forward Neural Network**, in PyTorch
- **k-means clustering** (unsupervised experiment), from scratch in NumPy

The script writes a Kaggle submission that uses, for each task, the model with the best validation accuracy.

---

## 1. Setup

**Environment used:** Python 3.14.0 on macOS (Apple M1, CPU only, no GPU required).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy==2.5.5 pandas==3.1.1 scikit-learn==1.9.1 matplotlib==3.11.2 torch==2.15.0
```

| Library | Version | Used for |
|---|---|---|
| NumPy | 2.5.5 | all from-scratch algorithms |
| pandas | 3.1.1 | reading CSVs, building the submission |
| scikit-learn | 1.9.1 | train/validation split, `StandardScaler`, metrics only |
| Matplotlib | 3.11.2 | confusion-matrix plots |
| PyTorch | 2.15.0 | neural network |

## 2. Data

Download the competition ZIP from the Kaggle **Data** tab and put it in the project root. It must contain:

```
image_train.csv   image_test.csv
sentiment_train.csv   sentiment_test.csv
sample_submission.csv
```

The script extracts the CSVs into `comp472_data/`. The data and the ZIP are not committed, as listed in `.gitignore`.

## 3. How to run

```bash
# Full experiments: every hyperparameter configuration (used for the report)
python comp472_experiments.py --zip comp-472-project-fall-2026-concordia.zip

# Quick: only the best known settings (QUICK_BEST in comp472/config.py)
python comp472_experiments.py --zip comp-472-project-fall-2026-concordia.zip --quick

# Fast tuning: skip the (slow, pure-Python) Decision Tree and Random Forest
python comp472_experiments.py --zip comp-472-project-fall-2026-concordia.zip --quick --skip-trees
```

| Option | Meaning |
|---|---|
| `--zip PATH` | competition ZIP (required) |
| `--quick` | train only the settings in `QUICK_BEST` |
| `--skip-trees` | skip Decision Tree and Random Forest |
| `--skip-neural-net` | skip the PyTorch MLP |
| `--workdir DIR` | extraction folder (default `comp472_data`) |
| `--output DIR` | results folder (default `comp472_results`) |

**Approximate runtime (Apple M1 CPU):**

| Command | Runtime |
|---|---|
| `--quick --skip-trees` | about 25 s |
| `--quick` | about 10 min |
| full run | about 30–60 min (most of it in the from-scratch Random Forest) |

## 4. Outputs (`comp472_results/`)

| File | Content |
|---|---|
| `submission.csv` | Kaggle submission (10,000 rows, `id,label`) |
| `validation_results.csv` | one row per configuration: accuracy, macro precision/recall/F1, train accuracy, fit/predict time |
| `nn_history.csv` | per-epoch training loss and validation accuracy for each MLP configuration |
| `{task}_confusion_matrix.png` | confusion matrix of the best model |
| `{task}_most_confused_pairs.csv` | the 20 most frequent (true → predicted) mistakes |
| `{task}_per_class_recall.csv` | recall per class for every algorithm |
| `{task}_error_overlap.csv` | how much the algorithms' mistakes overlap and how many of each other's errors they fix |
| `{task}_clustering.csv`, `clustering_results.csv` | k-means ARI, NMI, purity and inertia for each number of clusters |
| `{task}_cluster_vs_class_k*.csv` | cluster × true-class table when k equals the number of classes |

## 5. Project structure

```
comp472_experiments.py        entry point (command-line options)
comp472/
  config.py                   SEED and QUICK_BEST (best settings found)
  data.py                     loading, train/validation split, z-score and Hellinger transforms
  evaluation.py               metrics helpers, confusion-matrix plot, cluster purity
  submission.py               builds and validates submission.csv
  runner.py                   runs every experiment on both tasks
  models/                     the algorithms
    knn.py                    k-NN (from scratch)
    decision_tree.py          CART decision tree (from scratch)
    random_forest.py          random forest of our trees (from scratch)
    neural_net.py             MLP (PyTorch)
    kmeans.py                 k-means++ (from scratch)
  experiments/                one class per experiment / hyperparameter study
    knn.py  decision_tree.py  random_forest.py  neural_net.py
    clustering.py             unsupervised experiment
    error_analysis.py         best-model report and cross-model error analysis
```

## 6. Methodology

### Validation

- The labelled training data is split **80% train / 20% validation**, stratified by class, with `random_state=42`.
- Every model is compared, tuned and selected on the validation split only. The Kaggle leaderboard is not used for model selection.
- The z-score scaler is fitted on the 80% train split only, so no validation statistics leak into training.
- The Hellinger transform is a fixed per-row formula with nothing to fit.
- For the submission, k-NN uses all 100% of the training rows as neighbours, since it has nothing to train. The other models are trained on the 80% split.

### Feature preprocessing

| Name | Definition | Used for |
|---|---|---|
| `zscore` | per-feature standardization | all models |
| `hellinger` | `sqrt(x)` then L2-normalize each row | image only |
| `none` | raw features | k-NN comparison |

The image features are 85% zeros, non-negative and unit-norm, so they behave like histograms. That's why the Hellinger transform applies to them. The sentiment features have negative values, so the transform isn't defined for them.

### Experiments

Each experiment changes one hyperparameter at a time, relative to a base configuration. The configurations are listed in `comp472/experiments/`.

| Algorithm | Hyperparameters studied |
|---|---|
| k-NN | k ∈ {1, 3, 5, 10, 20, 30, 50, 100}; voting: uniform, 1/d, 1/d², Gaussian; distance: Euclidean, cosine; normalization: none, z-score, Hellinger |
| Decision Tree | max depth ∈ {3, 5, 10, 15, 20}; min samples per leaf ∈ {1, 5, 20}; criterion: Gini, entropy |
| Random Forest | trees ∈ {20, 40}; max depth ∈ {8, 14, 20}; features per split: √d, log₂d |
| Neural Network | learning rate ∈ {1e-2, 1e-3, 3e-4, 1e-4}; hidden layers (none/linear, 256-64, 512-256, 1024-512); dropout; weight decay; input representation |
| k-means | k ∈ {10, 25, 50, 100} (image), {2, 3, 5, 10} (sentiment) |

**Neural network training:**
- AdamW optimizer, cross-entropy loss, batch size 256, ReLU, 40 epochs
- the weights from the epoch with the best validation accuracy are kept (early stopping)

**Decision tree and random forest:**
- candidate split thresholds are 24 quantiles of each feature
- the forest uses bootstrap sampling and a majority vote

### Error analysis

The error analysis covers:
- per-class precision, recall and F1, plus a confusion matrix for the best model
- the most frequently confused class pairs
- per-class recall for every algorithm
- for each pair of algorithms, how much their errors overlap (Jaccard) and what fraction of one model's errors the other gets right

## 7. Results (validation accuracy, best configuration of each algorithm)

| Method | Image | Sentiment |
|---|---|---|
| k-NN | **0.572** (Hellinger, k=1) | 0.586 (z-score, k=50, 1/d²) |
| Decision Tree | 0.102 (depth 20) | 0.493 (depth 5) |
| Random Forest | 0.161 (40 trees, depth 14) | 0.565 (40 trees, depth 14) |
| Neural Network | 0.377 (512-256) | **0.681** (256-64, lr 3e-4, dropout 0.5/0.3, wd 1e-2) |

**Submitted models:**
- Image: k-NN on Hellinger features, k=1
- Sentiment: MLP

The average validation accuracy is about 0.627. Kaggle private-leaderboard score: _to be filled in_.

The full per-configuration results are in `comp472_results/validation_results.csv`.

## 8. Reproducibility

- `SEED = 42` (`comp472/config.py`) is used for Python `random`, NumPy, the train/validation split, every tree and forest (`default_rng`), k-means, and `torch.manual_seed`.
- The models run on CPU. Results should match up to small floating-point differences between machines and library versions.

## 9. Coding rules compliance

- k-NN, Decision Tree, Random Forest and k-means are implemented from scratch in NumPy. No `sklearn.neighbors`, `sklearn.tree`, `sklearn.ensemble` or `sklearn.cluster` models are used.
- The neural network uses `torch.nn`, `torch.optim` and `torch.utils.data`, which the rules allow.
- scikit-learn is used only for the train/validation split, standardization and evaluation metrics, as the rules allow.
- Only the provided feature vectors and training labels are used: no external data, no pretrained models, no extra features.
