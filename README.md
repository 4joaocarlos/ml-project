# COMP 472 Project — Image & Sentiment Classification on Feature Vectors

**Team:** COMP472_Group27_NN

**Members:**
- Juan Carlos Bastidas (40228022)
- Maxime Duquette (40248944)
- Kadio Paul-Philippe Ehui (40269547)

This repository contains the code for two classification tasks on pre-extracted feature vectors: visual object recognition (50 classes) and sentiment analysis (3 classes). It trains and evaluates k-Nearest Neighbours, a Decision Tree, a Random Forest, a feed-forward neural network and k-means clustering, then writes a Kaggle submission file. The methods and results are described in the project report.

---

## 1. Repository contents

| File / folder | Purpose |
|---|---|
| `comp472_experiments.py` | Entry point. Reads the command-line options and runs the full pipeline: pre-processing, training, evaluation and submission. |
| `comp472/config.py` | Global random seed (`SEED`) and the best known settings used by `--quick` (`QUICK_BEST`). |
| `comp472/data.py` | Data pre-processing: loads the CSVs, makes the train/validation split, and applies the z-score and Hellinger transforms. |
| `comp472/evaluation.py` | Evaluation helpers: metrics, confusion-matrix plot, cluster purity. |
| `comp472/submission.py` | Builds `submission.csv` and checks that its format is valid. |
| `comp472/runner.py` | Runs every experiment on both tasks. |
| `comp472/models/knn.py` | k-Nearest Neighbours model. |
| `comp472/models/decision_tree.py` | Decision Tree (CART) model. |
| `comp472/models/random_forest.py` | Random Forest built from our decision trees. |
| `comp472/models/neural_net.py` | Feed-forward neural network (PyTorch). |
| `comp472/models/kmeans.py` | k-means++ clustering. |
| `comp472/experiments/knn.py` | k-NN experiments (hyperparameter configurations). |
| `comp472/experiments/decision_tree.py` | Decision Tree experiments. |
| `comp472/experiments/random_forest.py` | Random Forest experiments. |
| `comp472/experiments/neural_net.py` | Neural network experiments. |
| `comp472/experiments/clustering.py` | Unsupervised k-means experiment. |
| `comp472/experiments/error_analysis.py` | Report on the best model and error analysis across models. |
| `.gitignore` | Excludes the competition data and ZIP from the repository. |

---

## 2. Setup

**Environment used:** Python 3.14.0 on macOS (Apple M1, CPU only, no GPU required).

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy==2.5.5 pandas==3.1.1 scikit-learn==1.9.1 matplotlib==3.11.2 torch==2.15.0
```

---

## 3. Data

1. Download the competition ZIP from the Kaggle **Data** tab.
2. Place the ZIP in the project root (the folder that contains `comp472_experiments.py`).
3. Check that the ZIP contains these files:

```
image_train.csv   image_test.csv
sentiment_train.csv   sentiment_test.csv
sample_submission.csv
```

The data is not included in the repository.

---

## 4. Data pre-processing

Pre-processing runs automatically as the first stage of the main command (Section 5). No separate command is needed. When you run `comp472_experiments.py`, it:

1. Extracts the CSV files from the ZIP into `comp472_data/` (or the folder given with `--workdir`).
2. Loads the training and test CSVs for both tasks (`comp472/data.py`).
3. Splits the labelled training data into 80% train and 20% validation, stratified by class, with a fixed seed.
4. Builds the feature representations used by the models: z-score standardization (fitted on the train split only) for both tasks, and the Hellinger transform for the image task.

---

## 5. Training and evaluation

Run all commands from the project root, with the virtual environment activated.

**Full run** (trains and evaluates every configuration):

```bash
python comp472_experiments.py --zip comp-472-project-fall-2026-concordia.zip
```

**Quick run** (trains only the best known settings in `QUICK_BEST`):

```bash
python comp472_experiments.py --zip comp-472-project-fall-2026-concordia.zip --quick
```

If your ZIP has a different name, give its path after `--zip`.

### Command-line options

| Option | Meaning |
|---|---|
| `--zip PATH` | path to the competition ZIP (required) |
| `--quick` | train only the settings in `QUICK_BEST` |
| `--skip-trees` | skip the Decision Tree and Random Forest |
| `--skip-neural-net` | skip the neural network |
| `--workdir DIR` | folder for the extracted CSVs (default `comp472_data`) |
| `--output DIR` | folder for the results (default `comp472_results`) |

### Approximate runtime (Apple M1 CPU)

| Command | Runtime |
|---|---|
| `--quick --skip-trees` | about 25 s |
| `--quick` | about 10 min |
| full run | about 30–60 min (mostly the Random Forest) |

### Evaluation outputs

Each model is evaluated on the validation split. The results are written to `comp472_results/` (or the folder given with `--output`):

| File | Content |
|---|---|
| `validation_results.csv` | one row per configuration: accuracy, macro precision/recall/F1, train accuracy, fit/predict time |
| `nn_history.csv` | training loss and validation accuracy per epoch for each neural network configuration |
| `{task}_confusion_matrix.png` | confusion matrix of the best model |
| `{task}_most_confused_pairs.csv` | the 20 most frequent (true → predicted) mistakes |
| `{task}_per_class_recall.csv` | recall per class for every algorithm |
| `{task}_error_overlap.csv` | how much the algorithms' mistakes overlap and how many of each other's errors they fix |
| `{task}_clustering.csv`, `clustering_results.csv` | k-means ARI, NMI, purity and inertia for each number of clusters |
| `{task}_cluster_vs_class_k*.csv` | cluster × true-class table when k equals the number of classes |

`{task}` is `image` or `sentiment`.

---

## 6. Applying the model (predictions on the test set)

The same command (full or quick run) also applies the models to the test data. For each task, the script selects the model with the best validation accuracy, predicts the labels of the test CSV, and writes:

```
comp472_results/submission.csv
```

The file has 10,000 rows with the columns `id,label`. Its format is checked by `comp472/submission.py`. To score it, upload `submission.csv` on the Kaggle competition page.

---

## 7. Reproducibility

All random operations use `SEED = 42`, set in `comp472/config.py`. Results should match up to small floating-point differences between machines and library versions.