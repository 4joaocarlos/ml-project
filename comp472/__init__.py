"""COMP 472 project: image and sentiment classification on pre-extracted feature vectors.

Package layout (one file per algorithm / experiment from the project rules):

  config.py              SEED and the QUICK_BEST settings used by --quick
  data.py                ZIP extraction, CSV loading, train/validation split + scaling
  evaluation.py          shared metrics helpers (score_model, confusion plot, purity)
  submission.py          builds and validates the Kaggle submission.csv
  runner.py              runs every experiment on both datasets

  models/                the algorithms themselves
    knn.py               k-NN (NumPy, from scratch)
    decision_tree.py     CART decision tree (NumPy, from scratch)
    random_forest.py     random forest built from the scratch trees
    neural_net.py        feed-forward MLP (PyTorch)
    kmeans.py            k-means++ (NumPy, from scratch)

  experiments/           the hyperparameter studies required by the rules
    base.py              ExperimentResult returned by every classifier experiment
    knn.py               k, distance, normalization, weighted voting
    decision_tree.py     depth, min_samples_leaf, criterion, train/val gap
    random_forest.py     n_trees, depth, max_features, train/val gap
    neural_net.py        learning rate, architecture, dropout, weight decay
    clustering.py        unsupervised k-means experiment (ARI, NMI, purity)
    error_analysis.py    best-model report, confusion pairs, per-class recall, error overlap
"""
