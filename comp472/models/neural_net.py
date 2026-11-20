"""Feed-forward neural network (PyTorch), trained from scratch on the features."""

import time

from sklearn.metrics import accuracy_score

from ..config import SEED


def run_neural_net(X_train, y_train, X_val, y_val, X_test, epochs=25,
                   batch_size=256, learning_rate=1e-3, hidden=(256, 64),
                   dropouts=(0.20, 0.10), weight_decay=1e-4, verbose=True):
    """Train one MLP. Returns (val_pred, test_pred, (fit_s, predict_s), info).
    info = best epoch, training accuracy and per-epoch history.
    The weights from the epoch with the best validation accuracy are kept."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch not installed; skipping neural network. Install with: pip install torch")
        return None, None, None, None

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = int(max(y_train.max(), y_val.max())) + 1

    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            layers, prev = [], input_dim
            for size, p in zip(hidden, dropouts):
                layers += [nn.Linear(prev, size), nn.ReLU()]
                if p > 0:
                    layers.append(nn.Dropout(p))
                prev = size
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = MLP(X_train.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    start = time.time()
    best_state, best_acc, best_epoch = None, -1, 0
    history = []
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    for epoch in range(epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_val, pred)
        history.append((epoch + 1, total_loss / seen, acc))
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    Neural net epoch {epoch + 1}/{epochs}: validation accuracy={acc:.4f}")
    fit_seconds = time.time() - start

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    start = time.time()
    with torch.no_grad():
        val_pred = model(Xv).argmax(dim=1).cpu().numpy()
        Xt = torch.tensor(X_test, dtype=torch.float32, device=device)
        test_pred = model(Xt).argmax(dim=1).cpu().numpy()
    predict_seconds = time.time() - start
    with torch.no_grad():
        Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
        train_acc = accuracy_score(y_train, model(Xtr).argmax(dim=1).cpu().numpy())
    info = {"best_epoch": best_epoch, "train_accuracy": train_acc, "history": history}
    return (val_pred, test_pred, (fit_seconds, predict_seconds), info)
