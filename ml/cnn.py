"""1D CNN classifier (optional, docs/guides/sadhil.md Part 5).

Operates on the raw 20x8 window instead of the 40 summary features -
metrics as channels, time as the sequence - so it can learn temporal
shapes the hand-engineered features don't capture. Normalized with
healthy-data statistics only (Part 4.3): computing mean/std from fault
windows would leak the answer into the input.

This is the most expendable model in the project. If it ties or loses
to the forest (ml/train.py) - or to Shravan's still-missing decision
tree - report that honestly and ship whichever simpler model won.

Run: python -m ml.cnn
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ml.dataset import _label_window
from ml.features import METRICS, WINDOW_SIZE, windows

EPOCHS = 200
LEARNING_RATE = 1e-2


class FaultCNN(nn.Module):
    def __init__(self, n_metrics: int, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_metrics, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def build_raw_dataset(metrics: pd.DataFrame, labels: pd.DataFrame, window_size: int = WINDOW_SIZE):
    """Like ml.dataset.build_dataset, but keeps each window as a raw
    (metrics, ticks) array instead of summarizing it.
    """
    arrays, rows = [], []

    for workload in metrics["workload"].unique():
        labels_for_workload = labels[labels["workload"] == workload]

        for window in windows(metrics, workload, window_size):
            ref_ts = window["ts"].iloc[-1]
            label, run_id, _ = _label_window(labels_for_workload, ref_ts)

            arrays.append(window[METRICS].to_numpy(dtype=float).T)
            rows.append({"workload": workload, "ts": ref_ts, "label": label, "run_id": run_id})

    return np.stack(arrays), pd.DataFrame(rows)


def normalize_on_healthy(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    healthy = X[(meta["label"] == "NORMAL").to_numpy()]
    mean = healthy.mean(axis=(0, 2), keepdims=True)
    std = healthy.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mean) / std


def train_cnn(X_train: np.ndarray, y_train: np.ndarray, n_metrics: int, n_classes: int) -> nn.Module:
    model = FaultCNN(n_metrics, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    return model


def main():
    metrics = pd.read_parquet("data/samples/metrics.parquet")
    labels = pd.read_csv("data/samples/labels.csv", parse_dates=["start_ts", "end_ts"])

    X, meta = build_raw_dataset(metrics, labels)
    meta = meta.merge(labels[["run_id", "pattern"]], on="run_id", how="left")
    X = normalize_on_healthy(X, meta)

    fault_mask = (meta["label"] != "NORMAL").to_numpy()
    X, meta = X[fault_mask], meta[fault_mask].reset_index(drop=True)

    classes = sorted(meta["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = meta["label"].map(class_to_idx).to_numpy()

    train_mask = (meta["pattern"] == "constant").to_numpy()
    test_mask = (meta["pattern"] == "ramp").to_numpy()

    start = time.time()
    model = train_cnn(X[train_mask], y[train_mask], n_metrics=X.shape[1], n_classes=len(classes))
    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X[test_mask], dtype=torch.float32)), dim=1)
        predictions = probs.argmax(dim=1).numpy()

    true_labels = [classes[i] for i in y[test_mask]]
    pred_labels = [classes[i] for i in predictions]

    print(f"trained in {elapsed:.1f}s on {int(train_mask.sum())} windows")
    print(f"test: {int(test_mask.sum())} windows\n")
    print(classification_report(true_labels, pred_labels))
    print(f"macro F1: {f1_score(true_labels, pred_labels, average='macro'):.3f}\n")
    print("confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(true_labels, pred_labels, labels=classes),
        index=classes, columns=classes,
    ))

    return model, classes


if __name__ == "__main__":
    main()
