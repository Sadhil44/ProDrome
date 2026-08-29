"""Shared feature code for Signal (detector) and Diagnosis (classifier).

Owned by Sagar per docs/guides/sagar.md Part 4.1: canonical metric order
and windowing. Sadhil extends it with feature summarization -- one file,
two owners. Reconciled out of the provisional copies in ml/dataset.py
(windows) and this file's original standalone version (summarize) per
both guides' Phase 0 notes.

Named features.py rather than signal.py: a module named signal.py
shadows Python's stdlib `signal` module for anything run from inside
ml/, since that adds ml/ to sys.path -- pandas imports subprocess,
which imports the real signal, and gets this file instead.

METRICS order is frozen: Sadhil and Shravan both index arrays by
position. Reordering this silently breaks both of them.
"""

import numpy as np
import pandas as pd

METRICS = [
    "cpu_cores",
    "mem_bytes",
    "mem_pct",
    "net_rx",
    "net_tx",
    "fs_reads",
    "fs_writes",
    "restarts",
]

WINDOW_SIZE = 20  # 20 ticks x 15s = 5 minutes of history


def windows(metrics: pd.DataFrame, workload: str, size: int = WINDOW_SIZE):
    """Yield sliding size-tick windows for one workload, oldest first.

    Missing-value handling, decided once here per Part 4.1: forward-fill,
    then fill any still-missing leading values with zero.
    """
    sub = metrics[metrics["workload"] == workload].sort_values("ts").reset_index(drop=True)
    sub[METRICS] = sub[METRICS].ffill().fillna(0)

    for start in range(len(sub) - size + 1):
        yield sub.iloc[start : start + size]


def summarize(window):
    """window: a table with one column per metric in METRICS, `size` rows.

    Returns a flat dict of 5 features per metric: mean, slope, std, max, last.
    """
    features = {}
    ticks = np.arange(len(window))

    for metric in METRICS:
        values = np.asarray(window[metric], dtype=float)
        slope = np.polyfit(ticks, values, 1)[0]

        features[f"{metric}_mean"] = values.mean()
        features[f"{metric}_slope"] = slope
        features[f"{metric}_std"] = values.std()
        features[f"{metric}_max"] = values.max()
        features[f"{metric}_last"] = values[-1]

    return features
