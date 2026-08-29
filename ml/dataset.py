"""Build a training-ready dataset from a metrics table and a labels file.
"""

import pandas as pd

from ml.features import summarize

WINDOW_SIZE = 20


def windows(metrics: pd.DataFrame, workload: str, size: int = WINDOW_SIZE):
    """Yield sliding size-tick windows for one workload, oldest first."""
    sub = metrics[metrics["workload"] == workload].sort_values("ts").reset_index(drop=True)
    for start in range(len(sub) - size + 1):
        yield sub.iloc[start : start + size]


def _label_window(labels_for_workload: pd.DataFrame, ref_ts):
    """Look up which fault (if any) was active at ref_ts.

    Returns (label, run_id, seconds_to_failure). NORMAL windows get
    run_id=None and seconds_to_failure=None - there's no failure
    pending to count down to.
    """
    match = labels_for_workload[
        (labels_for_workload["start_ts"] <= ref_ts) & (ref_ts <= labels_for_workload["end_ts"])
    ]
    if match.empty:
        return "NORMAL", None, None

    row = match.iloc[0]
    return row["fault_type"], row["run_id"], row["end_ts"] - ref_ts


def build_dataset(metrics: pd.DataFrame, labels: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """One row per window: 40 features + workload, ts, label, run_id, seconds_to_failure."""
    rows = []

    for workload in metrics["workload"].unique():
        labels_for_workload = labels[labels["workload"] == workload]

        for window in windows(metrics, workload, window_size):
            ref_ts = window["ts"].iloc[-1]
            label, run_id, seconds_to_failure = _label_window(labels_for_workload, ref_ts)

            row = summarize(window)
            row["workload"] = workload
            row["ts"] = ref_ts
            row["label"] = label
            row["run_id"] = run_id
            row["seconds_to_failure"] = seconds_to_failure
            rows.append(row)

    return pd.DataFrame(rows)
