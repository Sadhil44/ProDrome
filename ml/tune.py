"""Parameter sweep (Part 5.1).

Fits and replays the detector under each of the 24 (alpha, threshold, k, n)
combinations the guide names, and reports recall/lead-time per fault type
plus false positives per hour for each. Deliberately does not pick a
"winner" -- Part 5.2 says there's no config best on both precision and
recall, so that choice is a human call, not this script's.
"""

import itertools

import pandas as pd

from ml.detector import Detector
from ml.replay import fault_mask, firing_events, replay, train_and_replay

ALPHAS = [0.1, 0.3]
THRESHOLDS = [2.5, 3.0, 4.0]
KS = [2, 3]
NS = [1, 2]


def split_healthy(metrics: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Drop every tick that falls inside any labeled fault window."""
    return metrics[~fault_mask(metrics, labels)]


def _match_events_to_labels(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """For each labeled fault, find the earliest firing event (if any) whose
    onset falls inside [start_ts, end_ts] for that workload. Adds `detected`
    and `lead_time_s` (end_ts - onset_ts, seconds) columns to a copy of labels."""
    labels = labels.copy()
    labels["detected"] = False
    labels["lead_time_s"] = pd.NA

    for i, lab in labels.iterrows():
        matches = events[
            (events["workload"] == lab["workload"])
            & (events["onset_ts"] >= lab["start_ts"])
            & (events["onset_ts"] <= lab["end_ts"])
        ]
        if not matches.empty:
            onset = matches["onset_ts"].min()
            labels.at[i, "detected"] = True
            labels.at[i, "lead_time_s"] = (lab["end_ts"] - onset).total_seconds()

    return labels


def _false_positive_rate(events: pd.DataFrame, labels: pd.DataFrame, healthy: pd.DataFrame) -> float:
    """Firing events whose onset lands outside every labeled window, per healthy hour."""
    is_inside_any_label = pd.Series(False, index=events.index)
    for _, lab in labels.iterrows():
        mask = (
            (events["workload"] == lab["workload"])
            & (events["onset_ts"] >= lab["start_ts"])
            & (events["onset_ts"] <= lab["end_ts"])
        )
        is_inside_any_label |= mask

    false_positives = (~is_inside_any_label).sum()
    healthy_hours = len(healthy) * 15 / 3600  # 15s ticks
    return false_positives / healthy_hours if healthy_hours > 0 else float("nan")


def evaluate_config(alpha, threshold, k, n, healthy, metrics, labels) -> dict:
    det, log = train_and_replay(metrics, labels, alpha=alpha, threshold=threshold, k=k, n=n)
    events = firing_events(log)

    matched = _match_events_to_labels(events, labels)
    fp_per_hour = _false_positive_rate(events, labels, healthy)

    row = {"alpha": alpha, "threshold": threshold, "k": k, "n": n, "fp_per_hour": round(fp_per_hour, 2)}
    for fault_type, group in matched.groupby("fault_type"):
        recall = group["detected"].mean()
        lead_times = group.loc[group["detected"], "lead_time_s"].astype(float)
        median_lead = lead_times.median() if not lead_times.empty else float("nan")
        row[f"{fault_type}_recall"] = round(recall, 2)
        row[f"{fault_type}_lead_s"] = round(median_lead, 1) if pd.notna(median_lead) else float("nan")

    return row


def sweep(metrics: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    healthy = split_healthy(metrics, labels)
    rows = []
    for alpha, threshold, k, n in itertools.product(ALPHAS, THRESHOLDS, KS, NS):
        rows.append(evaluate_config(alpha, threshold, k, n, healthy, metrics, labels))
    return pd.DataFrame(rows)


def evaluate_config_real(alpha, threshold, k, n, real_healthy, chaos_metrics, chaos_labels) -> dict:
    """For genuinely separate healthy/chaos files (no time overlap) -- fit
    on real_healthy, replay chaos_metrics fresh. Unlike evaluate_config,
    this is safe to use fit_healthy + replay for (see ml.replay's
    module docstring on when each is correct)."""
    det = Detector.fit_healthy(real_healthy, alpha=alpha, threshold=threshold, k=k, n=n)
    log = replay(det, chaos_metrics)
    events = firing_events(log)

    matched = _match_events_to_labels(events, chaos_labels)
    clean_within_chaos = chaos_metrics[~fault_mask(chaos_metrics, chaos_labels)]
    fp_per_hour = _false_positive_rate(events, chaos_labels, clean_within_chaos)

    row = {"alpha": alpha, "threshold": threshold, "k": k, "n": n, "fp_per_hour": round(fp_per_hour, 2)}
    for fault_type, group in matched.groupby("fault_type"):
        recall = group["detected"].mean()
        lead_times = group.loc[group["detected"], "lead_time_s"].astype(float)
        median_lead = lead_times.median() if not lead_times.empty else float("nan")
        row[f"{fault_type}_recall"] = round(recall, 2)
        row[f"{fault_type}_lead_s"] = round(median_lead, 1) if pd.notna(median_lead) else float("nan")

    return row


def sweep_real(real_healthy: pd.DataFrame, chaos_metrics: pd.DataFrame, chaos_labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha, threshold, k, n in itertools.product(ALPHAS, THRESHOLDS, KS, NS):
        rows.append(evaluate_config_real(alpha, threshold, k, n, real_healthy, chaos_metrics, chaos_labels))
    return pd.DataFrame(rows)


def main():
    metrics = pd.read_parquet("data/samples/metrics.parquet")
    labels = pd.read_csv("data/samples/labels.csv", parse_dates=["start_ts", "end_ts"])

    results = sweep(metrics, labels)
    pd.set_option("display.width", 200)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
