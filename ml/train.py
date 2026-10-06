"""Train the random forest classifier on labeled fault windows.

Split by run, never by row (docs/guides/sadhil.md Part 4.3): trains on
constant-pattern runs, tests on ramp-pattern runs, so accuracy reflects
generalization instead of near-duplicate windows leaking across the split.

Run: python -m ml.train
    python -m ml.train --metrics data/chaos/metrics.parquet --labels data/chaos/labels.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from ml.dataset import build_dataset

FEATURE_SUFFIXES = ("_mean", "_slope", "_std", "_max", "_last")

# NORMAL: quiet periods - Sagar's detector filters those out before a
# window ever reaches the classifier (docs/guides/sadhil.md Part 8).
# POD_KILL: instantaneous, no diagnosis lever in control/policy.py -
# it's reported separately as the near-zero-lead-time honest failure
# case (docs/guides/shaurya.md Part 5.5), never a classification target.
NON_CLASSIFIABLE_LABELS = ("NORMAL", "POD_KILL")

# Prefer real chaos-run data over the Phase 0 synthetic fixture, whichever is
# actually on disk (SETUP.md S8: files, not services). This is the single
# definition; ml.classifier and ml.abstention both defer to it, so the model
# that ships and the numbers we report can never come from different datasets.
# They could before: ml.train defaulted to data/samples/ via argparse while
# classifier.fit_and_save() preferred data/chaos/, which is how a 0.99
# synthetic accuracy and a 0.82 real accuracy were both true at once.
CHAOS_METRICS = Path("data/chaos/metrics.parquet")
CHAOS_LABELS = Path("data/chaos/labels.csv")
SAMPLE_METRICS = Path("data/samples/metrics.parquet")
SAMPLE_LABELS = Path("data/samples/labels.csv")


def default_data_paths():
    """(metrics, labels) to use when the caller did not name any. No side effects."""
    if CHAOS_METRICS.exists() and CHAOS_LABELS.exists():
        return CHAOS_METRICS, CHAOS_LABELS
    return SAMPLE_METRICS, SAMPLE_LABELS


def load_labeled_windows(metrics_path="data/samples/metrics.parquet", labels_path="data/samples/labels.csv"):
    metrics = pd.read_parquet(metrics_path)
    labels = pd.read_csv(labels_path, parse_dates=["start_ts", "end_ts"])

    df = build_dataset(metrics, labels)
    df = df.merge(labels[["run_id", "pattern"]], on="run_id", how="left")

    return df[~df["label"].isin(NON_CLASSIFIABLE_LABELS)].reset_index(drop=True)


def split_by_run(df):
    train = df[df["pattern"] == "constant"]
    test = df[df["pattern"] == "ramp"]
    return train, test


def feature_columns(df):
    return [c for c in df.columns if c.endswith(FEATURE_SUFFIXES)]


def main():
    default_metrics, default_labels = default_data_paths()

    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(default_metrics))
    ap.add_argument("--labels", default=str(default_labels))
    args = ap.parse_args()

    df = load_labeled_windows(args.metrics, args.labels)
    print(f"source: {args.metrics}, {args.labels}\n")
    print("label distribution:\n", df["label"].value_counts(), "\n")

    train, test = split_by_run(df)
    features = feature_columns(df)

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)
    clf.fit(train[features], train["label"])

    predictions = clf.predict(test[features])

    print(f"train: {len(train)} windows, {train['run_id'].nunique()} runs")
    print(f"test:  {len(test)} windows, {test['run_id'].nunique()} runs\n")
    print(classification_report(test["label"], predictions))

    labels_order = sorted(df["label"].unique())
    print("confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(test["label"], predictions, labels=labels_order),
        index=labels_order, columns=labels_order,
    ))

    # Sanity check (docs/guides/sadhil.md Part 6.3): if mem_pct_slope isn't
    # near the top, something's off with the labels, not just the model.
    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print("\ntop 10 feature importances:")
    print(importances.head(10))

    return clf


if __name__ == "__main__":
    main()
