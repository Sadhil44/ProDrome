"""Train the random forest classifier on labeled fault windows.

Split by run, never by row (docs/guides/sadhil.md Part 4.3): trains on
constant-pattern runs, tests on ramp-pattern runs, so accuracy reflects
generalization instead of near-duplicate windows leaking across the split.

Run: python -m ml.train
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from ml.dataset import build_dataset

FEATURE_SUFFIXES = ("_mean", "_slope", "_std", "_max", "_last")


def load_labeled_windows(metrics_path="data/samples/metrics.parquet", labels_path="data/samples/labels.csv"):
    metrics = pd.read_parquet(metrics_path)
    labels = pd.read_csv(labels_path, parse_dates=["start_ts", "end_ts"])

    df = build_dataset(metrics, labels)
    df = df.merge(labels[["run_id", "pattern"]], on="run_id", how="left")

    # Classify firings, not quiet periods - Sagar's detector already
    # filters those out before a window ever reaches the classifier.
    # See docs/guides/sadhil.md Part 8.
    return df[df["label"] != "NORMAL"].reset_index(drop=True)


def split_by_run(df):
    train = df[df["pattern"] == "constant"]
    test = df[df["pattern"] == "ramp"]
    return train, test


def feature_columns(df):
    return [c for c in df.columns if c.endswith(FEATURE_SUFFIXES)]


def main():
    df = load_labeled_windows()
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

    return clf


if __name__ == "__main__":
    main()
