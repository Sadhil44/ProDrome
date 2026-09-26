"""Abstention experiment: train on two fault types, hold a third out
entirely, and measure whether the model knows what it doesn't know.

See docs/guides/sadhil.md Part 6.1. "Confidently wrong" is the
dangerous number - those are exactly the cases where Prodrome would
take a real, irreversible action based on a fault it has never seen.

Run: python -m ml.abstention
"""

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from control.policy import ABSTENTION_FLOOR
from ml.train import default_data_paths, feature_columns, load_labeled_windows

DANGEROUS_CONFIDENCE = 0.80


def run_abstention_experiment(df: pd.DataFrame, features, held_out_class: str) -> dict:
    train = df[df["label"] != held_out_class]
    test = df[df["label"] == held_out_class]

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)
    clf.fit(train[features], train["label"])

    confidence = clf.predict_proba(test[features]).max(axis=1)
    total = len(test)

    return {
        "held_out_class": held_out_class,
        "total": total,
        "correctly_abstained": int((confidence < ABSTENTION_FLOOR).sum()),
        "confidently_wrong": int((confidence > DANGEROUS_CONFIDENCE).sum()),
    }


def main():
    # Same defaults as ml.train. This used to call load_labeled_windows() with
    # no arguments, which hardcoded it to data/samples/ with no way to override
    # -- so the published "confidently wrong" number described the synthetic
    # fixture even when real chaos data was on disk. On real data that number
    # is far worse, which is the whole point of the experiment.
    default_metrics, default_labels = default_data_paths()

    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(default_metrics))
    ap.add_argument("--labels", default=str(default_labels))
    ap.add_argument("--held-out", default="DISK_STRESS",
                    help="fault class to hold out of training entirely")
    args = ap.parse_args()

    df = load_labeled_windows(args.metrics, args.labels)
    features = feature_columns(df)
    print(f"source: {args.metrics}, {args.labels}")

    result = run_abstention_experiment(df, features, held_out_class=args.held_out)

    correctly_abstained_pct = result["correctly_abstained"] / result["total"]
    confidently_wrong_pct = result["confidently_wrong"] / result["total"]

    print(f"held out: {result['held_out_class']} ({result['total']} windows, never seen in training)")
    print(f"correctly abstained (confidence < {ABSTENTION_FLOOR}): "
          f"{result['correctly_abstained']} ({correctly_abstained_pct:.1%})")
    print(f"confidently wrong (confidence > {DANGEROUS_CONFIDENCE}): "
          f"{result['confidently_wrong']} ({confidently_wrong_pct:.1%})  <- the dangerous number")


if __name__ == "__main__":
    main()
