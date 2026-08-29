"""Abstention experiment: train on two fault types, hold a third out
entirely, and measure whether the model knows what it doesn't know.

See docs/guides/sadhil.md Part 6.1. "Confidently wrong" is the
dangerous number - those are exactly the cases where Prodrome would
take a real, irreversible action based on a fault it has never seen.

Run: python -m ml.abstention
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from control.policy import ABSTENTION_FLOOR
from ml.train import feature_columns, load_labeled_windows

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
    df = load_labeled_windows()
    features = feature_columns(df)

    result = run_abstention_experiment(df, features, held_out_class="DISK_STRESS")

    correctly_abstained_pct = result["correctly_abstained"] / result["total"]
    confidently_wrong_pct = result["confidently_wrong"] / result["total"]

    print(f"held out: {result['held_out_class']} ({result['total']} windows, never seen in training)")
    print(f"correctly abstained (confidence < {ABSTENTION_FLOOR}): "
          f"{result['correctly_abstained']} ({correctly_abstained_pct:.1%})")
    print(f"confidently wrong (confidence > {DANGEROUS_CONFIDENCE}): "
          f"{result['confidently_wrong']} ({confidently_wrong_pct:.1%})  <- the dangerous number")


if __name__ == "__main__":
    main()
