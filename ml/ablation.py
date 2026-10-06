"""Feature ablation: does a fault class carry signal on its own metric?

Companion to ml/abstention.py -- a diagnosis tool that asks a safety question
about the shipped model rather than reporting an accuracy number.

The attribution work (ml/classifier.py explain(), 2026-10-03) showed that
CPU_HOG is predicted at confidence 1.00 with only 6.0% of its attribution mass
on cpu_cores and ~80% on memory and filesystem features. That says what the
model USES. It does not say whether usable CPU signal exists at all, and the two
have very different consequences:

  - no usable signal        -> CPU_HOG detection is an artifact of our injector's
                               side effects, and will not transfer to real CPU
                               pressure from genuine application load
  - signal exists, unused   -> the model is taking an easier shortcut, which is
                               a fixable modelling problem

This settles it by ablation: same train/test split as ml.train (fit on
constant-pattern runs, test on ramp-pattern runs -- split by run, never by row,
since sliding windows share WINDOW_SIZE-1 of their timesteps), varying only the
feature set.

Run: python -m ml.ablation [--metric cpu_cores] [--metrics PATH] [--labels PATH]
"""

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from ml.train import default_data_paths, feature_columns, load_labeled_windows

# Matches the shipped model in ml/classifier.py fit_and_save(), so the "all"
# row below is the configuration that actually goes live rather than a
# differently-tuned stand-in.
N_ESTIMATORS = 200
RANDOM_STATE = 0


def ablate(metric="cpu_cores", metrics_path=None, labels_path=None):
    """Compare all features vs only `metric`'s vs everything except `metric`'s."""
    if metrics_path is None or labels_path is None:
        default_metrics, default_labels = default_data_paths()
        metrics_path = metrics_path or str(default_metrics)
        labels_path = labels_path or str(default_labels)

    df = load_labeled_windows(metrics_path, labels_path)
    train = df[df["pattern"] == "constant"]
    test = df[df["pattern"] == "ramp"]

    all_features = feature_columns(df)
    owned = [c for c in all_features if c.startswith(f"{metric}_")]
    if not owned:
        raise SystemExit(f"no features for metric {metric!r}; have: {sorted(all_features)[:8]}...")
    others = [c for c in all_features if not c.startswith(f"{metric}_")]

    print(f"data    : {metrics_path}")
    print(f"windows : {len(train)} train (constant) / {len(test)} test (ramp)")
    print(f"features: {len(all_features)} total, {len(owned)} {metric}, {len(others)} other")
    print(f"test    : {test['label'].value_counts().to_dict()}")
    print()

    sets = {"all": all_features, f"{metric}_only": owned, f"no_{metric}": others}
    labels = sorted(test["label"].unique())
    results = {}

    for name, columns in sets.items():
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=RANDOM_STATE
        )
        model.fit(train[columns], train["label"])
        predicted = model.predict(test[columns])
        results[name] = {
            **dict(zip(labels, f1_score(test["label"], predicted, average=None, labels=labels))),
            "MACRO": f1_score(test["label"], predicted, average="macro"),
            "ACCURACY": float((predicted == test["label"]).mean()),
        }

    header = labels + ["MACRO", "ACCURACY"]
    width = max(13, max(len(h) for h in header) + 2)
    print("=" * (14 + width * len(header)))
    print(f"{'feature set':<14}" + "".join(f"{h[:width-2]:>{width}}" for h in header))
    print("=" * (14 + width * len(header)))
    for name in sets:
        print(f"{name:<14}" + "".join(f"{results[name][h]:>{width}.3f}" for h in header))
    print()

    return results, sets, train, test, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metric", default="cpu_cores",
                        help="metric whose features to ablate (default: cpu_cores)")
    parser.add_argument("--metrics", default=None, help="path to metrics.parquet")
    parser.add_argument("--labels", default=None, help="path to labels.csv")
    args = parser.parse_args()

    results, sets, train, test, _ = ablate(args.metric, args.metrics, args.labels)

    # The headline comparison, stated rather than left for the reader to spot.
    metric = args.metric
    target = {"cpu_cores": "CPU_HOG", "mem_bytes": "MEMORY_LEAK", "mem_pct": "MEMORY_LEAK",
              "fs_writes": "DISK_STRESS", "fs_reads": "DISK_STRESS"}.get(metric)
    if target and target in results["all"]:
        base = results["all"][target]
        alone = results[f"{metric}_only"][target]
        without = results[f"no_{metric}"][target]
        print(f"{target} vs its own metric {metric}:")
        print(f"  all features     F1 {base:.3f}")
        print(f"  {metric} only    F1 {alone:.3f}   ({alone - base:+.3f})")
        print(f"  WITHOUT {metric}  F1 {without:.3f}   ({without - base:+.3f})")
        print()
        if without >= base:
            print(f"  READ THIS: removing every {metric} feature does NOT hurt {target} -- it")
            print(f"  helps. The class named after {metric} is being recognised by something")
            print(f"  else entirely, and the real {metric} measurements are, on this data, noise")
            print(f"  that degrades the model. A {target} result from this model is not evidence")
            print(f"  that the system detects {metric} pressure.")

    columns = sets[f"{args.metric}_only"]
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=RANDOM_STATE
    ).fit(train[columns], train["label"])
    print(f"Full report, {args.metric}_only:")
    print(classification_report(test["label"], model.predict(test[columns]), zero_division=0))


if __name__ == "__main__":
    main()
