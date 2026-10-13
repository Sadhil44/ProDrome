"""Leave-one-workload-out: is the classifier diagnosing faults or recognising services?

THE QUESTION. The shipped headline (0.92 accuracy / 0.82 macro F1) is measured
with a split BY RUN: fit on `constant`-pattern runs, score on `ramp`-pattern
runs. Every run on both sides comes from the same three workloads -- nginx,
postgres, redis. So the model has seen each service's baseline before it is
asked about it. A model that had learned "this is what postgres looks like when
it is sick" rather than "this is what a memory leak looks like" would score
exactly the same way.

That is not hypothetical here. The novelty gate built earlier this session
looked excellent under a run split and then collapsed under a workload holdout
(gate AUROC nginx 0.998 / postgres 0.877 / redis 0.339 -- below chance), because
it was a service-identity detector wearing a novelty detector's name. The
classifier has the identical shape of exposure and had never been tested for it.

THE ANSWER (7 seeds, data/chaos, shipped feature set / model / hyperparameters).
Partly services. It is NOT the clean collapse the novelty gate was, and it is
not clean transfer either -- it depends almost entirely on which workload you
hold out:

  macro F1, in-distribution -> workload held out
    nginx      0.834 +/- 0.005  ->  0.788 +/- 0.055   (-0.046, INSIDE seed noise)
    redis      0.843 +/- 0.027  ->  0.759 +/- 0.021   (-0.083)
    postgres   0.765 +/- 0.020  ->  0.484 +/- 0.032   (-0.281)

DISK_STRESS transfers (0.97 -> 0.88..0.99 across the three arms; it is the only
class that is never meaningfully hurt). CPU_HOG half-transfers (-0.03, -0.08,
-0.28). MEMORY_LEAK is the class that breaks: -0.116 (noise), -0.186, -0.476.
On the postgres arm it is 0.038 +/- 0.088, and 5 of the 7 seeds score it exactly
0.000 -- the model never once correctly names a memory leak on a service it was
not trained on, and the failure is a collapse onto the majority class, not a
swap: 96.3% of true MEMORY_LEAK windows are called DISK_STRESS. Accuracy on
that arm is still 0.761, which is why accuracy alone would have hidden all of
this. The direction of the error pre-exists the holdout (the in-distribution
postgres arm already sends 33.7% of MEMORY_LEAK to DISK_STRESS); holding the
workload out completes it.

So: the headline 0.92 / 0.82 is not pure service recognition, but it is not
portable either, and the amount of portability it has was never measured and
varies by a factor of six across the three services we happen to own.

WHAT THIS DOES NOT ESTABLISH. It cannot separate "learned the service" from
"learned the fault injector's footprint on that service" -- both are workload-
specific and both break identically here. That distinction matters on this
project: ml/ablation.py showed CPU_HOG scores BETTER with every cpu feature
deleted (F1 0.883 -> 0.936), i.e. at least one class is already being recognised
by injector side effects rather than by the pressure it is named after. It also
cannot tell us how a FOURTH, unseen service would behave: n=3 workloads means
each arm is a single draw, and the three arms overlap in their training data so
they are not independent replicates. And it cannot separate "the workload is
unseen" from "dropping a workload shrank the training set", which is a live
confound for exactly the class that breaks: MEMORY_LEAK has 44 constant-pattern
training windows in total, so every holdout arm trains it on 27-33. Both
explanations predict what we observe, and this data cannot choose between them.

Nor is MEMORY_LEAK robust on its own terms. The effective sample size is RUNS,
not windows, and MEMORY_LEAK has 3-5 runs per workload per pattern -- its ~42
test windows are 5 runs' worth of near-duplicate sliding windows. Macro F1
weights that class equally with DISK_STRESS's 15 runs. Treat the MEMORY_LEAK
column, and so the macro column, as fragile on every arm.

METHOD. Every arm holds the TEST ROWS FIXED and changes only the training set,
so the delta is attributable to the holdout and nothing else:

  in-dist  train = constant runs, ALL three workloads   test = ramp runs of W
  holdout  train = constant runs, the OTHER two only    test = ramp runs of W

Both keep the shipped constant->ramp pattern split, so the pattern shift is
present on both sides of the delta and cancels out of it. The split is by run
throughout; sliding windows share WINDOW_SIZE-1 timesteps, so a row split would
train on near-duplicates of the test set (ml/CLAUDE.md). The feature set, model
and hyperparameters are exactly as shipped -- this measures the shipped model's
exposure, it is not a tuning harness.

Seeds vary only the forest's random_state; the data split is deterministic.
Everything is reported as mean +/- SD over seeds, and deltas are paired by seed,
because this project has already published a seed-noise artifact as a finding
once (a "8.9% confidently wrong" figure that was really 0.344 +/- 0.214).

Run (from the worktree; data/ is gitignored and lives in the main checkout):

    python -m ml.holdout --root /path/to/ProDrome/data/chaos
    python -m ml.holdout --root /path/to/ProDrome/data/chaos --seeds 7
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from ml.train import default_data_paths, feature_columns, load_labeled_windows

# Shipped configuration, matching ml/classifier.py fit_and_save() the same way
# ml/ablation.py does. Nothing here is a knob.
N_ESTIMATORS = 200
CLASS_WEIGHT = "balanced"
DEFAULT_SEEDS = 7

TRAIN_PATTERN = "constant"
TEST_PATTERN = "ramp"

# Below this many test windows of a class, its F1 is one or two runs' worth of
# near-duplicate windows, and the macro average inherits that noise.
THIN_SUPPORT = 60

BASELINE_ARM = "ALL (baseline)"


def resolve_paths(root=None, metrics_path=None, labels_path=None):
    """(metrics, labels). --root names a data dir; otherwise ml.train decides."""
    if root is not None:
        root = Path(root)
        return str(root / "metrics.parquet"), str(root / "labels.csv")
    default_metrics, default_labels = default_data_paths()
    return str(metrics_path or default_metrics), str(labels_path or default_labels)


def score(train, test, features, labels, seed):
    """Fit one forest, score it, free it. Returns (metrics dict, confusion matrix)."""
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight=CLASS_WEIGHT, random_state=seed
    )
    model.fit(train[features], train["label"])
    predicted = model.predict(test[features])
    del model
    gc.collect()  # tight memory on this box: never hold two fitted forests at once

    row = dict(zip(labels, f1_score(test["label"], predicted, average=None, labels=labels)))
    row["MACRO"] = f1_score(test["label"], predicted, average="macro")
    row["ACCURACY"] = float((predicted == test["label"]).mean())
    return row, confusion_matrix(test["label"], predicted, labels=labels)


def build_arms(train_all, test_all, workloads):
    """[(name, test, in_dist_train, holdout_train_or_None)], evaluated in order."""
    arms = [(BASELINE_ARM, test_all, train_all, None)]
    for held_out in workloads:
        arms.append((
            held_out,
            test_all[test_all["workload"] == held_out],
            train_all,
            train_all[train_all["workload"] != held_out],
        ))
    return arms


def run(root=None, metrics_path=None, labels_path=None, seeds=DEFAULT_SEEDS):
    metrics_path, labels_path = resolve_paths(root, metrics_path, labels_path)
    df = load_labeled_windows(metrics_path, labels_path)

    features = feature_columns(df)
    labels = sorted(df["label"].unique())
    workloads = sorted(df["workload"].unique())
    train_all = df[df["pattern"] == TRAIN_PATTERN]
    test_all = df[df["pattern"] == TEST_PATTERN]

    print(f"data     : {metrics_path}")
    print(f"windows  : {len(df)} ({len(train_all)} {TRAIN_PATTERN} train "
          f"/ {len(test_all)} {TEST_PATTERN} test)")
    print(f"features : {len(features)} (shipped set, unchanged)")
    print(f"model    : RandomForest(n_estimators={N_ESTIMATORS}, class_weight={CLASS_WEIGHT!r})")
    print(f"seeds    : {list(range(seeds))}")
    print()
    print("TRAIN support by workload (constant runs) -- the ceiling on every holdout arm:")
    print(pd.crosstab(train_all["workload"], train_all["label"]).to_string())
    print()
    print("TEST support by workload (ramp runs):")
    print(pd.crosstab(test_all["workload"], test_all["label"]).to_string())
    print()

    columns = labels + ["MACRO", "ACCURACY"]
    arms = build_arms(train_all, test_all, workloads)
    results = {}     # (arm, kind) -> DataFrame, one row per seed
    confusions = {}  # (arm, kind) -> confusion matrix summed over seeds

    for arm, test, in_dist_train, holdout_train in arms:
        for kind, train in (("in-dist", in_dist_train), ("holdout", holdout_train)):
            if train is None:
                continue  # the all-workload baseline has no workload to hold out
            rows, total = [], None
            for seed in range(seeds):
                row, matrix = score(train, test, features, labels, seed)
                rows.append(row)
                total = matrix if total is None else total + matrix
            results[(arm, kind)] = pd.DataFrame(rows)
            confusions[(arm, kind)] = total

    report(results, confusions, arms, labels, columns, test_all, seeds)
    return results, confusions, labels


def _cell(series):
    return f"{series.mean():.3f}+/-{series.std(ddof=1):.3f}"


def report(results, confusions, arms, labels, columns, test_all, seeds):
    width = max(16, max(len(c) for c in columns) + 2)
    head = (f"{'held out':<16}{'train':<9}{'test n':>7}  "
            + "".join(f"{c[:width - 2]:>{width}}" for c in columns))
    rule = "=" * len(head)

    print(rule)
    print(f"MEAN +/- SD OVER {seeds} SEEDS   (per-class F1, then macro F1 and accuracy)")
    print(rule)
    print(head)
    print("-" * len(head))

    for arm, test, _, _ in arms:
        kinds = [k for k in ("in-dist", "holdout") if (arm, k) in results]
        for kind in kinds:
            frame = results[(arm, kind)]
            print(f"{arm:<16}{kind:<9}{len(test):>7}  "
                  + "".join(f"{_cell(frame[c]):>{width}}" for c in columns))
        if len(kinds) == 2:
            in_dist, holdout = results[(arm, "in-dist")], results[(arm, "holdout")]
            cells = []
            for column in columns:
                # Paired by seed: the SD of the per-seed difference is the honest
                # error bar on a delta, not the SD of either arm on its own.
                diff = holdout[column] - in_dist[column]
                noise = "*" if abs(diff.mean()) < diff.std(ddof=1) else ""
                cells.append(f"{diff.mean():+.3f}{noise}".rjust(width))
            print(f"{'':<16}{'DELTA':<9}{'':>7}  " + "".join(cells))
        print("-" * len(head))

    print("* = |mean delta| is smaller than the SD of the per-seed differences: not")
    print("    distinguishable from seed noise, do not report it as an effect.")
    print()

    thin = test_all.groupby(["workload", "label"]).size()
    thin = thin[thin < THIN_SUPPORT]
    if len(thin):
        print(f"THIN TEST SUPPORT (< {THIN_SUPPORT} windows) -- macro F1 weights these equally")
        print("with the fat classes, so an arm listed here has a fragile macro number:")
        for (workload, label), count in thin.items():
            print(f"  {workload:<10} {label:<13} {count:>4} windows")
        print("  Windows inside a run share WINDOW_SIZE-1 timesteps, so the effective")
        print("  sample size is RUNS, not windows -- these are a few runs each.")
        print()

    # Worst arm by macro-F1 delta, with its confusion structure: collapsing onto
    # the majority class and systematically swapping two classes are different
    # failures that want different fixes.
    per_arm = {
        arm: results[(arm, "holdout")]["MACRO"].mean() - results[(arm, "in-dist")]["MACRO"].mean()
        for arm, _, _, _ in arms if (arm, "holdout") in results
    }
    worst = min(per_arm, key=per_arm.get)
    print(rule)
    print(f"CONFUSION STRUCTURE, WORST ARM: {worst} held out (macro F1 delta {per_arm[worst]:+.3f})")
    print(rule)
    for kind in ("in-dist", "holdout"):
        frame = pd.DataFrame(confusions[(worst, kind)], index=labels, columns=labels)
        fractions = frame.div(frame.sum(axis=1).replace(0, np.nan), axis=0)
        print(f"\n{kind}: rows = actual, cols = predicted, summed over {seeds} seeds")
        print(frame.to_string())
        print("row-normalised (what it says instead, per actual class):")
        print(fractions.round(3).to_string())
    print()

    print("ALL ARMS, macro F1 in-dist -> holdout:")
    for arm, _, _, _ in arms:
        if (arm, "holdout") not in results:
            continue
        before = results[(arm, "in-dist")]["MACRO"]
        after = results[(arm, "holdout")]["MACRO"]
        print(f"  {arm:<10} {before.mean():.3f}+/-{before.std(ddof=1):.3f}"
              f"  ->  {after.mean():.3f}+/-{after.std(ddof=1):.3f}"
              f"   ({after.mean() - before.mean():+.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=None,
                        help="data dir holding metrics.parquet and labels.csv; data/ is "
                             "gitignored, so from a worktree point this at the main "
                             "checkout's data/chaos")
    parser.add_argument("--metrics", default=None, help="path to metrics.parquet")
    parser.add_argument("--labels", default=None, help="path to labels.csv")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"number of seeds, 0..n-1 (default: {DEFAULT_SEEDS}); a "
                             "single-seed number is not a result on this project")
    args = parser.parse_args()
    run(args.root, args.metrics, args.labels, args.seeds)


if __name__ == "__main__":
    main()
