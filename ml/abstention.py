"""Abstention experiment: train on two fault types, hold a third out
entirely, and measure whether the model knows what it doesn't know.

See docs/guides/sadhil.md Part 6.1. "Confidently wrong" is the
dangerous number - those are exactly the cases where Prodrome would
take a real, irreversible action based on a fault it has never seen.

Two experiments live here.

`--mode floor` (the default) is the original: fit on everything except the
held-out class, score the held-out windows, count how many fall under
ABSTENTION_FLOOR and how many clear the policy table's 0.80 bar. It is the
number quoted everywhere, and it is terrible: 0% abstained, 71.5% confidently
wrong on an unseen DISK_STRESS.

`--mode novelty` asks why, and whether anything fixes it. A forest's
predict_proba is a normalized vote share over the classes it was shown. It has
no mass to assign to "none of these", so a confidence floor is not a novelty
detector and no choice of floor makes it one - the question is not which floor
but which statistic. This mode scores the same held-out windows with
alternatives: vote entropy and the top-two margin across trees, the training
support behind the leaves the input reaches, and unsupervised distance/density
against the training distribution. For each it reports the only trade-off that
matters:

    at a threshold that rejects X% of unseen-class windows,
    how many KNOWN-class windows does it also reject?

A detector that abstains on everything is useless, so a single operating point
proves nothing; the whole curve is printed, in both directions.

Measuring that honestly needs a known-class test set the model never saw, or
the known-class rejection rate is read off in-sample and flattered. So novelty
mode keeps ml.train's split-by-run discipline: it fits on the constant-pattern
runs of the kept classes, and scores both the ramp-pattern runs of the kept
classes (known) and the held-out class (unseen). `--novel-pattern ramp` holds
both sides on ramp runs, so any separation cannot be the load pattern rather
than the fault; `all` reproduces the full held-out population behind the
headline number instead.

Run: python -m ml.abstention
     python -m ml.abstention --mode novelty --held-out DISK_STRESS
     python -m ml.abstention --mode novelty --held-out MEMORY_LEAK --seeds 5
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from control.policy import ABSTENTION_FLOOR
from ml.train import default_data_paths, feature_columns, load_labeled_windows

DANGEROUS_CONFIDENCE = 0.80

# Rejection rates on unseen-class windows to report the known-class cost at,
# and known-class rejection budgets to report the unseen-class catch rate at.
# Both directions are printed because neither alone is the operating question:
# safety sets the first, tolerance for pointless pages sets the second.
RECALL_POINTS = (0.50, 0.70, 0.80, 0.90, 0.95)
BUDGET_POINTS = (0.01, 0.05, 0.10, 0.20)

# The two statistics `--mode novelty` found are complementary, combined by taking
# whichever is the more extreme against the known distribution. Neither works
# alone: distance is near-perfect on an unseen DISK_STRESS and at chance on an
# unseen CPU_HOG, and leaf support is the reverse. They fail on opposite classes
# because they measure different things - distance sees a fault whose metrics
# leave the training range entirely, leaf support sees one that stays inside it
# but lands where few training rows ever did.
GATE_PARTS = ("knn5 distance", "neg leaf support")


def run_abstention_experiment(df: pd.DataFrame, features, held_out_class: str, seed: int = 0) -> dict:
    train = df[df["label"] != held_out_class]
    test = df[df["label"] == held_out_class]

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)
    clf.fit(train[features], train["label"])

    confidence = clf.predict_proba(test[features]).max(axis=1)
    total = len(test)

    return {
        "held_out_class": held_out_class,
        "total": total,
        "correctly_abstained": int((confidence < ABSTENTION_FLOOR).sum()),
        "confidently_wrong": int((confidence > DANGEROUS_CONFIDENCE).sum()),
    }


# ---------------------------------------------------------------------------
# novelty scoring
# ---------------------------------------------------------------------------

def _entropy(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return -terms.sum(axis=1)


def leaf_support(clf, X_train, X_test):
    """Mean count of training samples in the leaf each test point reaches.

    A novel input can still be routed to a confident leaf, but it is more often
    routed to a thinly-populated one. This asks how much training evidence sits
    behind the prediction, which predict_proba throws away.
    """
    train_leaves = clf.apply(X_train)
    test_leaves = clf.apply(X_test)
    total = np.zeros(len(X_test))
    for tree in range(train_leaves.shape[1]):
        counts = np.bincount(train_leaves[:, tree])
        total += counts[np.clip(test_leaves[:, tree], 0, len(counts) - 1)]
    return total / train_leaves.shape[1]


def novelty_scores(X_train, y_train, X_test, seed=0):
    """Every candidate statistic for one fit. Higher always means more novel."""
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    ordered = np.sort(proba, axis=1)
    max_proba = ordered[:, -1]

    # Hard per-tree votes, which predict_proba does not expose: it averages each
    # tree's leaf class distribution, so a 200-to-0 vote split and 200 trees each
    # sitting on a 0.6/0.4 leaf both come out as 0.6 and are indistinguishable
    # from outside. If novel inputs make trees disagree, this is where it shows.
    votes = np.stack([t.predict(X_test).astype(int) for t in clf.estimators_], axis=1)
    shares = np.stack([(votes == k).sum(axis=1) for k in range(len(clf.classes_))], axis=1)
    shares = shares / votes.shape[1]
    vote_ordered = np.sort(shares, axis=1)

    scaler = StandardScaler().fit(X_train)
    Z_train = np.nan_to_num(scaler.transform(X_train))
    Z_test = np.nan_to_num(scaler.transform(X_test))

    distances, _ = NearestNeighbors(n_neighbors=min(5, len(Z_train))).fit(Z_train).kneighbors(Z_test)

    # 40 features on a few hundred training rows is a singular covariance, so
    # shrink it rather than pretend the empirical one is invertible.
    per_class = [
        LedoitWolf().fit(Z_train[y_train == c]).mahalanobis(Z_test)
        for c in np.unique(y_train)
        if (y_train == c).sum() > 5
    ]

    isolation = IsolationForest(n_estimators=200, random_state=seed).fit(Z_train)
    lof = LocalOutlierFactor(n_neighbors=min(20, len(Z_train) - 1), novelty=True).fit(Z_train)
    ocsvm = OneClassSVM(nu=0.1, gamma="scale").fit(Z_train)

    scores = {
        "1-maxprob (shipped)": 1 - max_proba,
        "proba entropy": _entropy(proba),
        "1-proba margin": 1 - (ordered[:, -1] - ordered[:, -2]),
        "tree vote entropy": _entropy(shares),
        "1-tree vote margin": 1 - (vote_ordered[:, -1] - vote_ordered[:, -2]),
        "neg leaf support": -leaf_support(clf, X_train, X_test),
        "knn1 distance": distances[:, 0],
        "knn5 distance": distances.mean(axis=1),
        "mahalanobis pooled": LedoitWolf().fit(Z_train).mahalanobis(Z_test),
        "isolation forest": -isolation.score_samples(Z_test),
        "local outlier factor": -lof.score_samples(Z_test),
        "one-class svm": -ocsvm.score_samples(Z_test),
    }
    if per_class:
        scores["mahalanobis nearest class"] = np.min(np.stack(per_class, axis=1), axis=1)
    return scores, max_proba


def tradeoff(known, unseen):
    """One statistic's full curve, from that statistic's values on both sets."""
    truth = np.r_[np.zeros(len(known)), np.ones(len(unseen))]
    row = {"auroc": float(roc_auc_score(truth, np.r_[known, unseen]))}
    for recall in RECALL_POINTS:
        threshold = np.quantile(unseen, 1 - recall)
        row[f"known_rej@{int(recall * 100)}"] = float((known >= threshold).mean())
    for budget in BUDGET_POINTS:
        threshold = np.quantile(known, 1 - budget)
        row[f"unseen_caught@{int(budget * 100)}"] = float((unseen >= threshold).mean())
    return row


def novelty_split(df, held_out_class, novel_pattern="ramp"):
    """(train, known_test, unseen_test). Split by run, never by row."""
    kept = df[df["label"] != held_out_class]
    unseen = df[df["label"] == held_out_class]
    if novel_pattern != "all":
        unseen = unseen[unseen["pattern"] == novel_pattern]
    return kept[kept["pattern"] == "constant"], kept[kept["pattern"] == "ramp"], unseen


def run_novelty_experiment(df, features, held_out_class, seeds=(0, 1, 2, 3, 4),
                           novel_pattern="ramp"):
    train, known_test, unseen_test = novelty_split(df, held_out_class, novel_pattern)
    X_train = train[features].to_numpy()
    y_train = train["label"].to_numpy()
    X_known = known_test[features].to_numpy()
    X_unseen = unseen_test[features].to_numpy()

    curves, baseline = {}, []
    for seed in seeds:
        known_scores, known_maxp = novelty_scores(X_train, y_train, X_known, seed)
        unseen_scores, unseen_maxp = novelty_scores(X_train, y_train, X_unseen, seed)
        baseline.append({
            "unseen abstained": float((unseen_maxp < ABSTENTION_FLOOR).mean()),
            "unseen confident": float((unseen_maxp > DANGEROUS_CONFIDENCE).mean()),
            "known confident": float((known_maxp > DANGEROUS_CONFIDENCE).mean()),
        })
        for name, values in known_scores.items():
            curves.setdefault(name, []).append(tradeoff(values, unseen_scores[name]))

    summary = pd.DataFrame({
        name: {key: float(np.mean([run[key] for run in runs])) for key in runs[0]}
        for name, runs in curves.items()
    }).T
    summary["auroc_sd"] = [float(np.std([r["auroc"] for r in curves[n]])) for n in summary.index]
    summary = summary.sort_values("auroc", ascending=False)
    return summary, pd.DataFrame(baseline), train, known_test, unseen_test


# ---------------------------------------------------------------------------
# the combined gate, and what it would actually cost
# ---------------------------------------------------------------------------

def percentile_rank(reference, values):
    """Where `values` sit inside `reference`, as a fraction in [0, 1].

    Puts statistics on different units (a standardized distance, a count of
    training rows) on one scale so they can be combined, using only known-class
    data as the yardstick.
    """
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def combined_gate(calibration_scores, target_scores, parts=GATE_PARTS):
    """Max over `parts` of each statistic's percentile against the calibration set."""
    stacked = np.stack([
        percentile_rank(calibration_scores[part], target_scores[part]) for part in parts
    ])
    return stacked.max(axis=0)


def calibration_split(known_test, seed=7):
    """Halve the known test set BY RUN, for calibration and for evaluation.

    Without this the gate's threshold is read off the same windows it is then
    scored against, and the known-class rejection rate comes out flattered.
    Splitting by run rather than by row for the same reason ml.train does:
    sliding windows share WINDOW_SIZE-1 of their timesteps, so a row split puts
    near-duplicates on both sides and the threshold transfers for free.
    """
    runs = np.array(sorted(known_test["run_id"].unique()))
    np.random.default_rng(seed).shuffle(runs)
    calibration = set(runs[: len(runs) // 2])
    in_calibration = known_test["run_id"].isin(calibration)
    return known_test[in_calibration], known_test[~in_calibration]


def run_gate_experiment(df, features, held_out_class, seeds=(0, 1, 2, 3, 4),
                        novel_pattern="ramp", budgets=(0.01, 0.05, 0.10)):
    """What the combined gate would do in front of the policy table.

    Reports, per known-class rejection budget: how much of the unseen class the
    gate catches, and - the number that matters - how many unseen windows still
    reach DANGEROUS_CONFIDENCE afterwards, next to what the shipped
    max-probability path lets through today.
    """
    train, known_test, unseen_test = novelty_split(df, held_out_class, novel_pattern)
    calibration, evaluation = calibration_split(known_test)

    X_train, y_train = train[features].to_numpy(), train["label"].to_numpy()
    X_calibration = calibration[features].to_numpy()
    X_evaluation = evaluation[features].to_numpy()
    X_unseen = unseen_test[features].to_numpy()

    rows = []
    for seed in seeds:
        cal_scores, _ = novelty_scores(X_train, y_train, X_calibration, seed)
        eval_scores, eval_maxp = novelty_scores(X_train, y_train, X_evaluation, seed)
        unseen_scores, unseen_maxp = novelty_scores(X_train, y_train, X_unseen, seed)

        gate_cal = combined_gate(cal_scores, cal_scores)
        gate_eval = combined_gate(cal_scores, eval_scores)
        gate_unseen = combined_gate(cal_scores, unseen_scores)

        row = {
            "auroc gate": tradeoff(gate_eval, gate_unseen)["auroc"],
            "auroc maxprob": tradeoff(eval_scores["1-maxprob (shipped)"],
                                      unseen_scores["1-maxprob (shipped)"])["auroc"],
            "unseen acted on, shipped": float((unseen_maxp > DANGEROUS_CONFIDENCE).mean()),
            "known acted on, shipped": float((eval_maxp > DANGEROUS_CONFIDENCE).mean()),
        }
        for budget in budgets:
            # The threshold comes from the calibration half only. Nothing from
            # the evaluation half or the unseen class informs it.
            threshold = np.quantile(gate_cal, 1 - budget)
            gated_unseen = gate_unseen >= threshold
            gated_known = gate_eval >= threshold
            tag = int(budget * 100)
            row[f"unseen gated@{tag}%"] = float(gated_unseen.mean())
            row[f"known gated@{tag}%"] = float(gated_known.mean())
            row[f"unseen acted on@{tag}%"] = float(
                ((unseen_maxp > DANGEROUS_CONFIDENCE) & ~gated_unseen).mean())
            row[f"known acted on@{tag}%"] = float(
                ((eval_maxp > DANGEROUS_CONFIDENCE) & ~gated_known).mean())
        rows.append(row)

    return pd.DataFrame(rows), train, calibration, evaluation, unseen_test


def _print_gate(df, features, held_out, seeds, novel_pattern):
    table, train, calibration, evaluation, unseen_test = run_gate_experiment(
        df, features, held_out, seeds, novel_pattern
    )

    print(f"held out: {held_out}   gate: max percentile over {list(GATE_PARTS)}")
    print(f"train       : {len(train)} windows, {train['run_id'].nunique()} runs (constant)")
    print(f"calibration : {len(calibration)} known windows, {calibration['run_id'].nunique()} runs "
          f"-- sets the threshold, nothing else does")
    print(f"known eval  : {len(evaluation)} windows, {evaluation['run_id'].nunique()} runs "
          f"-- disjoint runs from calibration")
    print(f"unseen      : {len(unseen_test)} windows, {unseen_test['run_id'].nunique()} runs")
    print(f"forests     : {len(seeds)} seeds\n")
    print("'acted on' = clears the 0.80 action bar AND is not gated, i.e. what the")
    print("controller would really do something about.\n")
    for column in table.columns:
        print(f"  {column:<26} {table[column].mean():.3f}   "
              f"(seeds {table[column].min():.3f}-{table[column].max():.3f})")


def _print_floor(df, features, held_out, seeds=(0,)):
    results = [run_abstention_experiment(df, features, held_out, seed) for seed in seeds]
    total = results[0]["total"]
    abstained = np.array([r["correctly_abstained"] for r in results]) / total
    wrong = np.array([r["confidently_wrong"] for r in results]) / total

    print(f"held out: {held_out} ({total} windows, never seen in training)")
    print(f"correctly abstained (confidence < {ABSTENTION_FLOOR}): "
          f"{results[0]['correctly_abstained']} ({abstained[0]:.1%})")
    print(f"confidently wrong (confidence > {DANGEROUS_CONFIDENCE}): "
          f"{results[0]['confidently_wrong']} ({wrong[0]:.1%})  <- the dangerous number")

    if len(seeds) > 1:
        # The single-seed version of this number is not always a number. On
        # DISK_STRESS it is stable to +-0.02 and can be quoted; on MEMORY_LEAK's
        # 169 windows it swings from 0.18 to 0.74 on the forest seed alone, so a
        # point estimate there describes random_state, not the model. The 0%
        # abstention rate, by contrast, holds on every class and every seed -
        # that is the finding.
        print(f"\nacross {len(seeds)} seeds:")
        print(f"  correctly abstained  {abstained.mean():.3f} +- {abstained.std():.3f}   "
              f"(range {abstained.min():.3f}-{abstained.max():.3f})")
        print(f"  confidently wrong    {wrong.mean():.3f} +- {wrong.std():.3f}   "
              f"(range {wrong.min():.3f}-{wrong.max():.3f})")
        if wrong.max() - wrong.min() > 0.10:
            print("  ^ that spread is the forest seed alone. Do not quote a point estimate.")


def _print_novelty(df, features, held_out, seeds, novel_pattern):
    summary, baseline, train, known_test, unseen_test = run_novelty_experiment(
        df, features, held_out, seeds, novel_pattern
    )

    print(f"held out: {held_out}")
    print(f"train       : {len(train)} windows, {train['run_id'].nunique()} runs (constant), "
          f"{train['label'].value_counts().to_dict()}")
    print(f"known test  : {len(known_test)} windows, {known_test['run_id'].nunique()} runs (ramp), "
          f"{known_test['label'].value_counts().to_dict()}")
    print(f"unseen test : {len(unseen_test)} windows, {unseen_test['run_id'].nunique()} runs "
          f"({novel_pattern} pattern)")
    print(f"forests     : {len(seeds)} seeds per statistic\n")

    print(f"max-probability, the statistic control/policy.py reads "
          f"(floor {ABSTENTION_FLOOR}, action bar {DANGEROUS_CONFIDENCE}):")
    for column in baseline.columns:
        print(f"  {column:<20} {baseline[column].mean():.3f}   "
              f"(seeds {baseline[column].min():.3f}-{baseline[column].max():.3f})")

    print("\ntrade-off for every statistic; higher score always means more novel.")
    print("  known_rej@N      share of KNOWN windows also rejected at the threshold that")
    print("                   rejects N% of unseen-class windows        (lower is better)")
    print("  unseen_caught@N  share of UNSEEN windows rejected at the threshold that")
    print("                   rejects N% of known windows               (higher is better)")
    pd.set_option("display.width", 240)
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))


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
    ap.add_argument("--mode", choices=("floor", "novelty", "gate"), default="floor",
                    help="floor: the shipped confidence-floor numbers. "
                         "novelty: alternative statistics and their trade-off curves. "
                         "gate: what the combined statistic would cost in front of the "
                         "policy table, with its threshold calibrated on held-out runs.")
    ap.add_argument("--seeds", type=int, default=None,
                    help="forests to average over. Defaults to 5 for novelty and gate, and to 1 "
                         "for floor so the published headline still prints verbatim; pass a "
                         "number there to see how much of it is the seed. A few points of "
                         "difference on a few hundred windows is well inside forest noise.")
    ap.add_argument("--novel-pattern", choices=("ramp", "all"), default="ramp",
                    help="ramp: match the known-class test runs, so separation cannot be the "
                         "load pattern. all: the full held-out population.")
    args = ap.parse_args()

    df = load_labeled_windows(args.metrics, args.labels)
    features = feature_columns(df)
    print(f"source: {args.metrics}, {args.labels}")

    if args.mode == "floor":
        _print_floor(df, features, args.held_out, tuple(range(args.seeds or 1)))
        return

    warnings.filterwarnings("ignore")
    seeds = tuple(range(args.seeds or 5))
    if args.mode == "gate":
        _print_gate(df, features, args.held_out, seeds, args.novel_pattern)
    else:
        _print_novelty(df, features, args.held_out, seeds, args.novel_pattern)


if __name__ == "__main__":
    main()
