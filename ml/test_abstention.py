"""Guards for the novelty machinery in ml/abstention.py.

These are not a result. They check that the reported trade-off means what the
printed header says it means, because the whole value of the open-set numbers
is that someone can trust the direction of the comparison. A sign error in
`tradeoff()` would silently turn "rejects 99.6% of unseen windows at a 1%
known-class cost" into its opposite and still print a plausible table.
"""

import numpy as np
import pandas as pd
import pytest

from ml.abstention import (
    RECALL_POINTS,
    leaf_support,
    novelty_scores,
    novelty_split,
    tradeoff,
)


def test_tradeoff_is_perfect_when_the_two_populations_do_not_overlap():
    known = np.linspace(0.0, 1.0, 200)
    unseen = np.linspace(5.0, 6.0, 200)

    row = tradeoff(known, unseen)

    assert row["auroc"] == pytest.approx(1.0)
    for recall in RECALL_POINTS:
        # No known window is above any threshold that catches unseen windows.
        assert row[f"known_rej@{int(recall * 100)}"] == pytest.approx(0.0)
    assert row["unseen_caught@1"] == pytest.approx(1.0)


def test_tradeoff_reports_chance_when_the_statistic_carries_nothing():
    rng = np.random.default_rng(0)
    known = rng.normal(size=2000)
    unseen = rng.normal(size=2000)

    row = tradeoff(known, unseen)

    assert row["auroc"] == pytest.approx(0.5, abs=0.05)
    # At chance, catching N% of unseen costs about N% of known.
    assert row["known_rej@80"] == pytest.approx(0.8, abs=0.08)


def test_tradeoff_auroc_falls_below_half_when_the_statistic_is_inverted():
    """The headline negative result is an AUROC under 0.5, so that has to be
    representable rather than silently absolute-valued: unseen windows really
    are *more* confident than known ones, and the table must be able to say so."""
    known = np.linspace(5.0, 6.0, 200)
    unseen = np.linspace(0.0, 1.0, 200)

    assert tradeoff(known, unseen)["auroc"] == pytest.approx(0.0)


def test_every_novelty_score_is_oriented_so_higher_means_more_novel():
    rng = np.random.default_rng(1)
    X_train = np.r_[rng.normal(0, 1, size=(80, 6)), rng.normal(6, 1, size=(80, 6))]
    y_train = np.array(["A"] * 80 + ["B"] * 80)
    near = rng.normal(0, 1, size=(60, 6))
    far = rng.normal(60, 1, size=(60, 6))

    near_scores, _ = novelty_scores(X_train, y_train, near, seed=0)
    far_scores, _ = novelty_scores(X_train, y_train, far, seed=0)

    # The confidence-derived statistics are allowed to fail this - that is the
    # finding. The distance and density ones are the ones being proposed, and a
    # flipped sign in any of them would invert its whole column of the table.
    distance_based = [
        "knn1 distance", "knn5 distance", "mahalanobis pooled",
        "mahalanobis nearest class", "isolation forest",
        "local outlier factor", "one-class svm",
    ]
    for name in distance_based:
        assert far_scores[name].mean() > near_scores[name].mean(), name


def test_leaf_support_counts_training_rows_not_trees():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(2)
    X = np.r_[rng.normal(0, 1, size=(50, 4)), rng.normal(8, 1, size=(50, 4))]
    y = np.array(["A"] * 50 + ["B"] * 50)
    clf = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)

    support = leaf_support(clf, X, X)

    assert support.shape == (100,)
    # Every leaf holds at least one training row and never more than all of them.
    assert support.min() >= 1.0
    assert support.max() <= len(X)


def test_novelty_split_never_puts_the_held_out_class_in_training():
    df = pd.DataFrame({
        "label": ["A", "A", "B", "B", "C", "C"],
        "pattern": ["constant", "ramp"] * 3,
        "x_mean": range(6),
    })

    train, known, unseen = novelty_split(df, "C", novel_pattern="all")

    assert set(train["label"]) == {"A", "B"}
    assert set(known["label"]) == {"A", "B"}
    assert set(unseen["label"]) == {"C"}
    assert set(train["pattern"]) == {"constant"}
    assert set(known["pattern"]) == {"ramp"}


def test_novelty_split_can_hold_both_test_sides_on_the_same_pattern():
    """Otherwise a 'novelty' score could be reading constant-vs-ramp load
    rather than the fault, and the separation would mean nothing."""
    df = pd.DataFrame({
        "label": ["A", "A", "C", "C"],
        "pattern": ["constant", "ramp", "constant", "ramp"],
        "x_mean": range(4),
    })

    _, known, unseen = novelty_split(df, "C", novel_pattern="ramp")

    assert set(known["pattern"]) == {"ramp"}
    assert set(unseen["pattern"]) == {"ramp"}
