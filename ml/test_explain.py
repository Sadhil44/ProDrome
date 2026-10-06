"""Does the decision log's `top_features` cell mean anything?

The end-to-end run on 2026-10-03 wrote 3276 decision rows with top_features NaN
on every one. The column matched SETUP.md S7 and carried no information: an
operator asking "why did this restart my pod at 3am" got a class, a confidence
and a blank. These tests exist so the replacement is auditable rather than
merely plausible.

The load-bearing test is the identity one. A feature-attribution number is easy
to produce and almost impossible to eyeball for correctness -- a sign error or a
missed tree would still yield a ranked list that looked perfectly reasonable. So
rather than assert that the answers seem sensible, these pin the arithmetic:
mean root probability plus the summed per-split contributions must reproduce
predict_proba() exactly. If that holds, the decomposition is the forest's own
probability and not an invented ranking.

Runs from a fresh clone: fits a small forest on synthetic windows, no artifact,
no cluster, no data handoff.
"""

import numpy as np
import pandas as pd
import pytest

from ml.classifier import RandomForestClassifier, TOP_FEATURES_N
from ml.features import METRICS, WINDOW_SIZE, summarize


def _window(rng, scale=1.0, drift=0.0):
    """One WINDOW_SIZE window of plausible metric values."""
    ticks = np.arange(WINDOW_SIZE)
    return pd.DataFrame({
        m: rng.normal(10.0 * scale, 1.0, WINDOW_SIZE) + drift * ticks
        for m in METRICS
    })


@pytest.fixture(scope="module")
def fitted():
    """A real sklearn forest over three separable synthetic classes.

    Separable on purpose: the identity under test is arithmetic and holds for any
    forest, but a model that cannot tell the classes apart would make the
    ranking tests vacuous rather than wrong.
    """
    from sklearn.ensemble import RandomForestClassifier as SK

    rng = np.random.default_rng(0)
    rows, labels = [], []
    for label, (scale, drift) in {
        "NORMAL": (1.0, 0.0),
        "CPU_HOG": (6.0, 0.0),
        "MEMORY_LEAK": (1.0, 2.0),
    }.items():
        for _ in range(25):
            rows.append(summarize(_window(rng, scale, drift)))
            labels.append(label)

    frame = pd.DataFrame(rows)
    columns = list(frame.columns)
    model = SK(n_estimators=25, random_state=0).fit(frame[columns], labels)
    return RandomForestClassifier(model, columns)


@pytest.fixture
def windows():
    rng = np.random.default_rng(7)
    return {
        "NORMAL": _window(rng, 1.0, 0.0),
        "CPU_HOG": _window(rng, 6.0, 0.0),
        "MEMORY_LEAK": _window(rng, 1.0, 2.0),
    }


# --- the one that matters ----------------------------------------------------

def test_contributions_reconstruct_predict_proba_exactly(fitted, windows):
    """bias + sum(contributions) == predict_proba(), to floating point.

    This is what separates a real decomposition from a ranked list of guesses.
    tree_.value is normalized per node in sklearn, so each node's value IS the
    class distribution there; telescoping parent-to-child differences along a
    decision path therefore sums to leaf minus root, and averaging over trees is
    exactly what the forest does to produce predict_proba. If this identity ever
    breaks -- a sign error, a skipped tree, a changed sklearn internal -- every
    number in the top_features column becomes fiction, silently.
    """
    for window in windows.values():
        classes, contributions, bias, _ = fitted._decompose(window)

        features = summarize(window)
        row = pd.DataFrame([features])[fitted.feature_columns]
        expected = fitted.model.predict_proba(row)[0]

        reconstructed = bias + contributions.sum(axis=1)

        assert list(classes) == list(fitted.model.classes_)
        np.testing.assert_allclose(reconstructed, expected, atol=1e-9)


def test_contributions_are_signed_both_ways(fitted, windows):
    """Attribution has to be able to say "this argued AGAINST the label".

    An implementation that returned absolute values, or importances, would pass
    a ranking test and lose the direction -- which is the half an operator needs
    to tell "memory was climbing" from "memory was flat, which ruled out a leak".
    """
    _, contributions, _, _ = fitted._decompose(windows["CPU_HOG"])
    assert (contributions > 0).any(), "nothing ever pushed toward any class"
    assert (contributions < 0).any(), "no negative contribution; sign was discarded"


# --- the operator-facing surface ---------------------------------------------

def test_explain_ranks_by_absolute_contribution_and_is_capped(fitted, windows):
    for window in windows.values():
        explained = fitted.explain(window)
        assert len(explained) == TOP_FEATURES_N

        names = [name for name, _ in explained]
        assert names == list(dict.fromkeys(names)), \
            "a feature appeared twice; contributions were not aggregated per feature"

        magnitudes = [abs(value) for _, value in explained]
        assert magnitudes == sorted(magnitudes, reverse=True)

        for name in names:
            assert name in fitted.feature_columns


def test_explain_names_real_features_not_column_indices(fitted, windows):
    """The cell is read by a human during an incident. "f17" is not an answer."""
    explained = fitted.explain(windows["MEMORY_LEAK"])
    for name, _ in explained:
        metric, _, statistic = name.rpartition("_")
        assert metric in METRICS, f"{name!r} is not <metric>_<stat>"
        assert statistic in {"mean", "slope", "std", "max", "last"}


def test_top_features_survives_a_csv_round_trip(fitted, windows, tmp_path):
    """The value lands in one CSV cell next to nine others.

    Uses no comma deliberately -- csv would quote it correctly, but the field
    becomes miserable to split by hand during an incident, which is the only
    time anyone reads it.
    """
    import csv

    cell = fitted.top_features(windows["CPU_HOG"])
    assert "," not in cell
    assert cell.count(";") == TOP_FEATURES_N - 1

    path = tmp_path / "log.csv"
    with path.open("w", newline="") as f:
        csv.writer(f).writerows([["top_features", "action"], [cell, "rolling_restart"]])

    back = pd.read_csv(path)
    assert back["top_features"].iloc[0] == cell

    for part in cell.split(";"):
        name, _, value = part.partition("=")
        assert name in fitted.feature_columns
        assert value[0] in "+-", "sign is the half an operator needs; keep it explicit"
        float(value)


def test_explain_refuses_a_short_window_like_predict_does(fitted, windows):
    """Same defect, same refusal. A short window yields a confident attribution
    over the wrong amount of history -- worse than predict()'s version of the
    bug, because a plausible-looking reason is more persuasive than a bare label.
    """
    short = windows["NORMAL"].head(5)
    with pytest.raises(ValueError, match=f"WINDOW_SIZE={WINDOW_SIZE}"):
        fitted.explain(short)
    with pytest.raises(ValueError):
        fitted.top_features(short)


def test_attribution_is_not_the_same_list_for_every_input(fitted, windows):
    """The whole reason feature_importances_ was rejected.

    A global ranking would write identical names into all 3276 rows and pass
    every other test in this file. Two genuinely different fault shapes must
    produce different explanations, or the column is decoration.
    """
    cpu = fitted.explain(windows["CPU_HOG"])
    leak = fitted.explain(windows["MEMORY_LEAK"])

    differs = (
        [n for n, _ in cpu] != [n for n, _ in leak]
        or [round(v, 6) for _, v in cpu] != [round(v, 6) for _, v in leak]
    )
    assert differs, "explanation did not vary with input; a global ranking in disguise"
