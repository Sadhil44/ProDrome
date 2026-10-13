"""SETUP.md section 10's ground rules, as executable checks.

Each ground rule exists to prevent a specific way of producing results that look
good and mean nothing. Stated in prose they are reminders; stated here they are
enforced. Every test below corresponds to a rule or to a bug that actually
happened in this repo -- no speculative coverage.
"""

import pandas as pd

from ml import classifier as classifier_mod
from ml.train import (
    NON_CLASSIFIABLE_LABELS,
    default_data_paths,
    load_labeled_windows,
    split_by_run,
)

METRICS_PATH = "data/samples/metrics.parquet"
LABELS_PATH = "data/samples/labels.csv"


def test_reported_metrics_and_shipped_model_use_the_same_data():
    """Regression: ml.train defaulted to data/samples/ via argparse while
    classifier.fit_and_save() used _data_paths() and preferred data/chaos/.
    The model that shipped and the accuracy we published came from different
    datasets, which is how 0.99 and 0.82 were both true at once.
    """
    assert default_data_paths() == classifier_mod._data_paths()


def test_split_is_by_run_never_by_row():
    """Ground rule 2. Consecutive windows overlap by 19 of 20 ticks, so a random
    split puts near-duplicates on both sides and produces meaningless accuracy.
    """
    df = load_labeled_windows(METRICS_PATH, LABELS_PATH)
    train, test = split_by_run(df)
    assert not train.empty and not test.empty
    overlap = set(train["run_id"].dropna()) & set(test["run_id"].dropna())
    assert overlap == set(), f"run_id(s) on both sides of the split: {sorted(overlap)}"


def test_non_classifiable_labels_never_reach_the_classifier():
    """NORMAL is filtered upstream by the detector; POD_KILL is instantaneous and
    has no diagnosis lever. Neither is a classification target.
    """
    df = load_labeled_windows(METRICS_PATH, LABELS_PATH)
    for label in NON_CLASSIFIABLE_LABELS:
        assert label not in set(df["label"])


def test_every_window_carries_a_run_id():
    df = load_labeled_windows(METRICS_PATH, LABELS_PATH)
    assert df["run_id"].notna().all()


def test_abstention_can_be_pointed_at_a_chosen_dataset():
    """Regression: ml/abstention.py called load_labeled_windows() with no
    arguments and had no CLI override, so its published "confidently wrong"
    figure could only ever describe the synthetic fixture.
    """
    from ml.abstention import run_abstention_experiment
    from ml.train import feature_columns

    df = load_labeled_windows(METRICS_PATH, LABELS_PATH)
    result = run_abstention_experiment(df, feature_columns(df), held_out_class="DISK_STRESS")
    assert result["total"] > 0
    assert result["held_out_class"] == "DISK_STRESS"
    assert 0 <= result["confidently_wrong"] <= result["total"]


def test_classifier_loads_from_any_entrypoint():
    """Regression, and a cross-seat one: save() pickled the wrapper INSTANCE, so
    the artifact recorded its class as __main__.RandomForestClassifier when
    written by `python -m ml.classifier`. Every other importer -- including the
    controller doing `from ml.classifier import classifier`, which README.md
    documents as the interface -- got AttributeError.

    This test runs under pytest, which is by definition not that __main__, so it
    fails against the old format and passes against the new one.
    """
    clf = classifier_mod.classifier
    assert hasattr(clf, "predict")
    assert clf.feature_columns, "loaded classifier has no feature columns"


def test_predict_returns_a_label_and_a_confidence():
    """The interface the controller codes against; do not change without telling cluster."""
    from ml.features import METRICS, WINDOW_SIZE

    window = pd.DataFrame({m: [1.0] * WINDOW_SIZE for m in METRICS})
    label, confidence = classifier_mod.classifier.predict(window)
    assert isinstance(label, str)
    assert 0.0 <= float(confidence) <= 1.0
