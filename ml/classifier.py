"""The real classifier: wraps the trained random forest so it matches
the interface the stub always promised.

    label, confidence = classifier.predict(window)

window: a raw table with one column per ml.features.METRICS, WINDOW_SIZE
rows - the same shape Sagar's detector and ml.dataset both consume.
Internally this runs ml.features.summarize() then the forest's
predict_proba(), so callers never touch features directly.

Ships as ml/classifier.pkl (gitignored, per SETUP.md S8 - written by
Sadhil, read by Shravan). Don't change predict()'s signature without
telling him. Run `python -m ml.classifier` to refit and re-pickle.
"""

import pickle
from pathlib import Path

import pandas as pd

from ml.features import summarize

CLASSIFIER_PATH = Path("ml/classifier.pkl")

# Prefer real chaos-run data over the Phase 0 synthetic fixture,
# whichever is actually on disk (SETUP.md S8: files, not services).
CHAOS_METRICS = Path("data/chaos/metrics.parquet")
CHAOS_LABELS = Path("data/chaos/labels.csv")
SAMPLE_METRICS = Path("data/samples/metrics.parquet")
SAMPLE_LABELS = Path("data/samples/labels.csv")


class RandomForestClassifier:
    """Thin wrapper: a fitted sklearn forest + the feature column order
    it expects, exposing predict(window) instead of predict(features)."""

    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns

    def predict(self, window):
        features = summarize(window)
        row = pd.DataFrame([features])[self.feature_columns]
        label = self.model.predict(row)[0]
        confidence = float(self.model.predict_proba(row).max())
        return label, confidence

    def save(self, path: Path = CLASSIFIER_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = CLASSIFIER_PATH) -> "RandomForestClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)


def _data_paths():
    if CHAOS_METRICS.exists() and CHAOS_LABELS.exists():
        return CHAOS_METRICS, CHAOS_LABELS
    return SAMPLE_METRICS, SAMPLE_LABELS


def fit_and_save() -> RandomForestClassifier:
    from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier

    from ml.train import feature_columns, load_labeled_windows

    metrics_path, labels_path = _data_paths()
    df = load_labeled_windows(str(metrics_path), str(labels_path))
    features = feature_columns(df)

    # Ship a model trained on everything - the constant/ramp split in
    # ml.train measures generalization (Part 4.3), it isn't data held
    # back from the artifact that actually goes live.
    model = SKRandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)
    model.fit(df[features], df["label"])

    wrapper = RandomForestClassifier(model, features)
    wrapper.save()
    print(f"fit on {metrics_path} ({len(df)} windows, classes={sorted(df['label'].unique())}), "
          f"saved to {CLASSIFIER_PATH}")
    return wrapper


def _load_or_fit() -> RandomForestClassifier:
    if CLASSIFIER_PATH.exists():
        return RandomForestClassifier.load()
    return fit_and_save()


classifier = _load_or_fit()


if __name__ == "__main__":
    fit_and_save()
