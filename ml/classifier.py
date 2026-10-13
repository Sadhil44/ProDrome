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

from ml.features import WINDOW_SIZE, summarize

CLASSIFIER_PATH = Path("ml/classifier.pkl")

# How many features the decision log's `top_features` cell carries. Three fits an
# operator's glance; the full 40-way decomposition is available from explain().
TOP_FEATURES_N = 3

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
        """window: WINDOW_SIZE rows by ml.features.METRICS. See the published
        contract in signal/interfaces/window-and-feature-contract.

        The length check is not pedantry. summarize() happily reduces a window of
        any length from 2 up, so a 5-row "window" used to return a confident
        label computed over 75 seconds of history by a model trained on 5
        minutes -- silently, with no indication the answer meant less than it
        appeared to. windows() never produces one, so this only ever bit a caller
        that assembles its own window: exactly what the controller does from a
        per-workload buffer at startup or just after a restart, which is also
        precisely when a wrong diagnosis is most expensive.
        """
        if len(window) != WINDOW_SIZE:
            raise ValueError(
                f"predict() needs exactly WINDOW_SIZE={WINDOW_SIZE} rows, got {len(window)}. "
                "Wait for the buffer to fill rather than scoring a short window: a short "
                "window still returns a confident label, computed over the wrong amount of history."
            )

        features = summarize(window)
        row = pd.DataFrame([features])[self.feature_columns]
        label = self.model.predict(row)[0]
        confidence = float(self.model.predict_proba(row).max())
        return label, confidence

    def explain(self, window, top_n=TOP_FEATURES_N):
        """Which features drove THIS prediction -- not which matter on average.

        Returns [(feature_name, signed_contribution), ...] for the predicted
        class, largest absolute contribution first.

        `model.feature_importances_` was the tempting shortcut and it is the
        wrong tool: it is global, so it would write the SAME three names into
        every row of the decision log and tell an operator nothing about why
        this pod was restarted at 3am. This instead decomposes the forest's own
        probability along each tree's decision path -- at every split, the change
        in class probability between parent and child is attributed to the
        feature that split there, then averaged over the trees.

        The decomposition is exact, not an approximation: tree_.value is
        normalized per node, so each node's value IS the class distribution
        there, and mean root probability plus the summed contributions
        reproduces predict_proba() to floating point. ml/test_explain.py pins
        that identity -- which is what makes these numbers auditable rather than
        merely plausible.

        It explains the MODEL, not the failure. "cpu_p95 contributed +0.31 to
        CPU_HOG" means the forest leaned on that feature, not that CPU pressure
        caused the incident. That distinction belongs in front of an operator,
        so it is stated here rather than assumed understood.
        """
        import numpy as np

        classes, contributions, _, label = self._decompose(window)
        column = classes.index(label)
        signed = contributions[column]

        order = np.argsort(-np.abs(signed))
        return [(self.feature_columns[i], float(signed[i])) for i in order[:top_n]]

    def top_features(self, window, top_n=TOP_FEATURES_N):
        """explain() flattened for the decision log's single `top_features` cell.

        Semicolon-separated, not comma: the value sits in a CSV column, and a
        comma would survive quoting but make the field miserable to grep or to
        split by hand during an incident.
        """
        return ";".join(
            f"{name}={value:+.3f}" for name, value in self.explain(window, top_n)
        )

    def _decompose(self, window):
        """Shared machinery: (classes, contributions, bias, label).

        contributions is (n_classes, n_features); bias is the mean root
        probability per class. Split out so the test can assert the identity
        bias + contributions.sum(axis=1) == predict_proba() without reaching
        into private tree internals itself.
        """
        import numpy as np

        if len(window) != WINDOW_SIZE:
            raise ValueError(
                f"explain() needs exactly WINDOW_SIZE={WINDOW_SIZE} rows, got {len(window)}. "
                "Same reason predict() refuses: a short window yields a confident "
                "attribution computed over the wrong amount of history."
            )

        features = summarize(window)
        row = pd.DataFrame([features])[self.feature_columns]
        label = self.model.predict(row)[0]
        classes = list(self.model.classes_)

        X = row.to_numpy(dtype=np.float64)
        contributions = np.zeros((len(classes), len(self.feature_columns)))
        bias = np.zeros(len(classes))

        for est in self.model.estimators_:
            tree = est.tree_
            values = tree.value[:, 0, :]
            bias += values[0]
            path = est.decision_path(X).indices
            for parent, child in zip(path[:-1], path[1:]):
                feature = tree.feature[parent]
                if feature < 0:  # a leaf splits on nothing
                    continue
                contributions[:, feature] += values[child] - values[parent]

        n = len(self.model.estimators_)
        return classes, contributions / n, bias / n, label

    def save(self, path: Path = CLASSIFIER_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pickle the PARTS, not self. Pickling the instance records its class by
        # module path, and `python -m ml.classifier` runs this module as
        # __main__ -- so the artifact came back as __main__.RandomForestClassifier
        # and could only be loaded from that same entrypoint. Every other caller,
        # including the controller doing `from ml.classifier import classifier`,
        # got AttributeError. A dict has no such dependency.
        payload = {"model": self.model, "feature_columns": self.feature_columns}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path = CLASSIFIER_PATH) -> "RandomForestClassifier":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, cls):  # artifact from before the format change
            return payload
        return cls(payload["model"], payload["feature_columns"])


def _data_paths():
    # Deferred to ml.train so the model that ships and the metrics we report
    # can never be fit on different datasets. Imported lazily: ml.train pulls
    # in sklearn, and importing this module already has the side effect of
    # loading or fitting the classifier.
    from ml.train import default_data_paths

    return default_data_paths()


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
        try:
            return RandomForestClassifier.load()
        except Exception as exc:  # noqa: BLE001 - any unreadable artifact, not one kind
            # A pickle written by an older format, a different sklearn, or a
            # half-written file should not brick every importer. The artifact is
            # gitignored and cheap to rebuild, so rebuild it.
            print(f"could not load {CLASSIFIER_PATH} ({type(exc).__name__}: {exc}); refitting")
    return fit_and_save()


classifier = _load_or_fit()


if __name__ == "__main__":
    fit_and_save()
