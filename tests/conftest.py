"""Shared fixtures for the contract suite.

Everything reads `data/samples/` only, because that is the sole committed data
(SETUP.md §5) and ground rule 8 says a result that is not reproducible from a
fresh clone is not a result. No cluster, no Docker, no collection handoff.

Two things live here rather than in a test module:

- the repo root on `sys.path`, so `pytest tests -q` works as well as
  `python -m pytest tests -q`. Only the latter worked before, because
  `python -m` happens to prepend the working directory; bare `pytest` does not,
  and `from ml.features import ...` then fails at collection.
- the four-stage loop (`run_loop`) and the fake controller it drives. The
  integration tests and the deliberate-failure tests need the same wiring; the
  point of both is what happens *between* stages, so the wiring is the fixture.

Nothing here imports `kubernetes` or `ml.cnn`. CI installs pandas, pyarrow,
numpy, scikit-learn and pytest and nothing else; that is deliberate
(.github/workflows/tests.yml).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from control import policy  # noqa: E402
from ml.features import METRICS, WINDOW_SIZE, windows  # noqa: E402
from ml.replay import fault_mask, train_and_replay  # noqa: E402

METRICS_PATH = REPO_ROOT / "data" / "samples" / "metrics.parquet"
LABELS_PATH = REPO_ROOT / "data" / "samples" / "labels.csv"

# SETUP.md §7, verbatim. The controller writes these; the dashboard and the eval
# harness read them by name.
DECISION_LOG_COLUMNS = [
    "ts", "workload", "detector_score", "fired", "predicted_class",
    "confidence", "top_features", "action", "result", "mode",
]

# Every action control/policy.py can emit. Pinned here rather than derived from
# POLICY on purpose: if diagnosis adds a lever, the fake controller refuses it
# and the integration test goes red, which is how cluster finds out there is a
# new action to implement -- instead of finding out from a live "unknown-action".
ACTION_VOCABULARY = frozenset({"scale_out", "rolling_restart", "alert_only", "nothing"})


# --- the data ---------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_metrics():
    return pd.read_parquet(METRICS_PATH)


@pytest.fixture(scope="session")
def sample_labels():
    return pd.read_csv(LABELS_PATH, parse_dates=["start_ts", "end_ts"])


@pytest.fixture(scope="session")
def healthy_metrics(sample_metrics, sample_labels):
    """Sample ticks with every labeled fault window removed -- ground rule #1.

    The detector must never see a fault during training, and the sample fixture
    interleaves both in one file, so "healthy" has to be carved out by mask.
    """
    return sample_metrics[~fault_mask(sample_metrics, sample_labels)]


@pytest.fixture(scope="session")
def firing_ticks(sample_metrics, sample_labels):
    """(workload, ts) of every tick the detector fired on, over the samples.

    Produced by ml.replay.train_and_replay, which is the only offline path that
    respects ground rule #1 on a single interleaved file. Session-scoped: it is
    the most expensive thing in the suite and every integration test needs it.
    """
    _, log = train_and_replay(sample_metrics, sample_labels)
    fired = log[log["fired"]]
    return {(row.workload, row.ts): float(row.score) for row in fired.itertuples()}


@pytest.fixture(scope="session")
def shipped_classifier():
    """The artifact the controller is documented to import (README.md, SETUP.md §8).

    Imported lazily because importing ml.classifier has a module-level side
    effect: it loads ml/classifier.pkl or refits it, which is most of this
    suite's runtime.
    """
    from ml.classifier import classifier

    return classifier


# --- the loop ---------------------------------------------------------------

class FakeController:
    """A controller that records instead of calling Kubernetes.

    Stands in for control/controller.py, which cannot be imported here at all:
    it does `from kubernetes import client, config` at module scope, and the
    suite must run without the kubernetes client. What this asserts is the
    contract cluster's controller has to satisfy -- an action it does not
    recognise is a loud failure, not a shrug.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.executed: list[tuple[str, str]] = []
        self.declined: list[str] = []

    def execute(self, workload: str, action: str) -> str:
        if action not in ACTION_VOCABULARY:
            raise ValueError(f"controller cannot execute action {action!r}")
        if action == "nothing":
            self.declined.append(workload)
            return "no-action"
        self.executed.append((workload, action))
        return "dry-run" if self.dry_run else "executed"


class _ShippedClassifier:
    """Adapter so the real classifier and the stubs below share one shape."""

    def __init__(self, clf):
        self._clf = clf

    def predict(self, window):
        return self._clf.predict(window)


class StubClassifier:
    """Returns a fixed (label, confidence) and counts how often it was asked.

    The count is the assertion for "the detector gates everything downstream":
    a stage that is supposed to be unreachable is only provably unreachable if
    something notices being reached.
    """

    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = confidence
        self.calls = 0

    def predict(self, window):
        self.calls += 1
        return self.label, self.confidence


def run_loop(metrics, firing, classifier, controller, mode="shadow"):
    """The four stages wired together, exactly as PRD.md describes the loop.

    detector (already replayed into `firing`) -> classifier -> policy ->
    controller. Returns one decision row per fired tick, keyed by the
    SETUP.md §7 decision-log columns, so the rows this produces are the rows a
    live controller would have to write.

    `firing` is the detector's output rather than a live Detector so a test can
    replace it with an empty gate and assert nothing downstream runs.
    """
    rows = []
    for workload in sorted(metrics["workload"].unique()):
        for window in windows(metrics, workload, WINDOW_SIZE):
            ref_ts = window["ts"].iloc[-1]
            score = firing.get((workload, ref_ts))
            if score is None:
                continue  # the detector did not fire: nothing downstream runs
            label, confidence = classifier.predict(window)
            action = policy.decide(label, confidence)
            result = controller.execute(workload, action)
            rows.append({
                "ts": ref_ts,
                "workload": workload,
                "detector_score": score,
                "fired": True,
                "predicted_class": label,
                "confidence": confidence,
                "top_features": "",
                "action": action,
                "result": result,
                "mode": mode,
            })
    return rows


@pytest.fixture(scope="session")
def decisions(sample_metrics, firing_ticks, shipped_classifier):
    """One full pass of the real pipeline over the samples, computed once."""
    controller = FakeController()
    rows = run_loop(sample_metrics, firing_ticks, _ShippedClassifier(shipped_classifier), controller)
    return rows, controller
