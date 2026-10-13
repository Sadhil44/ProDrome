"""The three cross-pair schemas from SETUP.md section 7, as executable checks.

These are the only things that need agreement across pairs, and the forum rule
is that changing one is never quiet. A test is the cheapest way to make "never
quiet" true: rename a column and this goes red in CI instead of surfacing three
days later as someone else's confusing KeyError.

Everything here runs against data/samples/, which is committed, so it works
from a fresh clone with no cluster and no collection run (ground rule 8).
"""

import pandas as pd
import pytest

METRICS_PATH = "data/samples/metrics.parquet"
LABELS_PATH = "data/samples/labels.csv"

# SETUP.md section 7, verbatim.
METRICS_COLUMNS = [
    "ts", "workload", "cpu_cores", "mem_bytes", "mem_pct",
    "net_rx", "net_tx", "fs_reads", "fs_writes", "restarts",
]
LABELS_COLUMNS = ["start_ts", "end_ts", "workload", "fault_type", "pattern", "run_id"]
DECISION_LOG_COLUMNS = [
    "ts", "workload", "detector_score", "fired", "predicted_class",
    "confidence", "top_features", "action", "result", "mode",
]


@pytest.fixture(scope="module")
def metrics():
    return pd.read_parquet(METRICS_PATH)


@pytest.fixture(scope="module")
def labels():
    return pd.read_csv(LABELS_PATH, parse_dates=["start_ts", "end_ts"])


def test_metrics_table_has_the_agreed_columns(metrics):
    assert list(metrics.columns) == METRICS_COLUMNS


def test_labels_table_has_the_agreed_columns(labels):
    assert list(labels.columns) == LABELS_COLUMNS


def test_workload_is_a_stable_name_not_a_pod_name(metrics):
    """`redis`, not `redis-7d9f8b-x2k1` -- pod-instance names do not join across runs."""
    for name in metrics["workload"].unique():
        parts = str(name).split("-")
        assert len(parts) == 1, (
            f"workload {name!r} looks like a pod name; SETUP.md section 7 requires a stable name"
        )


def test_run_id_is_never_null(labels):
    """Splits are by run. A null run_id makes a run unusable, not merely awkward."""
    assert labels["run_id"].notna().all()


def test_fault_intervals_are_ordered(labels):
    assert (labels["end_ts"] >= labels["start_ts"]).all()


def test_both_fault_patterns_are_present(labels):
    """Constant AND ramp. If every fault jumps to full intensity the classifier
    never sees what a slide toward failure looks like -- the case the project exists for."""
    assert {"constant", "ramp"} <= set(labels["pattern"].unique())


def test_metrics_are_numeric(metrics):
    for col in METRICS_COLUMNS:
        if col in ("ts", "workload"):
            continue
        assert pd.api.types.is_numeric_dtype(metrics[col]), f"{col} is not numeric"


def test_decision_log_columns_are_pinned():
    """cluster owns the writer; the dashboard and eval harness both read it by name.

    This asserts the agreed column list itself, so a change to the constant is a
    visible diff that has to be announced rather than discovered downstream.
    """
    assert DECISION_LOG_COLUMNS == [
        "ts", "workload", "detector_score", "fired", "predicted_class",
        "confidence", "top_features", "action", "result", "mode",
    ]
