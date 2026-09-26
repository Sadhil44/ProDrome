"""The published code interfaces, as executable checks.

signal owns the window/feature contract and the detector interface; diagnosis
owns classifier.predict and policy.decide. Both are published on the forum as
frozen, and both halves of the pipeline have to agree on them or the controller
feeds two models two different things.
"""

import inspect

import pandas as pd
import pytest

from control import policy
from ml.features import METRICS, WINDOW_SIZE, summarize, windows

# signal/interfaces/window-and-feature-contract, verbatim and in this order.
CONTRACT_METRICS = [
    "cpu_cores", "mem_bytes", "mem_pct", "net_rx",
    "net_tx", "fs_reads", "fs_writes", "restarts",
]
CONTRACT_SUFFIXES = ("_mean", "_slope", "_std", "_max", "_last")


def test_metric_order_matches_the_published_contract():
    """Frozen order: anything indexing by position depends on it."""
    assert METRICS == CONTRACT_METRICS


def test_window_size_matches_the_published_contract():
    assert WINDOW_SIZE == 20  # 20 ticks x 15s = 5 minutes


def test_summarize_emits_exactly_forty_features():
    frame = pd.DataFrame({m: [float(i) for i in range(WINDOW_SIZE)] for m in METRICS})
    features = summarize(frame)
    assert len(features) == len(METRICS) * len(CONTRACT_SUFFIXES) == 40
    for metric in METRICS:
        for suffix in CONTRACT_SUFFIXES:
            assert f"{metric}{suffix}" in features


def test_windows_yield_window_size_rows_oldest_first():
    n = WINDOW_SIZE + 5
    frame = pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=n, freq="15s"),
        "workload": ["redis"] * n,
        **{m: [float(i) for i in range(n)] for m in METRICS},
    })
    produced = list(windows(frame, "redis", WINDOW_SIZE))
    assert produced, "expected at least one window"
    for w in produced:
        assert len(w) == WINDOW_SIZE
    firsts = [w["ts"].iloc[0] for w in produced]
    assert firsts == sorted(firsts), "windows must be yielded oldest first"


def test_windows_fill_gaps_rather_than_emitting_nan():
    """ffill then fill 0: a hole must not reach the model as NaN."""
    n = WINDOW_SIZE + 2
    frame = pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=n, freq="15s"),
        "workload": ["redis"] * n,
        **{m: [float(i) for i in range(n)] for m in METRICS},
    })
    frame.loc[3, "mem_pct"] = None
    for w in windows(frame, "redis", WINDOW_SIZE):
        assert not w[METRICS].isna().any().any()


# --- diagnosis: the policy table -------------------------------------------------

def test_decide_signature_is_stable():
    assert list(inspect.signature(policy.decide).parameters) == ["predicted_class", "confidence"]


@pytest.mark.parametrize("label,confidence,expected", [
    ("CPU_HOG", 0.95, "scale_out"),
    ("CPU_HOG", 0.50, "scale_out"),
    ("MEMORY_LEAK", 0.95, "rolling_restart"),
    ("MEMORY_LEAK", 0.70, "nothing"),      # below its 0.80 bar
    ("DISK_STRESS", 0.95, "alert_only"),
    ("DISK_STRESS", 0.85, "nothing"),      # below its 0.90 bar
    ("NORMAL", 0.99, "nothing"),
])
def test_policy_table_maps_class_and_confidence_to_action(label, confidence, expected):
    assert policy.decide(label, confidence) == expected


def test_below_the_abstention_floor_nothing_ever_happens():
    for label in ("CPU_HOG", "MEMORY_LEAK", "DISK_STRESS"):
        assert policy.decide(label, policy.ABSTENTION_FLOOR - 0.01) == "nothing"


def test_an_unseen_class_never_produces_an_action():
    """POD_KILL has no policy row, and neither will the next fault type someone adds.

    An unmapped label must fall through to 'nothing' rather than KeyError or,
    far worse, some default action.
    """
    for label in ("POD_KILL", "CONNECTION_POOL_EXHAUSTION", "", None):
        assert policy.decide(label, 0.99) == "nothing"


def test_thresholds_are_ordered_by_how_expensive_being_wrong_is():
    """Cheap hedging unlocks early, expensive remediation only as confidence firms up.

    The consequence is deliberate and falls out of the thresholds; this pins the
    ordering so a future edit cannot silently make restarting easier than alerting.
    """
    scale = policy.POLICY["CPU_HOG"]["min_confidence"]
    restart = policy.POLICY["MEMORY_LEAK"]["min_confidence"]
    alert = policy.POLICY["DISK_STRESS"]["min_confidence"]
    assert scale < restart <= alert
