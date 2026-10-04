"""Break one stage; assert the next degrades safely.

A pipeline that is correct on good input and arbitrary on bad input is not a
pipeline you can put in front of a production namespace. Each test here breaks
exactly one stage and asserts the next one either does nothing or fails loudly --
never acts on a value it should not trust, and never returns a silently wrong
answer.

Same rules as the rest of the suite: data/samples/ only, no accuracy thresholds,
every test tied to a stated rule or a real defect.
"""

import pandas as pd
import pytest

from control import policy
from ml.features import METRICS, WINDOW_SIZE

from conftest import DECISION_LOG_COLUMNS, FakeController, StubClassifier, run_loop


def _window(rows=WINDOW_SIZE):
    return pd.DataFrame({m: [float(i + 1) for i in range(rows)] for m in METRICS})


# --- the detector gate ------------------------------------------------------

def test_a_detector_that_never_fires_runs_nothing_downstream(sample_metrics):
    """The classifier only ever sees windows the detector flagged -- that is why
    NORMAL is not a classification target (ml/train.py's NON_CLASSIFIABLE_LABELS).

    With an empty gate nothing downstream may run at all. Asserted by call count,
    not by output: a stage that is supposed to be unreachable is only provably
    unreachable if something notices being reached.
    """
    classifier = StubClassifier("CPU_HOG", 0.99)
    controller = FakeController()

    rows = run_loop(sample_metrics, {}, classifier, controller)

    assert rows == []
    assert classifier.calls == 0, "the classifier ran on a tick the detector did not fire on"
    assert controller.executed == [] and controller.declined == []


def test_the_gate_is_per_workload_and_per_tick_not_global(sample_metrics, firing_ticks):
    """A detector that fired on redis must not open the gate for nginx. Gating on
    "something somewhere fired" would have the controller remediate a healthy
    workload, which is worse than missing the fault.
    """
    target = sorted(sample_metrics["workload"].unique())[0]
    single = {k: v for k, v in firing_ticks.items() if k[0] == target}
    assert single, f"no sample firing on {target} to narrow to"

    classifier = StubClassifier("CPU_HOG", 0.99)
    rows = run_loop(sample_metrics, single, classifier, FakeController())

    assert rows
    assert {row["workload"] for row in rows} == {target}


# --- the classifier's output ------------------------------------------------

def test_an_unmapped_class_reaches_the_controller_as_no_action(sample_metrics, firing_ticks):
    """POD_KILL has no row in control/policy.py, deliberately: it is instantaneous
    and there is no diagnosis lever for it (ml/train.py's NON_CLASSIFIABLE_LABELS).

    So a confident POD_KILL must travel the whole loop and come out as 'nothing' --
    not a KeyError, and emphatically not a default action. Run end to end rather
    than against policy.decide() alone, because the failure mode being ruled out
    is a controller that treats an unrecognised class as "do the usual thing".
    """
    classifier = StubClassifier("POD_KILL", 0.99)
    controller = FakeController()

    rows = run_loop(sample_metrics, firing_ticks, classifier, controller)

    assert rows, "expected the loop to run"
    assert {row["action"] for row in rows} == {"nothing"}
    assert controller.executed == [], "an unmapped class caused an action"
    assert controller.declined


def test_below_the_abstention_floor_the_loop_records_the_class_but_acts_on_nothing(
    sample_metrics, firing_ticks
):
    """Abstention has to be visible, not invisible. A sub-floor prediction must be
    logged with the class it was unsure about -- that row is how anyone later
    measures how often the pipeline abstained (ml/abstention.py) -- while the
    action is 'nothing' and the controller does not move.
    """
    classifier = StubClassifier("MEMORY_LEAK", policy.ABSTENTION_FLOOR - 0.01)
    controller = FakeController()

    rows = run_loop(sample_metrics, firing_ticks, classifier, controller)

    assert rows
    assert {row["predicted_class"] for row in rows} == {"MEMORY_LEAK"}
    assert {row["action"] for row in rows} == {"nothing"}
    assert controller.executed == []


def test_a_confidence_just_under_a_threshold_does_not_act(sample_metrics, firing_ticks):
    """Above the abstention floor but below MEMORY_LEAK's own 0.80 bar: the class
    is trusted enough to name and not enough to restart a pod for. Restarting is
    the expensive-to-be-wrong action, which is why its bar is higher.
    """
    bar = policy.POLICY["MEMORY_LEAK"]["min_confidence"]
    classifier = StubClassifier("MEMORY_LEAK", bar - 0.01)
    controller = FakeController()

    run_loop(sample_metrics, firing_ticks, classifier, controller)
    assert controller.executed == []


# --- malformed windows ------------------------------------------------------

def test_a_window_missing_a_metric_fails_loudly(shipped_classifier):
    """summarize() indexes each column by name, so a dropped metric is a KeyError
    rather than a prediction from 35 features. Pinned because the alternative --
    filling the gap with a default -- would look like robustness and produce a
    confident answer about data that was never collected.
    """
    with pytest.raises(KeyError):
        shipped_classifier.predict(_window().drop(columns=["mem_pct"]))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("rows", [0, 1])
def test_a_window_with_too_few_rows_to_fit_a_slope_fails_loudly(shipped_classifier, rows):
    """A slope needs two points. One row or none raises out of np.polyfit rather
    than producing NaN features -- NaN would reach the forest, and a NaN-driven
    prediction is the silently-wrong outcome this file exists to rule out.
    """
    with pytest.raises((TypeError, ValueError)):
        shipped_classifier.predict(_window(rows))


def test_column_order_does_not_change_the_prediction(shipped_classifier):
    """The safe half of the malformed-window story, worth pinning as a guarantee.

    summarize() looks metrics up by name and predict() reindexes to
    self.feature_columns, so a caller that assembles a window in a different
    column order gets the identical answer. METRICS order matters for
    Detector.score(), which is positional; it must not start mattering here too.
    """
    window = _window()
    assert shipped_classifier.predict(window) == shipped_classifier.predict(
        window[list(reversed(METRICS))]
    )


def test_an_extra_column_is_ignored_rather_than_shifting_the_features(shipped_classifier):
    """A metrics table that gains a column (a new metric, a join artifact) must not
    change what the existing 40 features mean.
    """
    window = _window()
    assert shipped_classifier.predict(window) == shipped_classifier.predict(
        window.assign(unexpected_column=1.0)
    )


def test_a_window_of_the_wrong_length_is_rejected_rather_than_predicted_on(shipped_classifier):
    """Was the one genuinely silent failure left in the chain. Now fixed.

    predict() documented its input as WINDOW_SIZE rows and never checked. Any
    length from 2 upward summarized fine and came back as a confident label
    computed over the wrong amount of history -- a 5-row "window" is 75 seconds
    of data scored by a model trained on 5 minutes, and nothing in the pipeline
    noticed. windows() never produces one, so it only bit a caller that builds a
    window itself: exactly what the controller does from a short buffer at
    startup or after a restart, which is also when a wrong diagnosis costs most.

    It now raises ValueError with a message that says what to do instead (wait
    for the buffer), because the caller hitting this is mid-loop and needs to
    know the fix, not just the fault.
    """
    for rows in (2, WINDOW_SIZE - 1, WINDOW_SIZE + 1):
        with pytest.raises(ValueError, match="WINDOW_SIZE"):
            shipped_classifier.predict(_window(rows))

    # And the right length must still work, or this would "pass" by rejecting
    # everything.
    label, confidence = shipped_classifier.predict(_window(WINDOW_SIZE))
    assert isinstance(label, str)
    assert 0.0 <= float(confidence) <= 1.0


# --- the decision log -------------------------------------------------------

def test_a_decision_row_missing_a_contract_column_is_caught_by_strict_selection(decisions):
    """SETUP.md §7 names ten columns; a consumer that wants to notice a missing one
    has to select them by name. `df[DECISION_LOG_COLUMNS]` raises KeyError, while
    the `row.get(col, "")` pattern in dashboard/terminal.py renders blanks and
    says nothing.

    This pins the behaviour a consumer must have -- and the companion xfail in
    tests/test_integration.py records that the shipped dashboard does not have it.
    """
    rows, _ = decisions
    frame = pd.DataFrame(rows)
    assert list(frame[DECISION_LOG_COLUMNS].columns) == DECISION_LOG_COLUMNS

    for dropped in ("mode", "top_features", "ts"):
        with pytest.raises(KeyError):
            frame.drop(columns=[dropped])[DECISION_LOG_COLUMNS]
