"""The stages wired together, over data/samples/.

The unit tests ask whether each stage does what it claims. These ask whether the
stages fit: whether what the classifier returns is something the policy table can
consume, and whether what policy returns is an action a controller could
actually execute. Nothing here pins an accuracy number -- a mismatch between
stages is a defect at any accuracy.

The real controller cannot be imported (control/controller.py does
`from kubernetes import client, config` at module scope and CI has no kubernetes
client), so the executing stage is conftest's FakeController and the real one is
inspected as source with `ast`. That is a weaker check than calling it, and it is
the strongest one available from a fresh clone with no cluster.
"""

import ast
from pathlib import Path

import pytest

from control import policy
from ml.detector import Detector
from ml.features import METRICS, WINDOW_SIZE, windows

from conftest import ACTION_VOCABULARY, DECISION_LOG_COLUMNS, REPO_ROOT

CONTROLLER_SOURCE = REPO_ROOT / "control" / "controller.py"
DASHBOARD_SOURCE = REPO_ROOT / "dashboard" / "terminal.py"


def _string_constants(source: Path, function: str) -> set[str]:
    """Every string literal inside one function -- cheap stand-in for calling it."""
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return {
                n.value for n in ast.walk(node)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            }
    raise AssertionError(f"{source.name} has no function {function}()")


def _string_list_assignment(source: Path, name: str) -> list[str]:
    """A module-level `NAME = [...]` of string literals."""
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.List)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        ):
            return [e.value for e in node.value.elts]
    raise AssertionError(f"{source.name} has no {name} assignment")


# --- the loop end to end -----------------------------------------------------

def test_the_whole_loop_runs_over_the_samples_and_decides_something(decisions):
    """detector -> classifier -> policy -> controller, on committed data only.

    Guards against the version of this suite that passes because nothing happens:
    if the detector gate produced no fired ticks, every assertion below would be
    vacuously true. The sample fixture is synthetic and the offline replay path
    uses score_only(), so it does fire here -- unlike the live path on real data
    (see tests/test_units.py's desensitization test).
    """
    rows, controller = decisions
    assert rows, "the detector fired on no sample tick; every cross-stage test below is vacuous"
    assert controller.executed, "no action reached the controller"


def test_every_class_the_classifier_emits_has_a_policy_row(decisions):
    """The classifier/policy boundary. An emitted class with no row falls through
    to 'nothing' by design (POD_KILL), which is safe but silent -- on the shipped
    model, over real fired windows, silence would mean the pipeline never acts.
    """
    rows, _ = decisions
    emitted = {row["predicted_class"] for row in rows}
    assert emitted, "no predictions to check"
    unmapped = emitted - set(policy.POLICY)
    assert unmapped == set(), f"classifier emits classes policy has no row for: {sorted(unmapped)}"


def test_the_shipped_models_whole_label_space_is_mapped(shipped_classifier):
    """Stronger than the test above, which only covers the classes that happened to
    come up on the samples. `model.classes_` is everything the artifact can ever
    return, and the artifact is what cluster loads (SETUP.md §8).
    """
    known = set(shipped_classifier.model.classes_)
    assert known, "shipped classifier knows no classes"
    assert known <= set(policy.POLICY), f"unmapped: {sorted(known - set(policy.POLICY))}"


def test_every_action_the_loop_produces_is_in_the_implementable_vocabulary(decisions):
    rows, _ = decisions
    actions = {row["action"] for row in rows}
    assert actions <= ACTION_VOCABULARY, f"unimplementable: {sorted(actions - ACTION_VOCABULARY)}"


def test_confidence_is_a_probability_and_the_action_agrees_with_it(decisions):
    """Cross-stage consistency, not accuracy: whatever the classifier returned,
    the action on that row has to be what the policy table says for that
    (class, confidence) pair. Catches a loop that logs one thing and does another.
    """
    rows, _ = decisions
    for row in rows:
        assert isinstance(row["predicted_class"], str)
        assert 0.0 <= row["confidence"] <= 1.0
        assert row["action"] == policy.decide(row["predicted_class"], row["confidence"])


def test_the_loop_emits_rows_shaped_like_the_decision_log_contract(decisions):
    """SETUP.md §7's decision log is read by the dashboard and the eval harness.
    Whatever the controller ends up writing, these are the fields the loop has to
    be able to supply -- a loop that never computes `mode` cannot log it.
    """
    rows, _ = decisions
    for row in rows:
        assert list(row) == DECISION_LOG_COLUMNS


# --- the detector/controller boundary ---------------------------------------

def test_the_detector_scores_a_positional_array_in_metrics_order(healthy_metrics):
    """`Detector.score(workload, values)` takes an array of 8 readings in
    ml.features.METRICS order and zips it against METRICS by position.

    This is the one place in the pipeline where a wrong order is silently wrong
    rather than loud, which is exactly why METRICS order is frozen. Pinned so the
    controller knows it must build that list from METRICS, not from the frame's
    column order or a dict.
    """
    workload = sorted(healthy_metrics["workload"].unique())[0]
    row = healthy_metrics[healthy_metrics["workload"] == workload].iloc[-1]
    values = [float(row[m]) for m in METRICS]

    det = Detector.fit_healthy(healthy_metrics)
    score, fired = det.score(workload, values)
    assert isinstance(float(score), float)
    assert isinstance(fired, bool)

    shuffled = Detector.fit_healthy(healthy_metrics)
    other, _ = shuffled.score(workload, list(reversed(values)))
    assert other != score, "score ignores argument order; METRICS order would not be a contract"


def test_the_window_signal_produces_is_the_window_diagnosis_consumes(sample_metrics):
    """signal owns windows(); diagnosis owns classifier.predict(window). predict()
    documents its input as "one column per METRICS, WINDOW_SIZE rows" -- this
    asserts windows() actually hands over that, rather than each side agreeing
    with its own docstring.
    """
    workload = sorted(sample_metrics["workload"].unique())[0]
    produced = list(windows(sample_metrics, workload, WINDOW_SIZE))
    assert produced

    for window in produced[:5]:
        assert len(window) == WINDOW_SIZE
        assert set(METRICS) <= set(window.columns)
        assert not window[METRICS].isna().any().any()


# --- where the stages do not fit (open defects, xfail so CI stays green) -----

@pytest.mark.xfail(
    strict=True,
    reason="control/controller.py:execute_action() only branches on 'restart'; "
           "policy emits scale_out/rolling_restart/alert_only. Cluster's to fix.",
)
def test_the_controller_can_execute_every_action_the_policy_table_emits():
    """The mismatch this whole file exists to find.

    policy.decide() returns one of scale_out, rolling_restart, alert_only,
    nothing. control/controller.py's execute_action() compares `action` against
    the single literal 'restart' and returns 'unknown-action' for anything else.
    The intersection of the two vocabularies is empty: wired up as written, every
    decision the pipeline ever makes would be logged as unknown-action and no
    remediation would ever be applied -- while DRY_RUN=True masks it completely,
    because the dry-run branch returns before the comparison.

    Marked strict xfail rather than deleted: when cluster adds the branches this
    XPASSes, CI goes red, and the marker has to come off. That is the intent.
    """
    handled = _string_constants(CONTROLLER_SOURCE, "execute_action")
    emitted = {entry["action"] for entry in policy.POLICY.values()} - {"nothing"}
    assert emitted <= handled, f"controller cannot execute: {sorted(emitted - handled)}"


@pytest.mark.xfail(
    strict=True,
    reason="controller.log_decision() writes 'timestamp' and omits top_features "
           "and mode; SETUP.md §7 names ts, top_features and mode.",
)
def test_the_controller_writes_the_decision_log_columns_the_contract_names():
    """SETUP.md §7 pins the decision log at ts, workload, detector_score, fired,
    predicted_class, confidence, top_features, action, result, mode -- and says
    changing it requires telling everyone.

    log_decision() writes `timestamp` instead of `ts` and has no top_features or
    mode column at all. `mode` is what separates a shadow-mode observation from
    an executed action, so its absence makes the control-arm comparison
    (ground rule 4) unanswerable from the log. It also writes to
    control/decisions.csv while SETUP.md §8 names data/decisions/log.csv.
    """
    header = None
    tree = ast.parse(CONTROLLER_SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "log_decision":
            for inner in ast.walk(node):
                if isinstance(inner, ast.List) and inner.elts and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str) for e in inner.elts
                ):
                    header = [e.value for e in inner.elts]
                    break
    assert header is not None, "could not find log_decision()'s header row"
    assert header == DECISION_LOG_COLUMNS


@pytest.mark.xfail(
    strict=True,
    reason="dashboard/terminal.py COLUMNS mirrors the controller's actual header, "
           "not SETUP.md §7 -- and row.get(col, '') hides any missing column.",
)
def test_the_dashboard_reads_the_decision_log_columns_the_contract_names():
    """The reading half of the same divergence. dashboard/terminal.py declares
    COLUMNS without top_features or mode and renders each cell with
    `row.get(col, "")`, so a log that is missing a contract column displays as
    blanks rather than failing -- an operator cannot tell "we took no action"
    from "the column is gone".
    """
    assert _string_list_assignment(DASHBOARD_SOURCE, "COLUMNS") == DECISION_LOG_COLUMNS
