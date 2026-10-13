"""The four safety rails, driven by an adversarial caller and a spy API.

`control/controller.py` has exactly four things standing between a model's guess
and a production cluster: `DRY_RUN`, the `STOP` kill switch, the per-workload
`COOLDOWN_SECONDS`, and `MAX_REPLICAS`. Until this file existed none of them had
ever been called by a test. The project has already been burned by precisely
that gap: `execute_action()` compared `action` against the literal `"restart"`
while `policy.decide()` returns `scale_out`/`rolling_restart`/`alert_only`/
`nothing` -- the intersection was EMPTY, so no remediation would ever have
fired, and `DRY_RUN=True` masked it completely because the dry-run branch
returns before the comparison. `tests/test_integration.py` caught that by
parsing this file's source, which was the strongest check available at the time.
It is no longer: `kubernetes` is imported lazily inside
`connect_to_kubernetes()`, so every rail can be *called* offline.

Method, and the reason these tests are worth anything:

- The API is a spy (`SpyAppsApi`) that records every call and returns plausible
  objects, or `ForbiddenApi`, which raises on *any* attribute access. "No API
  call was made" is asserted against the object, not inferred from a return
  value.
- No rail is ever stubbed out to test another one. `DRY_RUN` is restored by
  monkeypatch; `STOP_FILE`, `LOG_FILE` and `last_action_time` are redirected to
  per-test temporaries so a test cannot leak cluster state into the next one.
- Labels and confidences are derived from `control.policy.POLICY` rather than
  hardcoded, so a new lever in the policy table shows up here as a new
  parametrised case instead of as a silent gap.

`DRY_RUN` must stay `True` in the committed source. That is asserted twice:
once against the parsed source (`test_dry_run_is_true_in_the_committed_source`)
and once in the autouse fixture, so a test that forgets to restore it cannot
make the rest of this file pass for the wrong reason.
"""

import ast
import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control import controller  # noqa: E402
from control import policy  # noqa: E402
from ml.features import METRICS, WINDOW_SIZE  # noqa: E402

CONTROLLER_SOURCE = REPO_ROOT / "control" / "controller.py"

# Every action the policy table can emit, and the subset that is supposed to
# touch the cluster. Derived, not hardcoded: a new lever in POLICY becomes a new
# parametrised case here rather than an untested code path.
ALL_ACTIONS = sorted({entry["action"] for entry in policy.POLICY.values()})
ACTIONABLE = sorted(set(ALL_ACTIONS) - {"nothing"})

# SETUP.md S7, verbatim -- the same list tests/conftest.py pins, repeated here on
# purpose so this file does not depend on Aahan's fixtures.
DECISION_LOG_COLUMNS = [
    "ts", "workload", "detector_score", "fired", "predicted_class",
    "confidence", "top_features", "action", "result", "mode",
]


# --- doubles -----------------------------------------------------------------

class ForbiddenApi:
    """Any attribute access at all is a test failure.

    Used wherever the claim is "the controller did not talk to the cluster".
    Stronger than a spy with an empty call list: it fails even if the controller
    merely reaches for a method it does not end up calling.
    """

    def __getattr__(self, name):
        raise AssertionError(f"the controller touched the cluster: apps_api.{name}")


class SpyAppsApi:
    """Records every call; optionally raises to simulate a cluster that refuses.

    `replicas` is what `read_namespaced_deployment` reports, so a test can put
    the cluster at, below or *above* MAX_REPLICAS and see what scale_out does.
    """

    def __init__(self, replicas=1, raises=None):
        self.replicas = replicas
        self.raises = raises
        self.calls = []

    def _maybe_raise(self):
        if self.raises is not None:
            raise self.raises

    def read_namespaced_deployment(self, name, namespace):
        self.calls.append(("read", name, namespace))
        self._maybe_raise()
        return SimpleNamespace(spec=SimpleNamespace(replicas=self.replicas))

    def patch_namespaced_deployment_scale(self, name, namespace, body):
        self.calls.append(("scale", name, namespace, body["spec"]["replicas"]))
        self._maybe_raise()

    def patch_namespaced_deployment(self, name, namespace, body):
        self.calls.append(("restart", name, namespace))
        self._maybe_raise()

    @property
    def mutations(self):
        """Calls that change the cluster. A read is not a mutation."""
        return [call for call in self.calls if call[0] != "read"]

    @property
    def scale_targets(self):
        return [call[3] for call in self.calls if call[0] == "scale"]


class FakeDetector:
    """Trivial stand-in: `data/healthy/metrics.parquet` is missing, so a real
    fitted Detector cannot be built offline, and none of these tests need one.
    Counts `on_restart` so "dry-run mutated no state" is checkable.
    """

    def __init__(self, fired=True, score=9.99):
        self.fired = fired
        self.score_value = score
        self.scored = []
        self.restarts = []

    def score(self, workload, values):
        self.scored.append((workload, tuple(values)))
        return self.score_value, self.fired

    def on_restart(self, workload):
        self.restarts.append(workload)


class FakeClassifier:
    def __init__(self, label, confidence):
        self.label = label
        self.confidence = confidence
        self.calls = 0

    def predict(self, window):
        self.calls += 1
        return self.label, self.confidence


def label_for(action):
    """A class the policy table maps to `action`, at confidence 1.0.

    Derived from POLICY so the test cannot drift from the table it is testing.
    """
    for label, entry in policy.POLICY.items():
        if entry["action"] == action:
            return label
    raise AssertionError(f"no POLICY row emits {action!r}")


def one_tick():
    return {metric: 0.0 for metric in METRICS}


def full_history():
    """Exactly enough ticks that `evaluate_once` reaches the classifier."""
    return [[0.0] * len(METRICS) for _ in range(WINDOW_SIZE)]


def run_once(apps_api, action, workload="nginx", detector=None, confidence=1.0):
    """Drive `evaluate_once` so that `policy.decide()` returns `action`."""
    detector = FakeDetector() if detector is None else detector
    classifier = FakeClassifier(label_for(action), confidence)
    result = controller.evaluate_once(
        apps_api, workload, one_tick(), detector, classifier, full_history()
    )
    return result, detector, classifier


def log_rows():
    with controller.LOG_FILE.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- isolation ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_controller_state(tmp_path, monkeypatch):
    """Redirect every piece of controller global state at a temporary.

    `STOP_FILE` and `LOG_FILE` are relative paths resolved against the working
    directory, so without this a test would read the repo's real STOP file and
    append to the real decision log.

    The assertions are the hard constraint: `DRY_RUN` is `True`, and any test
    that monkeypatches it must leave it that way.
    """
    monkeypatch.setattr(controller, "LOG_FILE", tmp_path / "decisions" / "log.csv")
    monkeypatch.setattr(controller, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(controller, "last_action_time", {})
    # The kill switch deliberately ALSO honours a cwd-relative STOP, so without
    # emptying this a test would read the repo's real working directory -- the
    # same leak `STOP_FILE` is redirected to avoid. The union behaviour gets its
    # own test, which puts a path back.
    monkeypatch.setattr(controller, "EXTRA_STOP_PATHS", [])
    # Note: the cooldown lease directory needs no redirect of its own. It is
    # derived from LOG_FILE (`controller.cooldown_dir()`), so redirecting the log
    # moves the shared cooldown state with it -- which is the point of deriving
    # it rather than naming it separately.
    assert controller.DRY_RUN is True, "DRY_RUN must be True at the start of every test"
    yield
    assert controller.DRY_RUN is True, "a test left DRY_RUN flipped"


def rewind_cooldown(workload, seconds):
    """Age `workload`'s cooldown by `seconds`, in BOTH places that hold it.

    The in-memory dict alone is no longer the state: `last_action()` takes the
    later of memory and the shared lease file, so a test that rewound only the
    dict would be asserting against a cooldown the lease is still holding open.
    That is the fail-safe direction, which is exactly why it has to be rewound
    deliberately rather than by forgetting.
    """
    when = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    controller.last_action_time[workload] = when
    lease = controller.cooldown_dir() / controller._lease_name(workload)
    if lease.exists():
        lease.write_text(when.isoformat())
    return when


def lease_for(workload):
    return controller.cooldown_dir() / controller._lease_name(workload)


@pytest.fixture
def live():
    """Turn off dry-run *for one test only*, restored in this fixture's finally.

    Every rail other than DRY_RUN is only observable with DRY_RUN off, because
    the dry-run branch returns before any of them run -- which is exactly how
    the original wiring bug stayed hidden.

    Deliberately NOT `monkeypatch.setattr`: monkeypatch is one shared
    function-scoped fixture, so its undo runs *after* every other fixture's
    teardown, including `isolate_controller_state`'s "a test left DRY_RUN
    flipped" assertion -- which then fired on every live test. Restoring here
    means the guard sees the restored value. The guard working is the point; it
    caught this.
    """
    original = controller.DRY_RUN
    controller.DRY_RUN = False
    try:
        yield controller
    finally:
        controller.DRY_RUN = original


@pytest.fixture
def stop_file():
    controller.STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    controller.STOP_FILE.write_text("halt\n")
    assert controller.kill_switch_active()
    return controller.STOP_FILE


# --- the hard constraint -----------------------------------------------------

def test_dry_run_is_true_in_the_committed_source():
    """Asserted against the parsed file, not the imported module.

    A test that monkeypatches `controller.DRY_RUN` and forgets to restore it
    cannot satisfy this, and neither can a module that computes DRY_RUN from an
    environment variable -- which is the shape that would let a default go live.
    """
    tree = ast.parse(CONTROLLER_SOURCE.read_text())
    assigned = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DRY_RUN" for t in node.targets)
    ]
    assert len(assigned) == 1, "DRY_RUN is assigned more or less than once at module scope"
    assert isinstance(assigned[0], ast.Constant) and assigned[0].value is True, (
        "DRY_RUN must be the literal True in committed source, not computed"
    )


def test_nothing_in_the_module_can_flip_dry_run():
    """No function body may assign DRY_RUN. A code path that flips it is the one
    thing this seat is not allowed to ship, so it is asserted rather than
    promised.
    """
    tree = ast.parse(CONTROLLER_SOURCE.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Global) and "DRY_RUN" in inner.names:
                offenders.append(node.name)
            elif isinstance(inner, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "DRY_RUN" for t in inner.targets):
                    offenders.append(node.name)
            elif isinstance(inner, ast.AugAssign):
                if isinstance(inner.target, ast.Name) and inner.target.id == "DRY_RUN":
                    offenders.append(node.name)
    assert offenders == [], f"these functions can flip DRY_RUN: {sorted(set(offenders))}"


# --- rail 1: the kill switch -------------------------------------------------

@pytest.mark.parametrize("action", ACTIONABLE)
def test_the_kill_switch_blocks_every_actionable_decision(live, stop_file, action):
    """STOP present => no API call, for every action the policy table can emit.

    Deliberately the most favourable possible conditions for acting: dry-run
    OFF, confidence 1.0, cooldown never recorded, detector firing. If the kill
    switch holds here it holds everywhere.
    """
    api = ForbiddenApi()
    result, detector, _ = run_once(api, action)
    assert result == "blocked-kill-switch"
    assert detector.restarts == [], "a blocked action still reset the detector"


def test_the_kill_switch_beats_an_inactive_cooldown_and_perfect_confidence(live, stop_file):
    """The ordering inside evaluate_once, pinned.

    The kill switch is checked before the cooldown on purpose: if the cooldown
    were checked first, a workload in cooldown would log `blocked-cooldown` and
    an operator reading the log could not tell the kill switch was even engaged.
    """
    assert not controller.cooldown_active("nginx")
    result, _, _ = run_once(ForbiddenApi(), "rolling_restart", confidence=1.0)
    assert result == "blocked-kill-switch"
    assert log_rows()[-1]["result"] == "blocked-kill-switch"


def test_a_kill_switch_block_records_no_cooldown(live, stop_file):
    """A blocked action must not record a timestamp it did not earn.

    If it did, deleting STOP would leave every workload in a cooldown caused by
    actions that never happened -- the kill switch would have silently converted
    itself into a lockout.
    """
    run_once(ForbiddenApi(), "scale_out")
    assert controller.last_action_time == {}
    assert not controller.cooldown_active("nginx")


def test_execute_action_honours_the_kill_switch_when_called_directly(live, stop_file):
    """Defence in depth: the rail lives at the point of action, not only in the
    orchestration above it.

    `evaluate_once` is not the only caller of `execute_action` -- `run_loop`,
    the eval harness, and any future head-to-head script can call it. A kill
    switch that is only enforced one level up is bypassed by every one of them.
    """
    api = ForbiddenApi()
    assert controller.execute_action(api, "nginx", "rolling_restart") == "blocked-kill-switch"
    assert controller.execute_action(api, "nginx", "scale_out") == "blocked-kill-switch"


def test_the_kill_switch_is_only_the_file(live):
    """No STOP file => the switch is open. Guards against a test suite that
    passes because the switch is stuck closed and nothing ever executes.
    """
    assert not controller.kill_switch_active()
    api = SpyAppsApi(replicas=1)
    result, _, _ = run_once(api, "rolling_restart")
    assert result == "executed", "nothing executes even without STOP; the tests above are vacuous"
    assert api.mutations, "claimed executed but never called the cluster"


# --- rail 2: the cooldown ----------------------------------------------------

def test_the_cooldown_blocks_a_second_action_inside_the_window(live):
    api = SpyAppsApi(replicas=1)
    first, _, _ = run_once(api, "rolling_restart")
    second, _, _ = run_once(api, "rolling_restart")
    assert (first, second) == ("executed", "blocked-cooldown")
    assert len(api.mutations) == 1, "the cooldown let a second patch through"


def test_the_cooldown_is_per_workload(live):
    """Per-workload, not global: an incident on one deployment must not stop
    Prodrome acting on another.
    """
    api = SpyAppsApi(replicas=1)
    assert run_once(api, "rolling_restart", workload="nginx")[0] == "executed"
    assert run_once(api, "rolling_restart", workload="redis")[0] == "executed"
    assert run_once(api, "rolling_restart", workload="nginx")[0] == "blocked-cooldown"
    assert sorted(controller.last_action_time) == ["nginx", "redis"]


def test_a_cooldown_block_does_not_extend_the_cooldown(live):
    """The lockout bug shape, asserted directly.

    If a blocked action recorded a timestamp, every detector firing inside the
    window would push the window forward, and at a 61% false-positive rate the
    workload would never leave cooldown again -- a permanent lockout that looks
    exactly like a working rail from the outside.
    """
    api = SpyAppsApi(replicas=1)
    run_once(api, "rolling_restart")
    recorded = controller.last_action_time["nginx"]
    for _ in range(5):
        assert run_once(api, "rolling_restart")[0] == "blocked-cooldown"
    assert controller.last_action_time["nginx"] == recorded, "a blocked action moved the window"


def test_the_cooldown_expires(live):
    """And the converse of the lockout: the window actually opens again.

    A rail that never expires is as broken as one that never engages.
    """
    api = SpyAppsApi(replicas=1)
    assert run_once(api, "rolling_restart")[0] == "executed"
    rewind_cooldown("nginx", controller.COOLDOWN_SECONDS + 1)
    assert not controller.cooldown_active("nginx")
    assert run_once(api, "rolling_restart")[0] == "executed"
    assert len(api.mutations) == 2


def test_cooldown_active_is_exclusive_at_the_boundary():
    """Pins `elapsed < COOLDOWN_SECONDS`: at exactly the window the rail lifts.

    Documented rather than argued about, because a rail whose boundary nobody
    can state is a rail nobody can reason about.
    """
    now = datetime.now(timezone.utc)
    controller.last_action_time["nginx"] = now - timedelta(
        seconds=controller.COOLDOWN_SECONDS - 1
    )
    assert controller.cooldown_active("nginx")
    controller.last_action_time["nginx"] = now - timedelta(
        seconds=controller.COOLDOWN_SECONDS
    )
    assert not controller.cooldown_active("nginx")


def test_an_unknown_workload_is_not_in_cooldown():
    assert controller.last_action_time == {}
    assert not controller.cooldown_active("never-seen")


def test_alert_only_does_not_consume_the_cooldown(live):
    """alert_only touches nothing, so it must not spend the budget that a real
    remediation needs. Otherwise one DISK_STRESS firing suppresses the
    rolling_restart that a concurrent MEMORY_LEAK diagnosis asks for.
    """
    api = SpyAppsApi(replicas=1)
    assert run_once(api, "alert_only")[0] == "no-op"
    assert api.calls == []
    assert controller.last_action_time == {}
    assert run_once(api, "rolling_restart")[0] == "executed"


def test_a_failed_action_is_still_logged_and_still_consumes_the_cooldown(live):
    """The gap that makes a false-positive storm unbounded.

    A patch that raises (409 conflict, 403, API server restart) may or may not
    have landed. Before this, the exception escaped `evaluate_once` *before*
    `log_decision`, so the attempt left no audit row at all AND recorded no
    cooldown -- so the loop retried every tick, forever, invisibly. An attempt
    whose outcome is unknown has to count as an action.
    """
    api = SpyAppsApi(replicas=1, raises=RuntimeError("409 Conflict"))
    with pytest.raises(RuntimeError):
        run_once(api, "rolling_restart")
    rows = log_rows()
    assert rows, "a failed action left no row in the decision log"
    assert rows[-1]["result"] == "failed"
    assert rows[-1]["action"] == "rolling_restart"
    assert controller.cooldown_active("nginx"), "a failed attempt was not rate limited"


# --- rail 3: MAX_REPLICAS ----------------------------------------------------

@pytest.mark.parametrize("over", [1, 2, 3, 10, 1000])
def test_scale_refuses_to_exceed_the_cap_and_makes_no_call(over):
    """The cap is enforced before the API call, not after.

    Asserted on the spy: a ValueError raised after the patch already went out
    would satisfy `pytest.raises` while the cluster was already over the cap.
    """
    api = SpyAppsApi()
    with pytest.raises(ValueError):
        controller.scale(api, "nginx", controller.MAX_REPLICAS + over)
    assert api.calls == [], "scale() called the cluster before refusing"


def test_scale_allows_exactly_the_cap():
    api = SpyAppsApi()
    controller.scale(api, "nginx", controller.MAX_REPLICAS)
    assert api.scale_targets == [controller.MAX_REPLICAS]


@pytest.mark.parametrize("current", list(range(0, 12)))
def test_scale_out_never_exceeds_the_cap_for_any_reachable_current(live, current):
    """`min(current + 1, MAX_REPLICAS)` over every reachable `current`, including
    a cluster already above the cap -- which is reachable, because an operator
    or an HPA can scale a deployment by hand at any time.
    """
    api = SpyAppsApi(replicas=current)
    run_once(api, "scale_out")
    for target in api.scale_targets:
        assert target <= controller.MAX_REPLICAS, f"scale_out asked for {target}"


@pytest.mark.parametrize("current", [6, 7, 10])
def test_scale_out_never_scales_in(live, current):
    """A cluster above the cap must not be scaled DOWN by a scale_out decision.

    `min(current + 1, MAX_REPLICAS)` reduces 8 replicas to 5. That is capacity
    being removed from a workload during an incident, by an action named
    scale_out, on a model's guess -- the most expensive possible way for this
    rail to be wrong, and it reads as correct at a glance.
    """
    api = SpyAppsApi(replicas=current)
    result, _, _ = run_once(api, "scale_out")
    for target in api.scale_targets:
        assert target >= current, f"scale_out reduced replicas from {current} to {target}"
    assert result != "executed" or api.scale_targets, "claimed executed without scaling"


def test_scale_out_at_the_cap_makes_no_write_and_says_so(live):
    """At the cap there is nothing to do, and the log has to say that rather than
    `executed`: an operator cannot tell a no-op patch from a real scale-up.
    """
    api = SpyAppsApi(replicas=controller.MAX_REPLICAS)
    result, _, _ = run_once(api, "scale_out")
    assert result == "at-max"
    assert api.mutations == [], "patched the deployment with its current replica count"
    assert controller.last_action_time == {}, "a no-op consumed the cooldown"


def test_scale_out_below_the_cap_adds_exactly_one(live):
    api = SpyAppsApi(replicas=2)
    assert run_once(api, "scale_out")[0] == "executed"
    assert api.scale_targets == [3]


def test_scale_out_survives_a_deployment_with_no_replica_count(live):
    """`spec.replicas` is optional in the Kubernetes API and means 1 when unset.

    Unhandled, `min(None + 1, MAX_REPLICAS)` raises TypeError mid-remediation.
    """
    api = SpyAppsApi(replicas=None)
    result, _, _ = run_once(api, "scale_out")
    assert result == "executed"
    assert api.scale_targets == [2]


# --- rail 4: dry-run mutates nothing ----------------------------------------

@pytest.mark.parametrize("action", ACTIONABLE)
def test_dry_run_makes_no_api_call_for_any_action(action):
    result, _, _ = run_once(ForbiddenApi(), action)
    assert result == "dry-run"


@pytest.mark.parametrize("action", ACTIONABLE)
def test_dry_run_records_no_cooldown_and_resets_no_detector(action):
    """The shadow arm must not accumulate state the live arm would.

    If `record_action` fired on a dry run, the shadow arm would suppress its own
    later observations and the control-arm comparison would be comparing two
    different policies. If `on_restart` fired, the shadow arm would blind the
    detector for a restart that never happened.
    """
    _, detector, _ = run_once(ForbiddenApi(), action)
    assert controller.last_action_time == {}, "dry-run recorded a cooldown"
    assert detector.restarts == [], "dry-run reset the detector"


def test_dry_run_logs_mode_shadow_with_the_contract_columns():
    run_once(ForbiddenApi(), "rolling_restart")
    with controller.LOG_FILE.open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header == DECISION_LOG_COLUMNS
    row = log_rows()[-1]
    assert row["mode"] == "shadow"
    assert row["result"] == "dry-run"
    assert row["action"] == "rolling_restart"


def test_mode_is_derived_from_dry_run_not_passed_in(live):
    """A row can never claim to be live while the process is dry-running, or the
    reverse. `mode` is the only column that separates an observation from an
    action, so it is computed from the live setting rather than supplied.
    """
    run_once(SpyAppsApi(replicas=1), "rolling_restart")
    assert log_rows()[-1]["mode"] == "live"


def test_a_live_restart_does_reset_the_detector(live):
    """The converse of the dry-run test: on a real restart the suppression
    window must engage, or the restart's own startup burst re-fires the detector
    and the controller restarts again (the Part 4.4 loop).
    """
    api = SpyAppsApi(replicas=1)
    _, detector, _ = run_once(api, "rolling_restart")
    assert detector.restarts == ["nginx"]


def test_a_detector_without_on_restart_is_tolerated(live):
    """The detector interface is consumed as published; `on_restart` is optional
    and its absence must not break remediation.
    """
    class NoHook:
        def score(self, workload, values):
            return 9.9, True

    api = SpyAppsApi(replicas=1)
    classifier = FakeClassifier(label_for("rolling_restart"), 1.0)
    result = controller.evaluate_once(
        api, "nginx", one_tick(), NoHook(), classifier, full_history()
    )
    assert result == "executed"


# --- the action vocabulary --------------------------------------------------

@pytest.mark.parametrize("action", ALL_ACTIONS)
def test_every_action_the_policy_table_can_emit_is_executable(live, action):
    """The defect this seat was burned by, now asserted by *calling* the code
    rather than by parsing it for string literals.

    A policy action with no branch in `execute_action` logs `unknown-action` and
    silently does nothing -- and with DRY_RUN on, the dry-run branch returns
    first, so the gap is invisible until the day dry-run is turned off.
    """
    api = SpyAppsApi(replicas=1)
    result = controller.execute_action(api, "nginx", action)
    assert result != "unknown-action", f"policy emits {action!r} and the controller cannot run it"


@pytest.mark.parametrize("action", ["restart", "scale_in", "", "SCALE_OUT", "drop_database", None])
def test_an_unrecognised_action_fails_closed_and_loudly(live, action):
    """Anything outside the vocabulary must do nothing *and say so*.

    `unknown-action` in the decision log is the signal that diagnosis shipped a
    lever cluster has not implemented. Returning `no-op` for it would make that
    indistinguishable from a deliberate decision not to act.
    """
    api = ForbiddenApi()
    assert controller.execute_action(api, "nginx", action) == "unknown-action"


def test_policy_decide_only_ever_emits_implementable_actions():
    """Swept over the whole (class, confidence) grid including out-of-range and
    unmapped classes, so the vocabulary is pinned from the producing side too.
    """
    classes = list(policy.POLICY) + ["POD_KILL", "", "not-a-class"]
    confidences = [0.0, 0.49, 0.5, 0.79, 0.8, 0.89, 0.9, 0.999, 1.0]
    for predicted in classes:
        for confidence in confidences:
            assert policy.decide(predicted, confidence) in set(ALL_ACTIONS)


def test_nothing_never_reaches_the_cluster(live):
    """`action == "nothing"` short-circuits before the rails even run, so it has
    to be unable to call the API by construction.
    """
    result, _, _ = run_once(ForbiddenApi(), "nothing")
    assert result == "abstained"
    assert controller.last_action_time == {}


def test_a_detector_that_does_not_fire_reaches_neither_classifier_nor_cluster():
    """The gate above all four rails. A stage that is meant to be unreachable is
    only provably unreachable if something notices being reached.
    """
    detector = FakeDetector(fired=False)
    classifier = FakeClassifier(label_for("rolling_restart"), 1.0)
    result = controller.evaluate_once(
        ForbiddenApi(), "nginx", one_tick(), detector, classifier, full_history()
    )
    assert result == "not-fired"
    assert classifier.calls == 0, "the classifier ran on a tick the detector did not fire on"
    assert log_rows()[-1]["fired"] == "False"


def test_a_firing_without_enough_history_does_not_act(live):
    """Fewer than WINDOW_SIZE ticks means the classifier cannot be asked, so the
    only safe action is none -- and in particular not the `NORMAL`/1.0 default
    being mistaken for a real diagnosis.
    """
    detector = FakeDetector()
    classifier = FakeClassifier(label_for("rolling_restart"), 1.0)
    result = controller.evaluate_once(
        ForbiddenApi(), "nginx", one_tick(), detector, classifier,
        full_history()[: WINDOW_SIZE - 1],
    )
    assert result == "abstained"
    assert classifier.calls == 0
    assert log_rows()[-1]["action"] == "nothing"


# --- the decision log's top_features column ---------------------------------
#
# This was a strict xfail: classifier.predict() returned (label, confidence)
# only, so evaluate_once had nothing to pass and log_decision wrote "", which
# pandas reads back as NaN -- the entire explanation for "top_features is NaN
# throughout" in the 3,276-decision export. Diagnosis widened the interface
# additively (classifier.top_features(window) / explain(window, top_n)), so the
# marker is gone and the behaviour is pinned instead. The tests below are what
# the xfail becomes once a finding is actually fixed.


class ExplainingClassifier(FakeClassifier):
    """A classifier with the attribution capability, shaped like the real one.

    Records the window it was handed, so a test can assert the attribution was
    computed from the SAME window as the prediction rather than a different
    slice of history -- an attribution over the wrong window is exactly the
    failure diagnosis's ValueError on short windows exists to prevent.
    """

    def __init__(self, label, confidence, value="fs_writes_max=+0.140;fs_writes_std=+0.120"):
        super().__init__(label, confidence)
        self.value = value
        self.explained = []

    def top_features(self, window, top_n=3):
        self.explained.append(window)
        return self.value


def run_explaining(apps_api, action, workload="nginx", classifier=None, history=None):
    detector = FakeDetector()
    classifier = ExplainingClassifier(label_for(action), 1.0) if classifier is None else classifier
    result = controller.evaluate_once(
        apps_api, workload, one_tick(), detector, classifier,
        full_history() if history is None else history,
    )
    return result, detector, classifier


def test_the_decision_log_carries_feature_attribution(live):
    _, _, classifier = run_explaining(SpyAppsApi(replicas=1), "rolling_restart")
    row = log_rows()[-1]
    assert row["top_features"] == classifier.value
    assert row["top_features"] != ""


def test_the_attribution_is_computed_from_the_window_that_was_predicted_on(live):
    """One window, one prediction, one explanation.

    If the attribution were recomputed from a different window it could name
    features from history the decision was not made on -- a plausible-looking
    reason for a decision that had another cause, which is worse than an empty
    cell because an operator would act on it.
    """
    _, _, classifier = run_explaining(SpyAppsApi(replicas=1), "rolling_restart")
    assert classifier.calls == 1
    assert len(classifier.explained) == 1
    window = classifier.explained[0]
    assert len(window) == WINDOW_SIZE
    assert list(window.columns) == METRICS


def test_the_attribution_survives_the_csv_round_trip_in_one_field(live):
    """Semicolon-separated, never comma: the value lives in a CSV cell and has to
    come back out of `csv.DictReader` as ONE field, unsplit and still greppable.
    """
    value = "fs_writes_max=+0.140;fs_writes_std=+0.120;fs_writes_last=-0.104"
    classifier = ExplainingClassifier(label_for("rolling_restart"), 1.0, value=value)
    run_explaining(SpyAppsApi(replicas=1), "rolling_restart", classifier=classifier)
    row = log_rows()[-1]
    assert row["top_features"] == value
    assert "," not in row["top_features"], "a comma would split the cell or force quoting"
    assert len(row) == len(DECISION_LOG_COLUMNS), "the attribution leaked into other columns"
    assert row["action"] == "rolling_restart" and row["result"] == "executed"


def test_a_failed_action_still_records_its_attribution(live):
    """The row an operator wants most: an action that may have half-landed, with
    the reason it was chosen attached.
    """
    api = SpyAppsApi(replicas=1, raises=RuntimeError("409 Conflict"))
    classifier = ExplainingClassifier(label_for("rolling_restart"), 1.0)
    with pytest.raises(RuntimeError):
        run_explaining(api, "rolling_restart", classifier=classifier)
    row = log_rows()[-1]
    assert row["result"] == "failed"
    assert row["top_features"] == classifier.value


def test_a_blocked_action_still_records_its_attribution(live, stop_file):
    """A kill-switch block is a decision too, and `what would it have done, and
    why` is the whole point of shadow mode.
    """
    run_explaining(ForbiddenApi(), "rolling_restart")
    row = log_rows()[-1]
    assert row["result"] == "blocked-kill-switch"
    assert row["top_features"] != ""


def test_dry_run_still_records_the_attribution_but_nothing_else():
    """Attribution is an audit string, not cluster state: the shadow arm must
    still explain itself while mutating nothing. Guards against "fixed" meaning
    the shadow arm started calling a second model path with side effects.
    """
    _, detector, classifier = run_explaining(ForbiddenApi(), "rolling_restart")
    row = log_rows()[-1]
    assert row["mode"] == "shadow" and row["result"] == "dry-run"
    assert row["top_features"] == classifier.value
    assert controller.last_action_time == {}
    assert detector.restarts == []


def test_the_attribution_is_not_computed_on_a_short_window(live):
    """`explain()` raises ValueError below WINDOW_SIZE, deliberately: a confident
    attribution over the wrong amount of history is worse than none. So the
    short-buffer path must not call it at all -- and must still write a row.
    """
    classifier = ExplainingClassifier(label_for("rolling_restart"), 1.0)
    result = controller.evaluate_once(
        ForbiddenApi(), "nginx", one_tick(), FakeDetector(), classifier,
        full_history()[: WINDOW_SIZE - 1],
    )
    assert result == "abstained"
    assert classifier.explained == [], "explain() was called on a short window"
    assert classifier.calls == 0
    row = log_rows()[-1]
    assert row["top_features"] == ""
    assert row["action"] == "nothing"


def test_a_tick_the_detector_ignored_has_no_attribution():
    """No prediction, so nothing to explain. The cell has to be empty rather
    than stale from a previous tick.
    """
    classifier = ExplainingClassifier(label_for("rolling_restart"), 1.0)
    controller.evaluate_once(
        ForbiddenApi(), "nginx", one_tick(), FakeDetector(fired=False), classifier,
        full_history(),
    )
    assert classifier.explained == []
    assert log_rows()[-1]["top_features"] == ""


def test_a_classifier_without_the_capability_degrades_to_an_empty_cell(live):
    """The published interface is consumed as published, and `top_features` is
    optional: a classifier artifact predating the capability must not take the
    control loop down over an audit string. Remediation still happens.
    """
    api = SpyAppsApi(replicas=1)
    result, _, _ = run_once(api, "rolling_restart")  # FakeClassifier: no top_features
    assert result == "executed"
    assert log_rows()[-1]["top_features"] == ""


def test_a_classifier_whose_attribution_raises_fails_loudly_before_acting(live):
    """Deliberately NOT swallowed.

    Swallowing it would recreate the defect this column just came out of: a
    silently empty field indistinguishable from "no features mattered". It is
    safe to be loud -- attribution runs before every rail and before any cluster
    call, so the raise cannot leave a half-applied remediation behind. Asserted:
    no API call, no cooldown recorded.
    """
    class Broken(FakeClassifier):
        def top_features(self, window, top_n=3):
            raise RuntimeError("tree decomposition blew up")

    api = ForbiddenApi()
    classifier = Broken(label_for("rolling_restart"), 1.0)
    with pytest.raises(RuntimeError):
        controller.evaluate_once(
            api, "nginx", one_tick(), FakeDetector(), classifier, full_history()
        )
    assert controller.last_action_time == {}, "a crashed attribution still burned the cooldown"


def test_the_shipped_classifier_publishes_the_attribution_capability():
    """A contract check on diagnosis's published interface, by parsing rather
    than importing: `import ml.classifier` loads or refits classifier.pkl, which
    is most of the suite's runtime and not something this file needs.

    If `top_features` were ever removed, every row would silently go back to ""
    and the only passing-test signal would be the degradation test above.
    """
    source = REPO_ROOT / "ml" / "classifier.py"
    tree = ast.parse(source.read_text())
    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert {"predict", "top_features", "explain"} <= defined, (
        f"ml/classifier.py no longer publishes the attribution interface: {sorted(defined)}"
    )


# --- gap 1: the cooldown has to hold BETWEEN processes -----------------------
#
# `last_action_time` was a plain module-level dict. Two controllers -- or one
# controller plus the eval harness calling `execute_action` directly -- each
# believed independently that it was within budget, and every row both wrote
# read `executed` and looked correct. The cooldown is now mirrored to a lease
# file per workload beside the decision log.
#
# A second process is simulated by clearing `last_action_time` and nothing else:
# that dict is precisely the state a second process would not share, so what
# survives the clear is what crosses the process boundary. Deliberately not by
# spawning a subprocess -- that would test the fork, not the rail.


def test_an_executed_action_leaves_a_lease_a_second_process_can_read(live):
    api = SpyAppsApi(replicas=1)
    assert run_once(api, "rolling_restart")[0] == "executed"
    lease = lease_for("nginx")
    assert lease.exists(), "the action left no shared record of itself"
    recorded = datetime.fromisoformat(lease.read_text())
    assert (datetime.now(timezone.utc) - recorded).total_seconds() < 60


def test_the_cooldown_holds_for_a_process_that_never_saw_the_action(live):
    """The gap itself. Nothing in this process's memory remembers the action;
    the rail has to hold anyway, or two controllers stack remediations on one
    workload at a 5.0% false-positive rate with neither aware of the other.
    """
    api = SpyAppsApi(replicas=1)
    assert run_once(api, "rolling_restart")[0] == "executed"

    controller.last_action_time.clear()  # a second process starts here

    assert controller.cooldown_active("nginx"), "a second process shares no cooldown"
    assert run_once(api, "rolling_restart")[0] == "blocked-cooldown"
    assert len(api.mutations) == 1, "the second process remediated the same workload again"


def test_a_direct_execute_action_caller_cannot_bypass_the_cooldown(live):
    """`evaluate_once` is not the only caller -- the eval harness and any
    head-to-head script call `execute_action`. A rail enforced one level up is
    bypassed by every one of them, which is the lesson the kill switch already
    taught this file.
    """
    api = SpyAppsApi(replicas=1)
    assert controller.execute_action(api, "nginx", "rolling_restart") == "executed"
    assert controller.execute_action(api, "nginx", "rolling_restart") == "blocked-cooldown"
    assert len(api.mutations) == 1


def test_the_later_of_memory_and_the_lease_wins(live):
    """Fail-safe combination: whichever source says "more recently" wins, so
    neither a stale lease nor a restarted process can SHORTEN a cooldown.
    """
    api = SpyAppsApi(replicas=1)
    run_once(api, "rolling_restart")

    # Lease fresh, memory ancient.
    controller.last_action_time["nginx"] = datetime.now(timezone.utc) - timedelta(days=1)
    assert controller.cooldown_active("nginx")

    # Memory fresh, lease ancient.
    controller.last_action_time["nginx"] = datetime.now(timezone.utc)
    lease_for("nginx").write_text(
        (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    )
    assert controller.cooldown_active("nginx")


def test_an_unreadable_lease_fails_closed_but_not_forever(live):
    """A corrupt lease must not fail OPEN -- ignoring it means no cooldown at
    all. It must not fail closed forever either: a permanent lockout caused by
    one bad file looks exactly like a working rail from the outside. Bounded to
    one window by falling back to the file's mtime.
    """
    controller.cooldown_dir().mkdir(parents=True, exist_ok=True)
    lease = lease_for("nginx")
    lease.write_text("this is not a timestamp")
    assert controller.cooldown_active("nginx"), "a corrupt lease failed open"

    old = datetime.now(timezone.utc) - timedelta(seconds=controller.COOLDOWN_SECONDS + 60)
    os.utime(lease, (old.timestamp(), old.timestamp()))
    assert not controller.cooldown_active("nginx"), "a corrupt lease became a lockout"


def test_a_lease_that_cannot_be_written_warns_and_still_remediates(live, capsys, monkeypatch):
    """The cluster write has already happened by the time the lease is recorded,
    so raising here would make `evaluate_once` log a successful remediation as
    `failed`. It must not be silent either: the cross-process guarantee is gone
    the moment this fails, and silence would degrade it back to exactly the
    process-local behaviour this code exists to replace.
    """
    def refuse(src, dst):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(controller.os, "replace", refuse)
    api = SpyAppsApi(replicas=1)
    assert run_once(api, "rolling_restart")[0] == "executed"

    assert "PROCESS-LOCAL ONLY" in capsys.readouterr().out, "a degraded rail was silent"
    assert controller.cooldown_active("nginx"), "the in-process cooldown degraded too"


def test_shadow_mode_neither_writes_nor_reads_a_lease():
    """DRY_RUN is on here. The shadow arm must describe what Prodrome would have
    done ON ITS OWN: if a live controller's leases could make shadow rows read
    `blocked-cooldown`, collect's control-arm comparison would be measuring the
    live arm's side effects through the back door.
    """
    api = ForbiddenApi()
    assert run_once(api, "rolling_restart")[0] == "dry-run"
    assert not lease_for("nginx").exists(), "the shadow arm wrote shared state"

    controller.cooldown_dir().mkdir(parents=True, exist_ok=True)
    lease_for("nginx").write_text(datetime.now(timezone.utc).isoformat())
    assert not controller.cooldown_active("nginx"), "shadow read another process's lease"
    assert run_once(api, "rolling_restart")[0] == "dry-run"


def test_the_lease_directory_follows_the_decision_log():
    """Derived from LOG_FILE, not named separately, so anything that redirects
    one redirects the other. Two files that disagree about what happened are
    worse than one.
    """
    assert controller.cooldown_dir() == controller.LOG_FILE.parent / "cooldowns"


@pytest.mark.parametrize("workload", ["nginx", "a/b", "..", "UPPER", "with space", ""])
def test_lease_names_are_path_safe(workload):
    """Percent-encoded rather than sanitised: sanitising maps two different
    workloads onto one file, and two workloads silently sharing a cooldown is a
    rail doing the wrong thing quietly.
    """
    name = controller._lease_name(workload)
    assert "/" not in name and "\\" not in name
    assert (controller.cooldown_dir() / name).parent == controller.cooldown_dir()


def test_every_workload_gets_its_own_lease_file():
    names = {controller._lease_name(w) for w in ["nginx", "a/b", "a%2Fb", "A/B"]}
    assert len(names) == 4, "two workloads would share one cooldown"


# --- gap 2: STOP must not be resolved against the working directory ----------


def test_the_committed_stop_file_is_anchored_to_the_repo_not_the_cwd():
    """Asserted against the parsed source, because the fixtures redirect
    STOP_FILE. `Path("STOP")` means an operator who creates STOP in the repo
    root while the controller runs from elsewhere gets a kill switch that
    silently does nothing -- the worst possible failure for this control.
    """
    tree = ast.parse(CONTROLLER_SOURCE.read_text())
    assigned = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "STOP_FILE" for t in node.targets)
    ]
    assert len(assigned) == 1, "STOP_FILE is assigned more or less than once at module scope"
    value = assigned[0]
    assert isinstance(value, ast.BinOp), "STOP_FILE is not built from an anchor path"
    assert isinstance(value.left, ast.Name) and value.left.id == "REPO_ROOT", (
        "STOP_FILE must be anchored to REPO_ROOT, not resolved against the cwd"
    )
    assert controller.REPO_ROOT == REPO_ROOT
    assert (REPO_ROOT / "STOP").is_absolute()


def test_the_kill_switch_is_found_from_any_working_directory(live, tmp_path, monkeypatch):
    """The behavioural half: the switch is wherever STOP_FILE points, no matter
    where the process was started.
    """
    anchored = tmp_path / "anchor"
    anchored.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(controller, "STOP_FILE", anchored / "STOP")
    monkeypatch.chdir(elsewhere)

    assert not controller.kill_switch_active()
    (anchored / "STOP").write_text("halt\n")
    assert controller.kill_switch_active(), "the kill switch moved with the working directory"
    assert controller.execute_action(ForbiddenApi(), "nginx", "rolling_restart") == (
        "blocked-kill-switch"
    )


def test_a_stop_in_the_working_directory_also_stops_the_controller(live, tmp_path, monkeypatch):
    """A kill switch should fail in the direction of stopping, so the rule is
    "ANY of these files", not "exactly this one". An operator who drops STOP in
    the directory they are standing in gets a stop too.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(controller, "EXTRA_STOP_PATHS", [Path("STOP")])

    assert not controller.kill_switch_active()
    (elsewhere / "STOP").write_text("halt\n")
    assert controller.kill_switch_active()
    assert controller.execute_action(ForbiddenApi(), "nginx", "scale_out") == (
        "blocked-kill-switch"
    )


def test_the_startup_report_names_every_path_that_stops_the_controller(stop_file):
    """A kill switch nobody can locate is not a kill switch. These paths used to
    be discoverable only by reading the source AND knowing the working directory.
    """
    report = controller.startup_report()
    for path in controller.kill_switch_paths():
        assert str(path) in report
    assert str(controller.STOP_FILE) in report
    assert "ENGAGED" in report, "STOP exists and the startup report did not say so"
    assert str(controller.LOG_FILE.resolve()) in report
    assert str(controller.cooldown_dir().resolve()) in report


def test_the_startup_report_says_open_when_there_is_no_stop_file():
    report = controller.startup_report()
    assert "ENGAGED" not in report and "open" in report
    assert "shadow" in report, "the startup report must say which mode it is in"


def test_run_loop_prints_the_startup_report():
    """Pinned by parsing rather than by running: `run_loop` does not return."""
    tree = ast.parse(CONTROLLER_SOURCE.read_text())
    loop = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_loop"
    )
    called = {
        inner.func.id for inner in ast.walk(loop)
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
    }
    assert "startup_report" in called, "the loop starts without saying where its rails are"


# --- gap 3: a global cap on actions across ALL workloads ---------------------
#
# The cooldown is per workload, so six workloads firing at once produced six
# simultaneous remediations, each individually within its rails -- and six
# workloads firing at once is the shape of a node-level problem, which is
# exactly when remediating everything at once is itself the destabilising event.


def test_the_fleet_budget_refuses_the_action_after_the_cap(live):
    api = SpyAppsApi(replicas=1)
    workloads = [f"w{i}" for i in range(controller.MAX_ACTIONS_PER_WINDOW + 1)]
    results = [run_once(api, "rolling_restart", workload=w)[0] for w in workloads]

    assert results[:-1] == ["executed"] * controller.MAX_ACTIONS_PER_WINDOW
    assert results[-1] == "blocked-rate-limit"
    assert len(api.mutations) == controller.MAX_ACTIONS_PER_WINDOW, (
        "more workloads were remediated than the fleet budget allows"
    )


def test_the_rate_limit_refusal_reaches_the_decision_log(live):
    """A refusal nobody can see is not a rail. This is the row that tells an
    operator the problem is bigger than any one workload.
    """
    api = SpyAppsApi(replicas=1)
    for i in range(controller.MAX_ACTIONS_PER_WINDOW + 1):
        run_once(api, "rolling_restart", workload=f"w{i}")

    row = log_rows()[-1]
    assert row["result"] == "blocked-rate-limit"
    assert row["fired"] == "True" and row["action"] == "rolling_restart"
    assert row["mode"] == "live"


def test_a_rate_limited_refusal_records_no_cooldown(live):
    """Same lockout argument as the kill switch: a blocked action that recorded
    a timestamp would push its own window forward on every firing.
    """
    api = SpyAppsApi(replicas=1)
    blocked = f"w{controller.MAX_ACTIONS_PER_WINDOW}"
    for i in range(controller.MAX_ACTIONS_PER_WINDOW + 1):
        run_once(api, "rolling_restart", workload=f"w{i}")

    assert blocked not in controller.last_action_time
    assert not lease_for(blocked).exists()
    assert not controller.cooldown_active(blocked)


def test_the_fleet_budget_reopens_when_the_window_passes(live):
    api = SpyAppsApi(replicas=1)
    for i in range(controller.MAX_ACTIONS_PER_WINDOW):
        run_once(api, "rolling_restart", workload=f"w{i}")
    assert controller.rate_limit_active()

    for i in range(controller.MAX_ACTIONS_PER_WINDOW):
        rewind_cooldown(f"w{i}", controller.RATE_LIMIT_WINDOW_SECONDS + 1)

    assert not controller.rate_limit_active(), "a rail that never expires is broken too"
    assert run_once(api, "rolling_restart", workload="fresh")[0] == "executed"


def test_the_fleet_budget_counts_another_process_actions(live):
    """The cap is fleet-wide across PROCESSES too, or two controllers each get
    their own budget and the real cap is whatever number you first thought of,
    times the number of controllers.
    """
    for i in range(controller.MAX_ACTIONS_PER_WINDOW):
        controller.record_action(f"other-{i}")
    controller.last_action_time.clear()  # a second process starts here

    assert controller.rate_limit_active()
    assert run_once(SpyAppsApi(replicas=1), "rolling_restart", workload="mine")[0] == (
        "blocked-rate-limit"
    )


def test_a_direct_execute_action_caller_cannot_bypass_the_fleet_budget(live):
    api = SpyAppsApi(replicas=1)
    for i in range(controller.MAX_ACTIONS_PER_WINDOW):
        assert controller.execute_action(api, f"w{i}", "rolling_restart") == "executed"
    assert controller.execute_action(api, "one-too-many", "rolling_restart") == (
        "blocked-rate-limit"
    )
    assert len(api.mutations) == controller.MAX_ACTIONS_PER_WINDOW


def test_actions_that_issue_no_write_do_not_spend_the_fleet_budget(live):
    """`alert_only` and a scale_out already at the cap touch nothing, so they
    must not consume a budget that real remediations need.
    """
    api = SpyAppsApi(replicas=controller.MAX_REPLICAS)
    for i in range(controller.MAX_ACTIONS_PER_WINDOW + 2):
        assert controller.execute_action(api, f"w{i}", "alert_only") == "no-op"
        assert controller.execute_action(api, f"s{i}", "scale_out") == "at-max"
    assert controller.recent_actions() == []
    assert not controller.rate_limit_active()


def test_the_kill_switch_and_the_cooldown_both_outrank_the_fleet_budget(live, stop_file):
    """Ordering, pinned: when several rails would all refuse, the log gets the
    most specific reason, because that is the one an operator can act on.
    """
    for i in range(controller.MAX_ACTIONS_PER_WINDOW):
        controller.record_action(f"w{i}")
    assert controller.rate_limit_active()
    assert run_once(ForbiddenApi(), "rolling_restart", workload="nginx")[0] == (
        "blocked-kill-switch"
    )

    stop_file.unlink()
    controller.record_action("nginx")
    assert run_once(ForbiddenApi(), "rolling_restart", workload="nginx")[0] == (
        "blocked-cooldown"
    )


def test_the_rate_limit_window_matches_the_cooldown_window():
    """Load-bearing, not cosmetic. One lease file per workload is an EXACT count
    of actions in the window only because a workload cannot act twice inside its
    own cooldown. A window longer than the cooldown silently undercounts.
    """
    assert controller.RATE_LIMIT_WINDOW_SECONDS == controller.COOLDOWN_SECONDS


def test_the_fleet_budget_is_invisible_in_shadow_mode():
    """DRY_RUN is on. Same reasoning as the cooldown: the shadow arm must not
    observe a budget it did not spend. The consequence, which collect already
    has in writing, is that shadow action counts are an UPPER BOUND on live ones
    rather than like-for-like.
    """
    controller.cooldown_dir().mkdir(parents=True, exist_ok=True)
    for i in range(controller.MAX_ACTIONS_PER_WINDOW + 3):
        lease_for(f"other-{i}").write_text(datetime.now(timezone.utc).isoformat())

    assert not controller.rate_limit_active()
    assert run_once(ForbiddenApi(), "rolling_restart")[0] == "dry-run"


# --- gap 4: `result="not-fired"` on a row where `fired=True` -----------------
#
# Announced as a decision-log contract change (cluster/interfaces, labelled for
# diagnosis and collect) before being made, because `result` values may be
# filtered downstream.


def test_a_fired_tick_the_policy_abstained_on_is_not_called_not_fired(live):
    """The mislabel itself: `fired=True` beside `result=not-fired`, two columns
    flatly contradicting each other, on one of the most common rows in the log.
    """
    run_once(ForbiddenApi(), "nothing")
    row = log_rows()[-1]
    assert row["fired"] == "True"
    assert row["result"] == "abstained"
    assert row["action"] == "nothing"


def test_not_fired_now_means_exactly_fired_is_false(live):
    controller.evaluate_once(
        ForbiddenApi(), "nginx", one_tick(), FakeDetector(fired=False),
        FakeClassifier(label_for("rolling_restart"), 1.0), full_history(),
    )
    row = log_rows()[-1]
    assert row["fired"] == "False"
    assert row["result"] == "not-fired"


def test_both_abstention_paths_log_abstained(live):
    """Two different ways to abstain -- below the policy's confidence floor, and
    too little history to ask the classifier at all -- and both happen on a tick
    that FIRED.
    """
    low = policy.POLICY[label_for("rolling_restart")]["min_confidence"] - 0.01
    assert run_once(ForbiddenApi(), "rolling_restart", confidence=low)[0] == "abstained"

    result = controller.evaluate_once(
        ForbiddenApi(), "nginx", one_tick(), FakeDetector(),
        FakeClassifier(label_for("rolling_restart"), 1.0),
        full_history()[: WINDOW_SIZE - 1],
    )
    assert result == "abstained"


def test_no_row_in_the_log_ever_claims_not_fired_while_fired_is_true(live):
    """Swept over every action and both abstention paths at once, because this
    is a consistency property of the log rather than of any single branch.
    """
    for action in ALL_ACTIONS:
        run_once(SpyAppsApi(replicas=1), action, workload=f"w-{action}")
        run_once(SpyAppsApi(replicas=1), action, workload=f"lo-{action}", confidence=0.0)
    controller.evaluate_once(
        ForbiddenApi(), "short", one_tick(), FakeDetector(),
        FakeClassifier(label_for("rolling_restart"), 1.0),
        full_history()[: WINDOW_SIZE - 1],
    )
    controller.evaluate_once(
        ForbiddenApi(), "quiet", one_tick(), FakeDetector(fired=False),
        FakeClassifier(label_for("rolling_restart"), 1.0), full_history(),
    )

    rows = log_rows()
    assert rows
    for row in rows:
        if row["result"] == "not-fired":
            assert row["fired"] == "False", f"contradictory row: {row}"
        if row["fired"] == "False":
            assert row["result"] == "not-fired", f"contradictory row: {row}"
