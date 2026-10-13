"""The stages wired together, over data/samples/.

The unit tests ask whether each stage does what it claims. These ask whether the
stages fit: whether what the classifier returns is something the policy table can
consume, and whether what policy returns is an action a controller could
actually execute. Nothing here pins an accuracy number -- a mismatch between
stages is a defect at any accuracy.

The executing stage used to be unreachable. control/controller.py did
`from kubernetes import client, config` at module scope, which made the whole
module unimportable without the package, so this file checked the controller by
parsing its source with `ast` and said so: "a weaker check than calling it, and
the strongest one available from a fresh clone with no cluster."

That is no longer true, and this file no longer says it. The kubernetes import
moved inside `connect_to_kubernetes()` (cluster, 2026-10-03), so
`control.controller` imports fine with no kubernetes package, no Docker and no
cluster, and the three checks that parsed source now CALL the thing they
describe. That matters beyond tidiness: a source-parsing test fails for the
wrong reasons. The decision-log check here walked `log_decision`'s body for a
list of string literals, so hoisting the header into a named constant made it
fail with "could not find the header row" -- a failure unrelated to the defect
it documented, while its strict xfail still looked satisfied. Each replacement
below was verified by reintroducing the original bug locally and watching the
new test go red.

What is still a source check, deliberately: `DRY_RUN` must stay the literal
`True` in the committed file, and no behavioural test can see a value that has
been monkeypatched back. control/test_safety_rails.py owns that assertion.

conftest's FakeController also stays. It is no longer standing in for an
unimportable module; it is the cross-stage assertion that an action outside
ACTION_VOCABULARY is a loud failure, which is a different question from what the
real controller does with the actions it already knows.
"""

import builtins
import csv
import importlib
import sys
from datetime import datetime

import pytest

from control import controller
from control import policy
from dashboard import terminal
from ml.detector import Detector
from ml.features import METRICS, WINDOW_SIZE, windows

from conftest import ACTION_VOCABULARY, DECISION_LOG_COLUMNS, REPO_ROOT, StubClassifier


class SpyAppsApi:
    """Records every cluster call and returns plausible objects.

    "The controller did something" is asserted against this object rather than
    inferred from a return value, so a branch that returns the right string
    while touching nothing cannot pass.
    """

    def __init__(self, replicas: int = 1):
        self._replicas = replicas
        self.calls: list[tuple[str, str]] = []

    def read_namespaced_deployment(self, name, namespace):
        self.calls.append(("read", name))
        return SimpleDeployment(self._replicas)

    def patch_namespaced_deployment_scale(self, name, namespace, body):
        self.calls.append(("scale", name))

    def patch_namespaced_deployment(self, name, namespace, body):
        self.calls.append(("restart", name))


class SimpleDeployment:
    def __init__(self, replicas):
        self.spec = type("Spec", (), {"replicas": replicas})()


@pytest.fixture
def live_controller(tmp_path, monkeypatch):
    """The real controller, wired to temporaries instead of a cluster.

    DRY_RUN off so the action branches are reachable at all -- the dry-run
    branch returns before any of them, which is exactly how the original wiring
    bug stayed invisible. STOP, the decision log and the cooldown map are
    redirected per test so nothing leaks between them, and monkeypatch puts
    DRY_RUN back to True afterwards.
    """
    monkeypatch.setattr(controller, "DRY_RUN", False)
    monkeypatch.setattr(controller, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(controller, "LOG_FILE", tmp_path / "decisions" / "log.csv")
    monkeypatch.setattr(controller, "last_action_time", {})
    return controller


# --- the loop end to end -----------------------------------------------------

def test_the_whole_loop_runs_over_the_samples_and_decides_something(decisions):
    """detector -> classifier -> policy -> controller, on committed data only.

    Guards against the version of this suite that passes because nothing happens:
    if the detector gate produced no fired ticks, every assertion below would be
    vacuously true. The sample fixture is synthetic and the offline replay path
    uses score_only(), so it does fire here -- unlike the live path on real data
    (see tests/test_units.py's desensitization test).
    """
    # `fake`, not `controller`: this module now imports the real control.controller
    # as well, and the two must not be confused for one another.
    rows, fake = decisions
    assert rows, "the detector fired on no sample tick; every cross-stage test below is vacuous"
    assert fake.executed, "no action reached the controller"


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


# --- where the stages meet ---------------------------------------------------

def test_the_controller_imports_with_no_kubernetes_client_installed(monkeypatch):
    """The precondition every call-based test in this file rests on.

    `control/controller.py` used to do `from kubernetes import client, config` at
    module scope, which made the rails, the action wiring and the decision-log
    format untestable offline -- they had to be checked by parsing the source,
    and this file said so. Cluster moved the import inside
    connect_to_kubernetes(). If it ever moves back, CI (which installs no
    kubernetes client) would fail at collection with an import error that looks
    like an environment problem; this says what it actually is.

    The local venv HAS the kubernetes package, so the absence is simulated rather
    than assumed -- otherwise this test would pass here and tell us nothing about
    the machine that matters.
    """
    for name in [m for m in sys.modules if m == "kubernetes" or m.startswith("kubernetes.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "control.controller", raising=False)

    real_import = builtins.__import__

    def without_kubernetes(name, *args, **kwargs):
        if name == "kubernetes" or name.startswith("kubernetes."):
            raise ImportError("No module named 'kubernetes' (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_kubernetes)

    offline = importlib.import_module("control.controller")
    assert callable(offline.execute_action)
    assert offline.DECISION_LOG_COLUMNS == DECISION_LOG_COLUMNS

    # ...and the one function that genuinely needs a cluster still says so.
    with pytest.raises(ImportError):
        offline.connect_to_kubernetes()


@pytest.mark.parametrize("action", sorted({e["action"] for e in policy.POLICY.values()}))
def test_the_controller_can_execute_every_action_the_policy_table_emits(live_controller, action):
    """The mismatch this file exists to find — and it was real.

    execute_action() used to compare `action` against the single literal
    'restart' while policy.decide() returns scale_out, rolling_restart,
    alert_only or nothing. The intersection was EMPTY: wired up as written,
    every decision would have logged as unknown-action and no remediation would
    ever have been applied — and DRY_RUN=True masked it completely, because the
    dry-run branch returns before the comparison. It would have surfaced on the
    day someone flipped DRY_RUN off, which is the worst possible time.

    Cluster fixed it (origin/main: "Fixed controller wiring bug"), and the
    recorded strict xfail is what caught the fix landing.

    This used to read execute_action()'s string literals with `ast`, because the
    module could not be imported without the kubernetes client. It now CALLS
    execute_action with DRY_RUN off and a spy API, which is strictly stronger in
    two ways the parse could not reach: a literal that appears in a docstring or
    an error message would have satisfied the old check, and a branch that
    recognises an action without issuing the write it names still fails here.

    `action` is parametrised off POLICY, so a new lever from diagnosis arrives as
    a new red case rather than as a live "unknown-action" in the decision log.
    """
    api = SpyAppsApi()
    result = live_controller.execute_action(api, "nginx", action)

    assert result != "unknown-action", (
        f"policy emits {action!r} but the controller does not recognise it; "
        f"every such decision would log as unknown-action and remediate nothing"
    )
    if action == "nothing":
        assert result == "no-op"
        assert api.calls == [], "'nothing' must never reach the cluster"
    elif action in ("scale_out", "rolling_restart"):
        # Recognising the action is not the same as acting on it.
        assert result == "executed"
        assert api.calls, f"{action} returned 'executed' without calling the API"


def test_the_writer_and_the_reader_agree_on_where_the_decision_log_lives():
    """SETUP.md §8 names data/decisions/log.csv. Both halves must use it.

    A real defect, found by diagnosis on 2026-10-06: the controller moved to
    data/decisions/log.csv and the dashboard kept its own
    `Path("control/decisions.csv")`. Nothing failed. The dashboard simply
    rendered "waiting for decisions" forever while the controller wrote good
    rows somewhere else -- and an operator watching a blank dashboard during an
    incident would reasonably conclude nothing was happening. A matching column
    list on two files that do not read the same file is not a contract.

    Diagnosis fixed it by importing the path from the controller rather than
    restating it, so the divergence is now impossible rather than merely
    detectable. This pins both facts: they agree, AND what they agree on is what
    the contract names. The second half matters -- aliasing alone would be
    satisfied by both being wrong together, which is the failure this pair of
    files already had once.

    Deliberately takes no fixture. `live_controller` redirects LOG_FILE to a
    temporary, so this assertion made inside it would be checking the test's own
    setup rather than the committed value.

    This asserted `controller.LOG_FILE == Path("data/decisions/log.csv")` until
    2026-10-12, which pinned the literal rather than the contract and blocked a
    fix: cluster anchored STOP_FILE to the repo root after finding that a
    controller started from anywhere but the repo root got a kill switch that
    silently did nothing, could not give LOG_FILE the same treatment, and said so
    in its handoff -- "tests/test_integration.py:311 asserts ... and tests/ is
    not mine." SETUP.md §8 names a FILE IN THIS REPO, not a string to resolve
    against whatever directory an operator was standing in, so the equality was
    asserting more than the contract says. It now pins the two halves of the
    contract that are actually load-bearing -- the path is called
    data/decisions/log.csv, and the writer and the reader name the same file --
    and leaves the anchoring free. `REPO_ROOT / "data" / "decisions" / "log.csv"`
    satisfies this; so does today's relative value. The gap between them is the
    strict xfail below, which is where that finding lives.
    """
    assert controller.LOG_FILE.parts[-3:] == ("data", "decisions", "log.csv"), (
        f"SETUP.md §8 names data/decisions/log.csv; the controller writes "
        f"{controller.LOG_FILE}"
    )
    assert terminal.DECISIONS_LOG == controller.LOG_FILE, (
        "the dashboard reads a path the controller does not write"
    )
    # Equality of two Path objects is not yet agreement about a file: while the
    # value is relative, "the same path" means "the same path RELATIVE TO EACH
    # PROCESS'S CWD", and the dashboard is routinely run from a second terminal.
    # Trivially true today because this is one process; it is here so that it
    # keeps being checked once the value is anchored and the two can differ.
    assert terminal.DECISIONS_LOG.resolve() == controller.LOG_FILE.resolve()


def test_the_decision_log_is_the_same_file_wherever_the_controller_was_started(
    tmp_path, monkeypatch
):
    """The defect the assertion above used to pin in place.

    Same shape as the STOP bug cluster fixed on 2026-10-12: a constant written as
    a relative Path is not a location, it is a location *plus* an assumption
    about the working directory, and nothing in the process checks that
    assumption. `startup_report()` prints LOG_FILE resolved, which makes it
    visible to someone who reads the startup banner, but visible is not the same
    as correct.

    Three consequences, in increasing order of how much they matter:

    1. The audit trail. A controller started from elsewhere writes
       data/decisions/log.csv under THAT directory, and the dashboard started
       from the repo root reads the repo's copy and renders "waiting for
       decisions" -- the exact failure that
       test_the_writer_and_the_reader_agree_on_where_the_decision_log_lives
       exists to prevent, except that aliasing the constant cannot prevent this
       half of it. Two processes holding the same relative Path still read and
       write two different files.
    2. The eval harness. `eval/recovery.py --decisions` defaults to the same
       relative string, so a run from a different directory silently scores an
       empty or stale log instead of failing.
    3. The cooldown. `cooldown_dir()` is `LOG_FILE.parent / "cooldowns"`, so the
       cross-process lease namespace is cwd-dependent too. Two controllers
       started from different directories share no cooldown and no rate-limit
       budget, each believes it is within budget, and every row in both logs
       reads `executed`. That is a safety rail failing OPEN and silently, which
       is a strictly worse outcome than a lost audit trail -- and it is the same
       argument cluster already accepted for STOP.

    Unlike STOP, this should NOT also honour a cwd-relative copy. A kill switch
    fails safe by stopping, so "any of these files" is right for it; a log that
    is written in two places at once is just a split audit trail, and the rails
    reading a split lease directory is the failure in (3). One anchored location.
    """
    repo_root_answer = controller.LOG_FILE.resolve()
    monkeypatch.chdir(tmp_path)
    elsewhere_answer = controller.LOG_FILE.resolve()

    assert elsewhere_answer == repo_root_answer, (
        f"the decision log is {repo_root_answer} when the controller is started "
        f"from the repo root and {elsewhere_answer} when it is started from "
        f"{tmp_path} -- an operator gets an audit trail nobody reads, the "
        f"dashboard shows nothing, and cooldown_dir() splits the lease namespace "
        f"with it so two controllers share no rail"
    )


def test_the_cooldown_leases_follow_the_decision_log_they_are_derived_from(live_controller):
    """`cooldown_dir()` is `LOG_FILE.parent / "cooldowns"`, and cluster chose that
    derivation specifically so this file's fixtures isolate the leases for free.

    Pinned because it is a cross-seat assumption that nothing else states: if the
    leases were ever given their own constant, every test here that redirects
    LOG_FILE would start writing real lease files into the developer's working
    tree and reading each other's cooldowns, and the first symptom would be
    flakiness in an unrelated test. Also the guarantee that the fix requested in
    the xfail above does not quietly change what these fixtures isolate.
    """
    assert live_controller.cooldown_dir() == live_controller.LOG_FILE.parent / "cooldowns"

    live_controller.record_action("nginx")
    written = sorted(p.name for p in live_controller.cooldown_dir().iterdir())
    assert written == ["nginx.lease"], f"leases landed elsewhere: {written}"
    assert REPO_ROOT not in live_controller.cooldown_dir().resolve().parents, (
        "a test just wrote a cooldown lease inside the repo"
    )


def test_the_controller_writes_the_decision_log_columns_the_contract_names(live_controller):
    """SETUP.md §7 pins the decision log at ts, workload, detector_score, fired,
    predicted_class, confidence, top_features, action, result, mode -- and says
    changing it requires telling everyone.

    This previously wrote `timestamp` instead of `ts` with no top_features and no
    mode at all, which made ground rule 4's control-arm comparison unanswerable
    from the log: you could not tell what Prodrome did from what it merely would
    have done. Fixed, along with the path (SETUP.md §8 names
    data/decisions/log.csv, not control/decisions.csv).

    Now asserted by CALLING log_decision and reading the file back. Two earlier
    versions of this test parsed the source and both failed for the wrong
    reason: one walked log_decision()'s body for a string list, so hoisting the
    header into a named constant made it fail with "could not find the header
    row"; its replacement pinned the constant, which cannot tell whether the
    writer uses it or keeps a second copy. The written file answers both at once,
    and is what the dashboard and the eval harness actually read.
    """
    live_controller.log_decision(
        workload="nginx",
        detector_score=4.2,
        fired=True,
        predicted_class="CPU_HOG",
        confidence=0.91,
        action="scale_out",
        result="executed",
        top_features="cpu_cores_last=+0.21",
    )

    with live_controller.LOG_FILE.open(newline="") as f:
        header, row = list(csv.reader(f))

    assert header == DECISION_LOG_COLUMNS
    assert len(row) == len(DECISION_LOG_COLUMNS), "a row the header cannot name"

    cells = dict(zip(header, row))
    datetime.fromisoformat(cells["ts"])  # `ts`, and an actual timestamp in it
    assert cells["workload"] == "nginx"
    assert cells["top_features"] == "cpu_cores_last=+0.21"
    # DRY_RUN is False in this fixture, so `mode` must say so rather than
    # defaulting to a literal -- a row may never claim to be shadow while live.
    assert cells["mode"] == "live"


def test_the_dashboard_reads_the_decision_log_columns_the_contract_names(live_controller):
    """The reading half of the same divergence, now fixed.

    dashboard/terminal.py declared COLUMNS without top_features or mode, mirroring
    whatever the controller happened to write rather than the contract -- so both
    halves drifted together and the divergence was invisible from either side. It
    also rendered every cell with `row.get(col, "")`, which made a missing column
    display as blanks; an operator could not tell "we took no action" from "the
    column is gone". It now surfaces missing columns in the table caption.

    Parsing the source for `COLUMNS = [...]` could only ever pin the first half.
    Importing the module and calling load_recent()/render() pins both, and closes
    the loop: the controller writes a row above and the dashboard reads THAT file
    rather than a frame this test invented. (This is why CI installs `rich`.)

    The parse would also have broken outright on 2026-10-06, when diagnosis
    replaced the literal with `COLUMNS = DECISION_LOG_COLUMNS` imported from the
    controller. The helper would have raised "dashboard/terminal.py has no
    COLUMNS assignment" -- red because the file got MORE correct. Second
    occurrence of the failure mode this file's own history records; the reason
    none of these checks parse source any more.

    Still worth asserting after the aliasing, because conftest's
    DECISION_LOG_COLUMNS is an independent copy pinned from SETUP.md §7: the
    alias makes the two halves agree with each other, this makes them agree with
    the contract.
    """
    assert terminal.COLUMNS == DECISION_LOG_COLUMNS

    live_controller.log_decision(
        workload="nginx", detector_score=4.2, fired=True, predicted_class="CPU_HOG",
        confidence=0.91, action="scale_out", result="executed", top_features="cpu_cores_last",
    )
    frame = terminal.load_recent(live_controller.LOG_FILE)
    assert frame is not None, "the dashboard cannot read the file the controller just wrote"
    assert list(frame.columns) == DECISION_LOG_COLUMNS

    rendered = terminal.render(frame)
    assert not getattr(rendered, "caption", None), (
        f"dashboard reports missing columns on a contract-compliant log: {rendered.caption}"
    )

    # And the other half of the defect: a column that IS missing must be named,
    # not quietly rendered as an empty cell.
    degraded = terminal.render(frame.drop(columns=["mode", "top_features"]))
    caption = str(getattr(degraded, "caption", "") or "")
    assert "mode" in caption and "top_features" in caption, (
        "a missing decision-log column renders as blanks instead of being reported"
    )


# --- a placeholder presented as a prediction ---------------------------------

class AlwaysFiringDetector:
    """A detector that fires on every tick, so the branch under test is reached.

    The real detector's firing depends on history and on the data; the question
    here is what the controller logs ONCE it has fired, so firing is made a
    given rather than arranged.
    """

    def __init__(self, score: float = 4.2):
        self.score_value = score

    def score(self, workload, values):
        return self.score_value, True


def _one_tick():
    return {metric: 1.0 for metric in METRICS}


def _history(depth):
    return [[1.0] * len(METRICS) for _ in range(depth)]


@pytest.mark.xfail(
    reason="evaluate_once() initialises predicted_class, confidence = "
           "('NORMAL', 1.0) and logs that pair on a fired tick whose history is "
           "shorter than WINDOW_SIZE, without calling the classifier. Raised by "
           "cluster, confirmed here, posted to crosstalk/questions for diagnosis "
           "and cluster 2026-10-12. Strict: when a row that no classifier "
           "produced stops claiming a prediction, this XPASSes and the marker "
           "comes off rather than rotting into a comment.",
    strict=True,
)
def test_a_decision_row_never_reports_a_prediction_no_classifier_made(live_controller):
    """SETUP.md §7 gives the decision log `predicted_class` and `confidence`, and
    the dashboard and the eval harness read it by name. A row may only fill those
    cells with something a model actually returned.

    The defect: `evaluate_once` seeds `predicted_class, confidence = ("NORMAL",
    1.0)` before the gate, and on a fired tick with fewer than WINDOW_SIZE ticks
    of history -- which is every tick of the first five minutes after any
    controller start, and after every restart -- it falls through to
    `log_decision` with those values untouched. `classifier.calls` is 0. The row
    says NORMAL at confidence 1.0.

    Why that is not merely untidy: the two rows this test writes are produced by
    two completely different paths -- one where the shipped model was asked and
    answered, one where it was never loaded -- and they are IDENTICAL in every
    column except `ts`. Same fired, same detector_score, same predicted_class,
    same confidence, same action, same result, same mode. An operator looking at
    the dashboard sees "NORMAL 1.00" and reads it as the model being certain
    nothing is wrong, at the exact moment the model has not been consulted at
    all. Anyone computing diagnosis accuracy or a confidence distribution over
    this column counts warm-up rows as perfect-confidence NORMAL predictions,
    which is the one value that most flatters the model.

    `result="abstained"` does not rescue it. Cluster introduced `abstained`
    precisely because `not-fired` on a `fired=True` row was two columns
    contradicting each other, and it covers three different things at once:
    below the confidence floor, an unmapped class, and not enough history. The
    first two are the model's answer; the third is its absence. The column
    cannot tell them apart, and `policy.decide("NORMAL", 1.0)` returns "nothing"
    too, so `action` cannot either.

    Deliberately asserts DISCRIMINABILITY rather than an encoding. The fix is
    control/controller.py's and ml/classifier.py's to choose -- empty cells, a
    sentinel class outside the model's label space, an empty confidence, a
    `predicted_class` of "" with the reason carried in `result` -- and this test
    passes for any of them. Pinning a particular spelling here would be
    renegotiating someone else's contract rather than asserting it. What it will
    not accept is the two rows staying interchangeable. "UNKNOWN" would be the
    one wrong answer, since policy.py already uses it for "the model answered,
    below the floor"; see the note on crosstalk.
    """
    detector = AlwaysFiringDetector()

    asked = StubClassifier("NORMAL", 1.0)
    live_controller.evaluate_once(
        SpyAppsApi(), "nginx", _one_tick(), detector, asked, _history(WINDOW_SIZE),
    )
    assert asked.calls == 1, "the full-history branch did not consult the classifier"

    never_asked = StubClassifier("NORMAL", 1.0)
    live_controller.evaluate_once(
        SpyAppsApi(), "nginx", _one_tick(), detector, never_asked, _history(WINDOW_SIZE - 1),
    )
    assert never_asked.calls == 0, (
        "the short-history branch consulted the classifier; this test no longer "
        "describes the situation it was written for"
    )

    with live_controller.LOG_FILE.open(newline="") as f:
        header, genuine_row, placeholder_row = list(csv.reader(f))
    genuine = dict(zip(header, genuine_row))
    placeholder = dict(zip(header, placeholder_row))

    assert genuine["fired"] == placeholder["fired"] == "True"

    prediction_cells = ("predicted_class", "confidence", "top_features")
    assert any(genuine[c] != placeholder[c] for c in prediction_cells), (
        "a fired tick with too little history logged "
        f"predicted_class={placeholder['predicted_class']!r} "
        f"confidence={placeholder['confidence']!r} without ever calling the "
        "classifier, which is byte-for-byte what the classifier's own answer "
        "looks like: "
        + ", ".join(f"{c}={genuine[c]!r}" for c in prediction_cells)
        + ". The log cannot distinguish a confident NORMAL from a question "
          "nobody asked, and every reader of that column is counting the "
          "placeholder as a prediction."
    )
