"""Offline tests for the two-arm injection runner.

`collect/twoarm.py` is the only part of the control-arm harness that needs a
cluster, so the whole point of these tests is that they drive it WITHOUT one:
every cluster call enters through an injected callable, so the scheduler, the
barrier, the skew accounting, the half-pair handling and the file writers are
all exercised against fakes. `kubectl` is never invoked and `time.sleep` is
never called.

Same precedent as `ml/test_restart_suppression.py` (a test beside the code it
checks, not in the testing seat's `tests/`), written as pytest functions so
`pytest collect -q` collects them, with a `main()` for standalone use.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect import chaos, twoarm  # noqa: E402
from collect.twoarm import ARMS, PairSpec  # noqa: E402
from eval.recovery import CONTROL_ARM, HEALTH_COLUMNS, PAIRS_COLUMNS, TREATMENT_ARM  # noqa: E402

T0 = pd.Timestamp("2026-10-03T20:00:00Z")


def spec(fault: str = "POD_KILL", workload: str = "redis",
         pattern: str = "none") -> PairSpec:
    return PairSpec(pair_id=f"{fault}_{pattern}_{workload}_001", fault_type=fault,
                    workload=workload, pattern=pattern, duration_s=300)


class FakeInjector:
    """Records which namespaces were hit and when, and can be told to fail one
    arm or to stall one arm."""

    def __init__(self, fail_ns: str | None = None, stall: dict | None = None):
        self.calls: list[tuple[str, str]] = []
        self.fail_ns = fail_ns
        self.stall = stall or {}
        self._lock = threading.Lock()

    def __call__(self, namespace: str, s: PairSpec) -> dict:
        if namespace in self.stall:
            threading.Event().wait(self.stall[namespace])
        with self._lock:
            self.calls.append((namespace, s.fault_type))
        if namespace == self.fail_ns:
            raise RuntimeError("kubectl exploded")
        return dict(start_ts=T0, end_ts=T0 + pd.Timedelta(seconds=300),
                    failure_ts=None, restarts_delta=1)


def fake_probe(namespace: str, workload: str) -> dict:
    return dict(ready_replicas=1, desired_replicas=1, restarts_total=0,
                probe_ok=True, probe_latency_ms=12.3)


# --- simultaneity -----------------------------------------------------------

def test_both_arms_are_injected_and_only_those_two():
    """Identical fault, both namespaces, nothing else touched. If this hit one
    namespace twice or a third one, the 'comparison' would not be one."""
    inj = FakeInjector()
    res = twoarm.inject_both(spec(), injector=inj)
    assert sorted(ns for ns, _ in inj.calls) == ["control", "prodrome"]
    assert set(res) == set(ARMS)
    assert all(r.inject_ok for r in res.values())


def test_both_arms_receive_the_identical_fault_spec():
    """The same PairSpec object goes to both arms, so 'identical faults' is a
    property of the code rather than a claim in a docstring. Reusing
    collect/chaos.py's own plan functions is the other half of that -- a second
    injector would be a second fault definition."""
    inj = FakeInjector()
    twoarm.inject_both(spec("MEMORY_LEAK", "postgres", "ramp"), injector=inj)
    assert {f for _, f in inj.calls} == {"MEMORY_LEAK"}
    assert len(inj.calls) == 2


def test_skew_is_measured_not_assumed():
    """We never claim simultaneity; we record what we achieved. A clock that
    hands out increasing timestamps stands in for two kubectl round-trips that
    did not land together."""
    ticks = iter([T0, T0 + pd.Timedelta(seconds=3)])
    lock = threading.Lock()

    def clock():
        with lock:
            try:
                return next(ticks)
            except StopIteration:
                return T0 + pd.Timedelta(seconds=3)

    inj = FakeInjector()

    def bare(namespace, s):           # no start_ts, so our own clock is used
        inj(namespace, s)
        return {}

    res = twoarm.inject_both(spec(), injector=bare, clock=clock)
    assert twoarm.skew_seconds(res) == 3.0


def test_skew_is_none_for_a_half_pair():
    """A half-pair has no meaningful skew, and `None` is what
    eval.recovery.attest_pair_skew reports as "cannot show the arms were
    simultaneous" -- rather than a reassuring 0.0, which is what a `min`/`max`
    over one timestamp would give."""
    res = {TREATMENT_ARM: twoarm.ArmResult(TREATMENT_ARM, "prodrome", start_ts=T0),
           CONTROL_ARM: twoarm.ArmResult(CONTROL_ARM, "control", start_ts=None)}
    assert twoarm.skew_seconds(res) is None


def test_one_arm_failing_does_not_abort_the_other():
    """REGRESSION, a real race found while testing this module.

    The error handler used to call barrier.abort() to fail the other arm fast.
    CPython's Barrier re-checks its state as a released thread resumes, so that
    abort -- issued after this arm's own wait() had already returned -- could
    still raise BrokenBarrierError in the OTHER arm before it reached its
    injector. Observed: control's injector raised, and prodrome then never
    injected AT ALL and reported BrokenBarrierError, turning a one-arm failure
    into a zero-arm pair non-deterministically.

    The surviving arm must keep its real measurement; the pair is then a
    half-pair, which eval/recovery.py excludes from the delta while still
    counting the arm that worked.
    """
    inj = FakeInjector(fail_ns="control")
    res = twoarm.inject_both(spec(), injector=inj)
    assert res[TREATMENT_ARM].inject_ok, res[TREATMENT_ARM].error
    assert res[TREATMENT_ARM].start_ts is not None
    assert not res[CONTROL_ARM].inject_ok
    assert "kubectl exploded" in res[CONTROL_ARM].error
    assert res[CONTROL_ARM].end_ts is not None     # still has a row to write
    assert sorted(ns for ns, _ in inj.calls) == ["control", "prodrome"]


def test_injector_reported_timestamps_win_over_our_own():
    """collect/chaos.py knows when stress-ng really started; our clock only
    knows when we called it. The injector's own start_ts/end_ts are the better
    witness and must take precedence."""
    res = twoarm.inject_both(spec(), injector=FakeInjector())
    assert res[TREATMENT_ARM].start_ts == T0
    assert res[TREATMENT_ARM].restarts_delta == 1


# --- health sampling --------------------------------------------------------

def test_one_sampler_covers_both_arms_per_tick():
    """One loop for both arms, not one per arm: a single loop keeps the two
    arms sampled within a few hundred ms of each other for the whole
    experiment, so a recovery-time difference cannot come from one arm having
    been watched more closely than the other."""
    s = twoarm.HealthSampler(workloads=["redis", "nginx"])
    s.probe_fn = fake_probe
    s.sample_once()
    df = s.frame()
    assert len(df) == 4                                   # 2 arms x 2 workloads
    assert set(df["arm"]) == set(ARMS)
    assert list(df.columns) == HEALTH_COLUMNS


def test_a_probe_that_throws_is_recorded_as_not_serving():
    """A probe that raises IS health information: the API server or the pod is
    not answering. Dropping the sample would turn an outage into a gap in the
    data, and a gap reads as 'nothing happened' -- which would hide exactly the
    event being measured."""
    s = twoarm.HealthSampler(workloads=["redis"])
    s.probe_fn = lambda ns, w: (_ for _ in ()).throw(RuntimeError("no route to host"))
    s.sample_once()
    df = s.frame()
    assert len(df) == 2
    assert (df["probe_ok"] == False).all()               # noqa: E712
    assert (df["ready_replicas"] == 0).all()


def test_sampler_thread_starts_and_stops():
    s = twoarm.HealthSampler(workloads=["redis"], interval=0.01)
    s.probe_fn = fake_probe
    s.start()
    threading.Event().wait(0.1)
    s.stop()
    assert len(s.rows) >= 2


# --- the schedule -----------------------------------------------------------

def test_plan_pairs_every_fault_and_pattern_on_every_workload():
    plan = twoarm.paired_plan()
    # 3 workloads x (3 stress faults x 2 patterns + 1 pod kill)
    assert len(plan) == len(chaos.WORKLOADS) * (len(chaos.FAULTS) * 2 + 1)
    assert {s.fault_type for s in plan} == set(chaos.FAULTS) | {"POD_KILL"}
    assert {s.pattern for s in plan if s.fault_type == "POD_KILL"} == {"none"}


def test_pair_ids_are_unique_and_stable_across_calls():
    """Deterministic so a crashed campaign resumes instead of renumbering --
    the same property collect/chaos.py's _plan() has, for the same reason."""
    a = [s.pair_id for s in twoarm.paired_plan()]
    b = [s.pair_id for s in twoarm.paired_plan()]
    assert a == b
    assert len(set(a)) == len(a)


def test_rounds_keep_counting_so_a_second_round_resumes():
    one = [s.pair_id for s in twoarm.paired_plan(rounds=1)]
    two = [s.pair_id for s in twoarm.paired_plan(rounds=2)]
    assert two[:len(one)] == one                 # round 2 appends, never renumbers
    assert any(s.endswith("_002") for s in two)


def test_gap_is_tied_to_the_detector_dead_zone_not_hardcoded():
    """The same landmine as in collect/chaos.py, and the bias here runs ONE
    WAY: a pair scheduled inside the detector's post-restart dead zone cannot
    fire on the treatment arm, while the control arm has no detector and no
    dead zone. So too short a gap makes Prodrome look worse -- the direction
    nobody checks."""
    from ml.detector import POST_RESTART_SUPPRESS_TICKS
    assert chaos.DEFAULT_GAP >= POST_RESTART_SUPPRESS_TICKS * chaos.DETECTOR_TICK_SECONDS


# --- row writing ------------------------------------------------------------

def test_pair_rows_are_two_rows_one_per_arm_in_the_expected_schema():
    """Two rows per pair_id is what makes eval/recovery.py's comparison paired
    rather than two independent campaigns compared after the fact."""
    res = twoarm.inject_both(spec(), injector=FakeInjector())
    rows = twoarm.pair_rows(spec(), res, load_attested=True)
    assert len(rows) == 2
    assert [r["arm"] for r in rows] == list(ARMS)
    assert all(set(PAIRS_COLUMNS) >= set(r) for r in rows)
    assert all(r["skew_s"] is not None for r in rows)


def test_run_pair_writes_both_files_and_recovery_can_read_them(tmp_path):
    """The handoff between the two halves of the harness, end to end: the
    runner writes, the measurement module loads, without a cluster anywhere.
    This is the seam most likely to rot, because the two files are the only
    thing the halves share."""
    from eval import recovery

    twoarm.run_pair(spec(), load_attested=True, outdir=str(tmp_path),
                    injector=FakeInjector(), probe_fn=fake_probe,
                    sleep=lambda _s: None)
    pairs, health = recovery.load(tmp_path)
    assert len(pairs) == 2
    assert list(pairs["arm"]) == list(ARMS)
    assert len(health) >= 2
    measured = recovery.measure(pairs, health)
    assert len(measured) == 2
    assert set(measured["kind"]) == {"outage"}


def test_campaign_resumes_and_skips_finished_pairs(tmp_path):
    """Appended after every pair, same as collect/chaos.py: a campaign that
    dies two hours in keeps everything it already measured."""
    inj = FakeInjector()
    twoarm.campaign(rounds=1, fault_types=["POD_KILL"], outdir=str(tmp_path),
                    gap=0, load_attested=True, injector=inj,
                    probe_fn=fake_probe, sleep=lambda _s: None)
    first = len(inj.calls)
    assert first == len(chaos.WORKLOADS) * 2          # 3 pairs x 2 arms

    twoarm.campaign(rounds=1, fault_types=["POD_KILL"], outdir=str(tmp_path),
                    gap=0, load_attested=True, injector=inj,
                    probe_fn=fake_probe, sleep=lambda _s: None)
    assert len(inj.calls) == first                    # all skipped, nothing re-injected
    assert len(pd.read_csv(tmp_path / "pairs.csv")) == first


def test_campaign_appends_a_single_header(tmp_path):
    twoarm.campaign(rounds=1, fault_types=["POD_KILL"], outdir=str(tmp_path),
                    gap=0, load_attested=True, injector=FakeInjector(),
                    probe_fn=fake_probe, sleep=lambda _s: None)
    text = (tmp_path / "pairs.csv").read_text()
    assert text.count("pair_id,arm,namespace") == 1


# --- transport / wiring -----------------------------------------------------

def test_arm_namespaces_match_the_manifests_and_the_controller():
    """The treatment arm must be the namespace the controller actually targets,
    and the control arm must not be. Read from control/controller.py rather
    than duplicated here."""
    from control import controller
    assert twoarm.ARM_NAMESPACE[TREATMENT_ARM] == controller.NAMESPACE
    assert twoarm.ARM_NAMESPACE[CONTROL_ARM] != controller.NAMESPACE


def test_kubectl_transport_is_overridable(monkeypatch):
    """SETUP.md section 1 puts Windows users inside WSL, and on this host that
    is mandatory: the kubeconfig only exists inside WSL, so a bare `kubectl`
    from the Windows side falls back to localhost:8080 and every call fails
    with a connection refused that looks like a dead cluster.

    The PRODROME_KUBECTL override is the escape hatch and is exercised here for
    real -- by reimporting the module with it set -- because the platform
    default cannot be: the venv in this environment is WSL-only, so
    sys.platform is always "linux" and the win32 branch never executes. An
    untested default with a tested override is the honest arrangement; the
    override is what anyone actually reaches for when the default is wrong.
    """
    import importlib

    assert chaos.KUBECTL_CMD, "KUBECTL_CMD must never be empty"
    assert chaos.KUBECTL_CMD[-1] == "kubectl"

    monkeypatch.setenv("PRODROME_KUBECTL", "wsl.exe -e kubectl")
    reloaded = importlib.reload(chaos)
    try:
        assert reloaded.KUBECTL_CMD == ["wsl.exe", "-e", "kubectl"]
    finally:
        monkeypatch.delenv("PRODROME_KUBECTL")
        importlib.reload(chaos)
    assert chaos.KUBECTL_CMD[-1] == "kubectl"


def test_campaign_requires_load_attestation_via_the_cli():
    """An idle control arm recovers faster for reasons unrelated to Prodrome,
    so the CLI refuses to run without the operator attesting load on BOTH
    namespaces (docs/guides/shaurya.md Part 4)."""
    argv = sys.argv
    sys.argv = ["twoarm.py", "--campaign"]
    try:
        with pytest.raises(SystemExit) as e:
            twoarm.main()
        assert e.value.code == 2
    finally:
        sys.argv = argv


def test_null_run_is_the_only_other_way_past_the_load_gate():
    """--null-run records load_attested=False honestly, which makes
    attest_load_parity() fail and the run unpublishable. What is forbidden is a
    campaign with no statement either way, because that is the one a later
    reader cannot assess."""
    from eval import recovery
    res = twoarm.inject_both(spec(), injector=FakeInjector())
    rows = twoarm.pair_rows(spec(), res, load_attested=False)
    att = recovery.attest_load_parity(pd.DataFrame(rows))
    assert not att.ok


# --- the settle gate --------------------------------------------------------
#
# The gate replaces the blind inter-pair sleep's IMPLICIT half -- "whatever the
# last pair broke has probably finished unbreaking by now" -- with a checked
# precondition. It does NOT replace the explicit half (the detector dead zone),
# and `test_gap_is_tied_to_the_detector_dead_zone_not_hardcoded` above still
# guards that. These tests cover the properties that make the gate safe to use
# in a comparison at all.

class FakeStatus:
    """Scripted `{workload: (ready, desired, restarts)}` per arm, per poll."""

    def __init__(self, script: list[dict[str, dict[str, tuple]]]):
        self.script = script
        self.polls = 0
        self._seen: set[str] = set()

    def __call__(self, namespace: str, workloads: list[str]) -> dict[str, tuple]:
        self._seen.add(namespace)
        # Advance one step only once BOTH arms have been read for this poll.
        step = self.script[min(self.polls, len(self.script) - 1)]
        if self._seen >= {twoarm.ARM_NAMESPACE[a] for a in ARMS}:
            self.polls += 1
            self._seen = set()
        return step[namespace]


def _ok(restarts: int = 0) -> dict[str, tuple]:
    return {"redis": (1, 1, restarts)}


def test_settle_gate_waits_for_both_arms_not_just_one():
    """The gate must be SYMMETRIC. A gate that started as soon as the treatment
    arm was healthy would hand that arm a clean pod while the control arm was
    still unwinding the previous fault -- a recovery-time difference caused
    entirely by the scheduler, which is exactly the bias this harness exists to
    avoid."""
    prodrome_ns, control_ns = twoarm.ARM_NAMESPACE[TREATMENT_ARM], twoarm.ARM_NAMESPACE[CONTROL_ARM]
    script = [
        # prodrome healthy from the start; control still down for two polls.
        {prodrome_ns: _ok(), control_ns: {"redis": (0, 1, 3)}},
        {prodrome_ns: _ok(), control_ns: {"redis": (0, 1, 3)}},
    ] + [{prodrome_ns: _ok(), control_ns: _ok(3)}] * 6
    st = FakeStatus(script)
    res = twoarm.wait_settled(["redis"], status_fn=st, poll=0, stable_samples=3,
                              timeout=1e6, sleep=lambda _s: None,
                              clock=lambda: 0.0)
    assert res["settled"] is True
    # Two polls lost to the control arm, one to establishing restart stability,
    # then three clean -- it did not start the moment prodrome went green.
    assert res["polls"] >= 5


def test_settle_gate_requires_restart_counter_to_stop_moving():
    """Ready is not settled. A pod in CrashLoopBackOff is Ready between
    crashes, so readiness alone lets a still-crashing workload through and its
    next crash lands inside the following pair's baseline."""
    ns_t, ns_c = twoarm.ARM_NAMESPACE[TREATMENT_ARM], twoarm.ARM_NAMESPACE[CONTROL_ARM]
    # Ready the whole time, but the restart counter keeps ticking for 3 polls.
    script = [{ns_t: _ok(r), ns_c: _ok(r)} for r in (1, 2, 3)] + \
             [{ns_t: _ok(3), ns_c: _ok(3)}] * 6
    st = FakeStatus(script)
    res = twoarm.wait_settled(["redis"], status_fn=st, poll=0, stable_samples=3,
                              timeout=1e6, sleep=lambda _s: None, clock=lambda: 0.0)
    assert res["settled"] is True
    assert res["polls"] >= 6          # the ticking polls did not count


def test_settle_gate_needs_two_readings_before_it_can_claim_stability():
    """One poll shows Ready but says nothing about whether the restart counter
    is still moving -- that needs a second reading. Counting the first poll
    toward the streak would let a crashing pod through on a lucky sample."""
    ns_t, ns_c = twoarm.ARM_NAMESPACE[TREATMENT_ARM], twoarm.ARM_NAMESPACE[CONTROL_ARM]
    st = FakeStatus([{ns_t: _ok(), ns_c: _ok()}])
    res = twoarm.wait_settled(["redis"], status_fn=st, poll=0, stable_samples=3,
                              timeout=1e6, sleep=lambda _s: None, clock=lambda: 0.0)
    assert res["settled"] is True
    assert res["polls"] == 4          # 1 to prime + 3 stable, never 3


def test_settle_gate_times_out_without_killing_the_campaign():
    """A gate that raised at 3am would cost far more than the pair is worth.
    The timeout is RECORDED (settled_ok=False) so eval/ can drop the pair --
    which is strictly better than a blind sleep that never knew either way."""
    ns_t, ns_c = twoarm.ARM_NAMESPACE[TREATMENT_ARM], twoarm.ARM_NAMESPACE[CONTROL_ARM]
    st = FakeStatus([{ns_t: {"redis": (0, 1, 0)}, ns_c: _ok()}])
    clock = iter([0.0, 10.0, 20.0, 30.0, 40.0, 50.0] + [999.0] * 20)
    res = twoarm.wait_settled(["redis"], status_fn=st, poll=0, stable_samples=3,
                              timeout=60.0, sleep=lambda _s: None,
                              clock=lambda: next(clock))
    assert res["settled"] is False
    assert "timed out" in res["detail"] and "not ready" in res["detail"]


def test_a_throwing_status_call_resets_the_streak_rather_than_crashing():
    """An API server that stops answering mid-gate is a reason to keep waiting,
    not a reason to lose the night's campaign."""
    ns_t, ns_c = twoarm.ARM_NAMESPACE[TREATMENT_ARM], twoarm.ARM_NAMESPACE[CONTROL_ARM]
    calls = {"n": 0}

    def flaky(namespace, workloads):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("the api server is having a moment")
        return _ok()

    res = twoarm.wait_settled(["redis"], status_fn=flaky, poll=0, stable_samples=2,
                              timeout=1e6, sleep=lambda _s: None, clock=lambda: 0.0)
    assert res["settled"] is True


def test_settle_result_is_recorded_on_both_arm_rows():
    """Recorded, not assumed -- a reader has to be able to see which pairs
    started from a verified-clean cluster and which ran out of patience."""
    res = twoarm.inject_both(spec(), injector=FakeInjector())
    rows = twoarm.pair_rows(spec(), res, load_attested=False,
                            settle=dict(settled=False, waited_s=540.0), trail_s=300.0)
    assert all(set(PAIRS_COLUMNS) >= set(r) for r in rows)
    assert [r["settled_ok"] for r in rows] == [False, False]
    assert [r["settled_s"] for r in rows] == [540.0, 540.0]
    # The censoring horizon travels with the row: "NOT_RECOVERED" is unreadable
    # without the number of seconds we actually watched.
    assert [r["trail_s"] for r in rows] == [300.0, 300.0]


# --- ordering ---------------------------------------------------------------

def test_rotate_order_makes_every_prefix_of_the_campaign_balanced():
    """For an overnight run, being interrupted is the case that happens. Under
    the original workload-major order a campaign cut off at 40% has three fault
    types on ONE workload and nothing on the other two -- every number
    workload-confounded. Rotate costs precision instead of whole rows."""
    plan = twoarm.paired_plan(rounds=1, order="rotate")
    faults = [s.fault_type for s in plan]
    assert set(faults[:4]) == set(chaos.FAULTS) | {"POD_KILL"}   # all four in 4 pairs
    # and the first three pairs of any one fault type cover all three workloads
    first_cpu = [s.workload for s in plan if s.fault_type == "CPU_HOG"][:3]
    assert set(first_cpu) == set(chaos.WORKLOADS)


def test_changing_the_order_does_not_renumber_and_so_does_not_re_run():
    """`pair_id` is the resume key. If it encoded the order, switching orders
    would silently re-run a finished campaign from scratch instead of
    continuing it."""
    nested = {s.pair_id for s in twoarm.paired_plan(rounds=2, order="nested")}
    rotate = {s.pair_id for s in twoarm.paired_plan(rounds=2, order="rotate")}
    assert nested == rotate


def test_an_unknown_order_is_refused_rather_than_silently_nested():
    with pytest.raises(ValueError, match="unknown order"):
        twoarm.paired_plan(order="whatever")


def test_duration_is_plumbed_through_to_every_spec():
    assert {s.duration_s for s in twoarm.paired_plan(duration_s=180)} == {180}


def test_campaign_runs_the_gate_before_each_pair_and_records_it(tmp_path):
    """Before, not after: a campaign resumed minutes after one died
    mid-MEMORY_LEAK is exactly the case where the cluster is still unwinding
    and nothing has waited for it."""
    seen = []

    def fake_gate(workloads):
        seen.append(tuple(workloads))
        return dict(settled=True, waited_s=31.0, polls=4, detail="fine")

    twoarm.campaign(rounds=1, fault_types=["POD_KILL"], outdir=str(tmp_path),
                    gap=0, load_attested=True, injector=FakeInjector(),
                    probe_fn=fake_probe, sleep=lambda _s: None,
                    settle_fn=fake_gate)
    assert len(seen) == len(chaos.WORKLOADS)          # one gate per pair
    written = pd.read_csv(tmp_path / "pairs.csv")
    assert set(written["settled_ok"]) == {True}
    assert set(written["settled_s"]) == {31.0}


def main() -> None:
    raise SystemExit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    main()
