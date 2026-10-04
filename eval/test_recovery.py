"""Offline tests for the control-arm recovery measurement.

Follows `ml/test_restart_suppression.py`'s precedent -- a test living next to
the code it checks rather than in `tests/`, which is the testing seat's -- but
written as pytest functions so `pytest eval -q` collects them, with a `main()`
so `python eval/test_recovery.py` still works standalone.

Every case here exists for a stated reason: a SETUP.md ground rule, a landmine
that has already cost this project a result, or an arithmetic edge that would
silently flatter one arm. None of it needs a cluster, Docker, or any file under
`data/` -- the fixtures are built in-process, which is the only way this can
run from a fresh clone (ground rule 8).

The bias to keep in mind while reading: almost every way of getting this
measurement subtly wrong favours the TREATMENT arm, because the treatment arm
is the one whose interventions truncate the series being measured. Scoring a
prevented failure as "recovered in 0s", or treating "we stopped watching" as
"it recovered", both hand Prodrome a win it did not earn. Those two are tested
first and hardest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import recovery  # noqa: E402
from eval.recovery import (  # noqa: E402
    CONTROL_ARM,
    DEGRADED,
    HEALTH_COLUMNS,
    NO_DATA,
    NO_DEGRADATION,
    NO_OUTAGE,
    NOT_RECOVERED,
    PAIRS_COLUMNS,
    RECOVERED,
    STILL_DEGRADED,
    TREATMENT_ARM,
)

T0 = pd.Timestamp("2026-10-03T20:00:00Z")
STEP = pd.Timedelta(seconds=5)


# --- fixture builders -------------------------------------------------------

def health(arm: str, workload: str, serving: list[bool], latency: list[float] | None = None,
           start: pd.Timestamp = T0, step: pd.Timedelta = STEP) -> pd.DataFrame:
    """A health series as `collect/twoarm.py` writes it. `serving[i]` is whether
    the workload was serving at sample i; latency defaults to a quiet 10ms."""
    latency = latency if latency is not None else [10.0] * len(serving)
    rows = [dict(ts=start + i * step, arm=arm, workload=workload,
                 ready_replicas=1 if s else 0, desired_replicas=1,
                 restarts_total=0, probe_ok=bool(s), probe_latency_ms=latency[i])
            for i, s in enumerate(serving)]
    return pd.DataFrame(rows, columns=HEALTH_COLUMNS)


def pair(pair_id: str, fault: str, workload: str = "redis", pattern: str = "constant",
         start: pd.Timestamp = T0, skew: float = 0.0,
         load_attested: bool = True) -> pd.DataFrame:
    """The two rows -- one per arm -- that make a pair."""
    return pd.DataFrame([
        dict(pair_id=pair_id, arm=arm, namespace=arm, fault_type=fault,
             workload=workload, pattern=pattern, duration_s=300,
             start_ts=start, end_ts=start + pd.Timedelta(seconds=300),
             failure_ts=None, restarts_delta=0, inject_ok=True, skew_s=skew,
             load_attested=load_attested)
        for arm in (TREATMENT_ARM, CONTROL_ARM)], columns=PAIRS_COLUMNS)


# --- first_sustained: the primitive every number rests on -------------------

def test_first_sustained_returns_the_start_of_the_run_not_the_end():
    """The hold requirement must not inflate a recovery time.

    If this returned the END of the sustained-healthy run, every recovery time
    in the report would be inflated by exactly HOLD_SECONDS -- uniformly, on
    both arms, so the paired delta would still look plausible while every
    absolute number was wrong by 30s.
    """
    ts = [T0 + i * STEP for i in range(12)]
    ok = [True] * 12
    assert recovery.first_sustained(ts, ok, hold_seconds=30.0) == T0


def test_first_sustained_rejects_a_flap_shorter_than_the_hold():
    """A one-sample blip of health is a probe winning a race, not a recovery.

    Without the hold, a CrashLoopBackOff pod -- up for a sample, down again --
    reports a recovery time as short as one sample interval, which is how a
    pod that never actually came back ends up as the fastest recovery in the
    table.
    """
    ts = [T0 + i * STEP for i in range(10)]
    ok = [False, True, False, False, True, True, True, True, True, True]
    # The 1-sample blip at index 1 must not win; the run from index 4 must.
    assert recovery.first_sustained(ts, ok, hold_seconds=25.0) == ts[4]


def test_first_sustained_will_not_call_the_end_of_the_window_a_recovery():
    """We stopped watching is not the same as it recovered.

    This is the censoring rule. An unbroken healthy run at the tail of the
    series that is SHORTER than the hold returns None, so the caller reports
    NOT_RECOVERED rather than imputing a time. Imputing one would reward
    whichever arm we happened to stop watching sooner.
    """
    ts = [T0 + i * STEP for i in range(4)]
    ok = [False, False, True, True]        # only 5s of health at the end
    assert recovery.first_sustained(ts, ok, hold_seconds=30.0) is None


def test_first_sustained_on_empty_series():
    assert recovery.first_sustained([], [], hold_seconds=30.0) is None


# --- outage outcomes --------------------------------------------------------

def test_outage_recovery_time_is_failure_to_sustained_health():
    """The basic POD_KILL shape: serving, gone for 30s, back for good."""
    serving = [True] * 3 + [False] * 6 + [True] * 10      # down 30s from index 3
    h = health(CONTROL_ARM, "redis", serving)
    res = recovery.outage_outcome(h, T0, hold_seconds=30.0)
    assert res["outcome"] == RECOVERED
    assert res["t_fail"] == T0 + 3 * STEP
    assert res["recovery_seconds"] == 30.0


def test_a_prevented_failure_is_not_a_zero_second_recovery():
    """GROUND RULE 4, the single most important case in this file.

    A MEMORY_LEAK that Prodrome restarts before the OOMKill never produces an
    outage. If that were scored as `recovery_seconds = 0`, the treatment arm
    would post a median recovery of 0s against the control arm's real tens of
    seconds -- a spectacular, entirely manufactured result, because the
    treatment arm's number would be the absence of a measurement rather than a
    fast one.

    It must come back as NO_OUTAGE with recovery_seconds None, counted in its
    own column.
    """
    h = health(TREATMENT_ARM, "redis", [True] * 20)
    res = recovery.outage_outcome(h, T0, hold_seconds=30.0)
    assert res["outcome"] == NO_OUTAGE
    assert res["recovery_seconds"] is None


def test_an_arm_still_down_at_the_end_is_censored_not_slow():
    """NOT_RECOVERED carries no number. Clamping it to the window length would
    cap the control arm's badness at however long we watched -- making a
    catastrophic failure look merely slow, which understates the control arm's
    weakness and so understates Prodrome."""
    h = health(CONTROL_ARM, "redis", [True] * 2 + [False] * 18)
    res = recovery.outage_outcome(h, T0, hold_seconds=30.0)
    assert res["outcome"] == NOT_RECOVERED
    assert res["recovery_seconds"] is None
    assert res["t_fail"] == T0 + 2 * STEP


def test_outage_ignores_an_outage_that_predates_injection():
    """A pod already broken before this pair started is somebody else's fault
    -- the previous pair's, usually. Counting it would attribute a leftover
    outage to this fault, and because the arms are injected simultaneously the
    leftover is not shared between them."""
    serving = [False, False, True, True, True, True, True, True, True, True, True]
    t_inject = T0 + 2 * STEP                   # injection starts after it recovered
    h = health(CONTROL_ARM, "redis", serving)
    res = recovery.outage_outcome(h, t_inject, hold_seconds=30.0)
    assert res["outcome"] == NO_OUTAGE


def test_no_health_samples_is_no_data_not_a_tie():
    res = recovery.outage_outcome(health(CONTROL_ARM, "redis", []), T0)
    assert res["outcome"] == NO_DATA


# --- degradation outcomes ---------------------------------------------------

def test_degradation_band_comes_from_this_arms_own_baseline():
    """A standing latency difference between the namespaces must not read as a
    result. The band is per arm, so an arm that is simply slower all the time
    gets a correspondingly higher band and is not scored as degraded for its
    own normal behaviour."""
    slow = health(CONTROL_ARM, "redis", [True] * 12, latency=[400.0] * 12)
    fast = health(TREATMENT_ARM, "redis", [True] * 12, latency=[10.0] * 12)
    assert recovery.latency_band(slow) > recovery.latency_band(fast)


def test_degradation_band_has_a_floor_for_a_flat_baseline():
    """A near-constant baseline gives MAD ~ 0; without the floor the band
    collapses onto the median and ordinary jitter reads as degradation on both
    arms at once."""
    flat = health(CONTROL_ARM, "redis", [True] * 12, latency=[10.0] * 12)
    assert recovery.latency_band(flat) >= 10.0 + recovery.LATENCY_BAND_FLOOR_MS


def test_degradation_band_needs_enough_baseline_samples():
    thin = health(CONTROL_ARM, "redis", [True] * 3, latency=[10.0] * 3)
    assert recovery.latency_band(thin) is None


def test_cpu_hog_reports_degradation_seconds_and_never_recovery_seconds():
    """THE LANDMINE, asserted. CPU_HOG has no failure instant -- failure_ts is
    null for all 30 CPU_HOG runs and all 90 DISK_STRESS runs in
    data/chaos/runs.csv, because end_ts is only when stress-ng stopped. So
    CPU_HOG must never produce a recovery_seconds, no matter what the health
    series looks like: the quantity does not exist for it.
    """
    base = health(TREATMENT_ARM, "redis", [True] * 12, latency=[10.0] * 12)
    window = health(TREATMENT_ARM, "redis", [True] * 18,
                    latency=[500.0] * 8 + [10.0] * 10,
                    start=T0 + 12 * STEP)
    res = recovery.degradation_outcome(window, T0 + 12 * STEP, base, hold_seconds=30.0)
    assert res["outcome"] == DEGRADED
    assert res["recovery_seconds"] is None
    assert res["degradation_seconds"] == 40.0


def test_degradation_that_never_clears_is_still_degraded():
    base = health(CONTROL_ARM, "redis", [True] * 12, latency=[10.0] * 12)
    window = health(CONTROL_ARM, "redis", [True] * 10, latency=[900.0] * 10,
                    start=T0 + 12 * STEP)
    res = recovery.degradation_outcome(window, T0 + 12 * STEP, base, hold_seconds=30.0)
    assert res["outcome"] == STILL_DEGRADED
    assert res["degradation_seconds"] is None


def test_probe_that_never_leaves_the_band_is_no_degradation():
    base = health(CONTROL_ARM, "redis", [True] * 12, latency=[10.0] * 12)
    window = health(CONTROL_ARM, "redis", [True] * 10, latency=[12.0] * 10,
                    start=T0 + 12 * STEP)
    res = recovery.degradation_outcome(window, T0 + 12 * STEP, base, hold_seconds=30.0)
    assert res["outcome"] == NO_DEGRADATION


# --- the per-fault-type dispatch -------------------------------------------

def test_recovery_kind_covers_exactly_the_four_fault_types():
    """Every fault type the chaos runner can inject has to declare which
    quantity it produces. A fault with no entry raises rather than silently
    picking one."""
    assert set(recovery.RECOVERY_KIND) == {"CPU_HOG", "MEMORY_LEAK",
                                           "DISK_STRESS", "POD_KILL"}


def test_measure_raises_on_an_undeclared_fault_type():
    p = pair("X_001", "CONNECTION_POOL")       # a plausible future fault type
    with pytest.raises(ValueError, match="RECOVERY_KIND"):
        recovery.measure(p, health(TREATMENT_ARM, "redis", [True] * 10))


def test_measure_dispatches_per_fault_type_not_per_failure_ts():
    """POD_KILL -> outage, CPU_HOG -> degradation, decided by fault type.

    Deriving the kind from whether failure_ts happens to be populated would
    reclassify a fault the first time the injector's OOM detection got lucky or
    unlucky, which would change what the column means mid-campaign.
    """
    h = pd.concat([
        health(a, "redis", [True] * 4 + [False] * 6 + [True] * 10) for a in
        (TREATMENT_ARM, CONTROL_ARM)], ignore_index=True)
    m = recovery.measure(pair("PK_001", "POD_KILL"), h)
    assert set(m["kind"]) == {"outage"}
    assert m["recovery_seconds"].notna().all()

    m2 = recovery.measure(pair("CH_001", "CPU_HOG"), h)
    assert set(m2["kind"]) == {"degradation"}
    assert m2["recovery_seconds"].isna().all()


# --- reporting: no aggregate, and designed ties are labelled ---------------

def test_lever_for_matches_the_policy_table():
    """Read from control/policy.py, not copied. POD_KILL has no row there, so
    it falls through to UNKNOWN -> nothing: the treatment arm is specified to
    do exactly what stock Kubernetes does."""
    assert recovery.lever_for("MEMORY_LEAK") == "rolling_restart"
    assert recovery.lever_for("CPU_HOG") == "scale_out"
    assert recovery.lever_for("DISK_STRESS") == "alert_only"
    assert recovery.lever_for("POD_KILL") == "nothing"


def test_only_two_fault_types_are_marked_comparable():
    """GROUND RULE 6, sharpened. DISK_STRESS (alert_only) and POD_KILL (no
    policy row) are designed ties: the treatment arm has no lever, so a tie
    there is the specification working, not a negative result. The table must
    say so, or a reader averaging four rows concludes Prodrome does nothing.
    """
    m = pd.concat([recovery.measure(pair(f"{f}_001", f),
                                    pd.concat([health(a, "redis", [True] * 10)
                                               for a in (TREATMENT_ARM, CONTROL_ARM)],
                                              ignore_index=True))
                   for f in recovery.RECOVERY_KIND], ignore_index=True)
    t = recovery.per_fault_arm_table(m)
    comparable = set(t[t["comparable"]]["fault_type"])
    assert comparable == {"CPU_HOG", "MEMORY_LEAK"}


def test_per_fault_table_has_one_row_per_fault_type_per_arm_and_no_total():
    m = pd.concat([recovery.measure(pair(f"{f}_001", f),
                                    pd.concat([health(a, "redis", [True] * 10)
                                               for a in (TREATMENT_ARM, CONTROL_ARM)],
                                              ignore_index=True))
                   for f in ("POD_KILL", "CPU_HOG")], ignore_index=True)
    t = recovery.per_fault_arm_table(m)
    assert len(t) == 4                                   # 2 faults x 2 arms
    assert set(t["arm"]) == {TREATMENT_ARM, CONTROL_ARM}
    assert "ALL" not in set(t["fault_type"])             # no aggregate row, ever


def test_prevented_failures_are_counted_separately_from_the_delta():
    """The MEMORY_LEAK thesis cannot be expressed as a difference of durations,
    because the treatment arm's duration does not exist. It has to surface as
    a count, and that pair must NOT enter the median delta."""
    h = pd.concat([
        health(TREATMENT_ARM, "redis", [True] * 20),                      # no outage
        health(CONTROL_ARM, "redis", [True] * 3 + [False] * 6 + [True] * 11),
    ], ignore_index=True)
    m = recovery.measure(pair("ML_001", "MEMORY_LEAK"), h)
    d = recovery.paired_delta_table(m)
    row = d.iloc[0]
    assert row["n_prevented_by_prodrome"] == 1
    assert row["pairs_in_delta"] == 0        # nothing to subtract
    assert row["median_delta_s"] is None


def test_paired_delta_subtracts_within_a_pair():
    """Treatment minus control, negative = Prodrome faster. Paired because both
    arms took the fault at the same instant under the same load, so the
    per-pair difference removes between-pair variance an unpaired comparison
    of medians leaves in."""
    h = pd.concat([
        health(TREATMENT_ARM, "redis", [True] * 2 + [False] * 2 + [True] * 16),   # 10s
        health(CONTROL_ARM, "redis", [True] * 2 + [False] * 8 + [True] * 10),     # 40s
    ], ignore_index=True)
    m = recovery.measure(pair("PK_001", "POD_KILL"), h)
    d = recovery.paired_delta_table(m).iloc[0]
    assert d["pairs_in_delta"] == 1
    assert d["median_delta_s"] == -30.0
    assert d["prodrome_faster_pairs"] == 1
    assert d["tied_pairs"] == 0


def test_a_censored_control_arm_is_surfaced_not_just_excluded():
    """A pair where the control arm never recovered has no number to subtract,
    so it leaves the delta -- even though it is the control arm failing as
    badly as it can. That exclusion discards a Prodrome ADVANTAGE, so it has to
    appear as its own count rather than vanishing into an `excluded` total."""
    h = pd.concat([
        health(TREATMENT_ARM, "redis", [True] * 2 + [False] * 2 + [True] * 16),  # recovers
        health(CONTROL_ARM, "redis", [True] * 2 + [False] * 18),                 # never does
    ], ignore_index=True)
    m = recovery.measure(pair("ML_001", "MEMORY_LEAK"), h)
    assert set(m["outcome"]) == {RECOVERED, NOT_RECOVERED}
    d = recovery.paired_delta_table(m).iloc[0]
    assert d["pairs_in_delta"] == 0
    assert d["excluded_censored"] == 1
    assert d["median_delta_s"] is None


def test_a_sub_resolution_difference_is_a_tie_not_a_win():
    """REGRESSION, from the first live two-arm run.

    That run came back prodrome 15.2097s vs control 15.2036s and the table
    reported "control faster: 1 pair" -- a 6 MILLISECOND difference, measured
    by a sampler whose achieved cadence was 6.4 seconds. The sign of a delta
    three orders of magnitude below the sample interval is noise, and counting
    it as directional is how a null experiment grows a result.
    """
    h = pd.concat([
        health(TREATMENT_ARM, "redis", [True] * 2 + [False] * 4 + [True] * 14),
        # control's outage ends at the same sample; shift its clock 20ms so the
        # two recovery times differ by far less than one sample interval.
        health(CONTROL_ARM, "redis", [True] * 2 + [False] * 4 + [True] * 14,
               start=T0 + pd.Timedelta(milliseconds=20)),
    ], ignore_index=True)
    m = recovery.measure(pair("PK_001", "POD_KILL"), h)
    d = recovery.paired_delta_table(m, tie_band=5.0).iloc[0]
    assert d["pairs_in_delta"] == 1
    assert d["tied_pairs"] == 1
    assert d["prodrome_faster_pairs"] == 0
    assert d["control_faster_pairs"] == 0


def test_a_real_difference_still_counts_as_directional():
    """The tie band must not swallow a genuine win. 30s apart at a 5s band is
    a result; the band only suppresses what the sampler cannot resolve."""
    h = pd.concat([
        health(TREATMENT_ARM, "redis", [True] * 2 + [False] * 2 + [True] * 16),   # 10s
        health(CONTROL_ARM, "redis", [True] * 2 + [False] * 8 + [True] * 10),     # 40s
    ], ignore_index=True)
    m = recovery.measure(pair("PK_001", "POD_KILL"), h)
    d = recovery.paired_delta_table(m, tie_band=5.0).iloc[0]
    assert d["prodrome_faster_pairs"] == 1
    assert d["tied_pairs"] == 0
    assert d["median_delta_s"] == -30.0


def test_sampling_resolution_reports_the_achieved_cadence():
    """Self-calibrating: the tie band comes from the cadence the sampler really
    managed, not the one it was asked for. Live it was 6.4s against a requested
    5.0s, because each kubectl call costs 600-900ms."""
    h = health(TREATMENT_ARM, "redis", [True] * 10, step=pd.Timedelta(seconds=6.4))
    assert recovery.sampling_resolution(h) == pytest.approx(6.4, abs=0.1)


def test_sampling_resolution_takes_the_worst_arm():
    """Set by the arm we watched least closely, not by an average that hides
    it -- the worse-resolved arm is the one that limits the comparison."""
    h = pd.concat([
        health(TREATMENT_ARM, "redis", [True] * 10, step=pd.Timedelta(seconds=2)),
        health(CONTROL_ARM, "redis", [True] * 10, step=pd.Timedelta(seconds=9)),
    ], ignore_index=True)
    assert recovery.sampling_resolution(h) == pytest.approx(9.0, abs=0.1)


def test_a_skewed_pair_is_excluded_from_the_delta():
    """Arms injected far apart saw different cluster states -- different point
    on the k6 diurnal curve, different page cache. That pair is evidence of
    nothing and must not be averaged in; it is excluded and counted."""
    h = pd.concat([
        health(TREATMENT_ARM, "redis", [True] * 2 + [False] * 2 + [True] * 16),
        health(CONTROL_ARM, "redis", [True] * 2 + [False] * 8 + [True] * 10),
    ], ignore_index=True)
    m = recovery.measure(pair("PK_001", "POD_KILL", skew=120.0), h)
    d = recovery.paired_delta_table(m).iloc[0]
    assert d["pairs_in_delta"] == 0
    # Counted in PAIRS, not arm-rows: the skew is a property of the pair, and a
    # pair is the unit the delta is computed over.
    assert d["excluded_skew"] == 1
    assert d["pairs"] == 1
    assert d["median_delta_s"] is None


# --- attestation: can this experiment produce a comparison at all? ---------

def test_manifest_parity_holds_for_the_real_repo_files():
    """The arms are the same workloads with the same limits and the same
    probes, or the comparison is between two different systems. Resource limits
    set the OOMKill threshold and the probes set how fast Kubernetes notices
    anything, so a one-line drift in either file silently becomes the result.

    Verified against the committed infra/ files, which is why this test is
    worth more than a fixture: it goes red when cluster's seat edits one arm.
    """
    a = recovery.attest_manifest_parity(REPO_ROOT / "infra" / "workloads.yaml",
                                        REPO_ROOT / "infra" / "workloads-control.yaml")
    assert a.ok, a.detail


def test_manifest_parity_catches_a_real_drift(tmp_path):
    """A changed memory limit must fail parity. This is the specific drift that
    would do the most damage: the memory limit IS the OOMKill threshold, so a
    control arm with a higher one simply never OOMs and 'Prodrome prevented the
    failure' becomes an artifact of the manifest."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text('namespace: prodrome\nlimits:\n  memory: "512Mi"\n')
    b.write_text('namespace: control\nlimits:\n  memory: "1024Mi"\n')
    assert not recovery.attest_manifest_parity(a, b).ok


def test_manifest_parity_ignores_the_namespace_and_comments(tmp_path):
    """The namespace is the one thing that is SUPPOSED to differ, and a comment
    carries no behaviour. The real infra/ files differ by exactly one inline
    comment on the postgres cpu limit, so a naive diff would fail forever and
    get ignored."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text('namespace: prodrome\ncpu: "1000m"   # keep in sync\n')
    b.write_text('# a leading comment\nnamespace: control\ncpu: "1000m"   # different note\n')
    assert recovery.attest_manifest_parity(a, b).ok


def test_arm_isolation_fails_if_the_controller_targets_the_control_namespace():
    assert recovery.attest_arm_isolation("prodrome").ok
    assert not recovery.attest_arm_isolation("control").ok


def test_the_real_controller_cannot_touch_the_control_arm():
    """control/controller.py pins NAMESPACE at module scope. Read, not assumed,
    so this goes red if the controller ever becomes namespace-configurable
    without the two-arm runner being taught to exclude the control arm."""
    from control import controller
    assert recovery.attest_arm_isolation(controller.NAMESPACE).ok


def test_shadow_mode_is_not_publishable_as_a_comparison():
    """THE EXPENSIVE FAILURE MODE. Run the experiment with DRY_RUN=True, get
    two indistinguishable arms, report 'Prodrome ties stock Kubernetes' -- when
    what happened is that Prodrome was never switched on. execute_action()
    returns 'dry-run' before touching the API, so the treatment arm IS a second
    control arm and every row is a tautology.
    """
    log = pd.DataFrame([dict(mode="shadow", result="dry-run")] * 5)
    assert not recovery.attest_treatment_arm_live(log).ok
    assert not recovery.attest_treatment_arm_live(None, dry_run=True).ok
    assert recovery.attest_treatment_arm_live(None, dry_run=False).ok


def test_live_mode_with_no_executed_action_is_also_not_publishable():
    """mode=live is necessary and not sufficient. A log full of
    blocked-cooldown or blocked-kill-switch rows means no remediation was
    applied, which is the same tautology wearing a different label."""
    log = pd.DataFrame([dict(mode="live", result="blocked-cooldown")] * 5)
    assert not recovery.attest_treatment_arm_live(log).ok
    log2 = pd.DataFrame([dict(mode="live", result="executed")])
    assert recovery.attest_treatment_arm_live(log2).ok


def test_a_log_without_a_mode_column_is_not_publishable():
    """SETUP.md 7 requires `mode`. Without it shadow cannot be told from live,
    so the control-arm question is unanswerable from the log -- which is
    exactly why `mode` was added to the contract."""
    log = pd.DataFrame([dict(result="executed")])
    assert not recovery.attest_treatment_arm_live(log).ok


def test_the_repo_as_it_stands_is_not_publishable():
    """Documents tonight's actual state rather than asserting a hope.

    DRY_RUN is True in control/controller.py, so the treatment arm cannot act
    and no two-arm run made right now may be reported as a comparison. If
    cluster's seat flips DRY_RUN to False this test fails, which is the signal
    to re-read the attestation output and update this expectation deliberately
    -- not a bug.
    """
    from control import controller
    assert controller.DRY_RUN is True
    atts = recovery.attest(controller_namespace=controller.NAMESPACE,
                           dry_run=controller.DRY_RUN)
    assert not recovery.publishable(atts)
    failed = [a.name for a in atts if not a.ok]
    assert failed == ["treatment-arm-live"], (
        f"expected ONLY the dry-run attestation to fail; also failing: {failed}")


def test_load_parity_must_be_attested():
    """An idle control arm recovers faster for reasons that have nothing to do
    with Prodrome. The runner records the attestation rather than inferring it,
    and an un-attested campaign is not publishable."""
    assert recovery.attest_load_parity(pair("P_001", "POD_KILL", load_attested=True)).ok
    assert not recovery.attest_load_parity(
        pair("P_001", "POD_KILL", load_attested=False)).ok


def test_render_states_the_limitations_next_to_the_numbers():
    """GROUND RULE 7. The caveats ship in the report itself, not in a README
    nobody opens next to the number they are quoting."""
    out = recovery.render(pd.DataFrame(), pd.DataFrame(),
                          [recovery.Attestation("x", False, "because")])
    assert "PUBLISHABLE AS A CONTROL-ARM COMPARISON: NO" in out
    assert "alert_only" in out and "failure_ts is null" in out
    assert "lower bound" in out


# --- end to end over a whole fake campaign ---------------------------------

def test_end_to_end_null_experiment_shows_no_difference():
    """The shape of tonight's real run: DRY_RUN=True, so both arms behave
    identically. The harness must come back with a delta of 0 and refuse to
    publish. A null experiment that showed a difference would mean the harness
    itself is biased -- so this is the self-check, not a result.
    """
    frames, pairs = [], []
    for i in range(4):
        start = T0 + pd.Timedelta(minutes=20 * i)
        pairs.append(pair(f"PK_{i:03d}", "POD_KILL", start=start))
        for arm in (TREATMENT_ARM, CONTROL_ARM):
            frames.append(health(arm, "redis",
                                 [True] * 4 + [False] * 5 + [True] * 12, start=start))
    m = recovery.measure(pd.concat(pairs, ignore_index=True),
                         pd.concat(frames, ignore_index=True))
    d = recovery.paired_delta_table(m).iloc[0]
    assert d["pairs_in_delta"] == 4
    assert d["median_delta_s"] == 0.0
    assert d["prodrome_faster_pairs"] == 0 and d["control_faster_pairs"] == 0
    assert d["tied_pairs"] == 4          # all four within the resolution band
    assert not d["comparable"]          # POD_KILL has no lever -- a designed tie


def main() -> None:
    raise SystemExit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    main()
