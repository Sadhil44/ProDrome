"""Offline tests for the replay-level positive control.

`eval/sensitivity.py` exists to answer "can this harness see an effect that is
really there." These tests answer the question one level down: can the
POSITIVE CONTROL ITSELF be trusted to put the effect in? A sensitivity curve
produced by a broken injector is worse than no curve, because it would license
exactly the claim the whole exercise is meant to make honest.

So the cases here are mostly about the injector's own integrity:

  * it must delay, never advance, and never move the FAILURE instant;
  * it must put in the size it says it put in, or report that it could not;
  * it must never write outside the pair it is injecting into;
  * injecting zero must change nothing at all, byte for byte;
  * and the sign must come from the data, so hurting the treatment arm has to
    produce the opposite sign from hurting the control arm.

No cluster, no files under `data/`, fixtures built in-process -- same contract
as `eval/test_recovery.py`, whose builders this file reuses rather than
duplicating.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import recovery, sensitivity  # noqa: E402
from eval.recovery import CONTROL_ARM, TREATMENT_ARM  # noqa: E402
from eval.test_recovery import STEP, T0, health, pair  # noqa: E402


def outage_pair(pair_id: str = "PK_001", down: int = 4, trail: int = 40,
                workload: str = "redis") -> tuple[pd.DataFrame, pd.DataFrame]:
    """A symmetric POD_KILL pair: both arms down for `down` samples, then up.

    Symmetric on purpose. Every test below asks what the INJECTION did, and a
    fixture with a pre-existing difference would make that unanswerable.
    """
    serving = [True] * 4 + [False] * down + [True] * trail
    h = pd.concat([health(a, workload, serving) for a in (TREATMENT_ARM, CONTROL_ARM)],
                  ignore_index=True)
    p = pair(pair_id, "POD_KILL", workload=workload, start=T0 + 4 * STEP)
    return p, h


# --- the injector's integrity ----------------------------------------------

def test_zero_injection_changes_nothing():
    """The N=0 point anchors the whole curve. If it perturbs the series, every
    size above it is measured against a moved baseline."""
    p, h = outage_pair()
    out, book = sensitivity.inject_delay(p, h, 0.0, CONTROL_ARM)
    before = recovery.measure(p, h)
    after = recovery.measure(p, out)
    pd.testing.assert_series_equal(before["recovery_seconds"],
                                   after["recovery_seconds"])
    assert not book["applied"].any()
    assert book["rows_rewritten"].sum() == 0


def test_injection_lands_on_the_arm_asked_for_and_only_that_arm():
    """The untouched arm is the reference. If the injector leaks into it the
    measured delta is not the manufactured one and nothing downstream knows."""
    p, h = outage_pair()
    out, _ = sensitivity.inject_delay(p, h, 30.0, CONTROL_ARM)
    for col in ("ready_replicas", "probe_ok"):
        left = h[h["arm"] == TREATMENT_ARM][col].astype(str).tolist()
        right = out[out["arm"] == TREATMENT_ARM][col].astype(str).tolist()
        assert left == right, f"{col} changed on the arm that was not hurt"
    assert (out[out["arm"] == CONTROL_ARM]["ready_replicas"] == 0).sum() > \
           (h[h["arm"] == CONTROL_ARM]["ready_replicas"] == 0).sum()


def test_injection_delays_recovery_and_does_not_move_the_failure_instant():
    """Recovery time is `t_recovered - t_fail`. An injector that moved `t_fail`
    EARLIER would inflate the delta without delaying anything, which is the
    cheapest way to manufacture a positive control that proves nothing.
    """
    p, h = outage_pair()
    out, _ = sensitivity.inject_delay(p, h, 30.0, CONTROL_ARM)
    before = recovery.measure(p, h).set_index("arm")
    after = recovery.measure(p, out).set_index("arm")

    assert after.loc[CONTROL_ARM, "t_fail"] == before.loc[CONTROL_ARM, "t_fail"]
    grew = (after.loc[CONTROL_ARM, "recovery_seconds"]
            - before.loc[CONTROL_ARM, "recovery_seconds"])
    assert grew >= 30.0
    assert after.loc[TREATMENT_ARM, "recovery_seconds"] == \
        before.loc[TREATMENT_ARM, "recovery_seconds"]


def test_recovered_delta_has_the_size_that_was_injected():
    """The headline property. Within one sampling interval, because a recovery
    is only ever observed at the first sample after it happens."""
    p, h = outage_pair(trail=60)
    row = sensitivity.sweep_point(p, h, 30.0, CONTROL_ARM, tie_band=5.0).iloc[0]
    assert row["recovered_median_delta_s"] <= -30.0
    assert row["magnitude_error_s"] <= STEP.total_seconds()
    assert row["correct_direction_pairs"] == 1
    assert row["wrong_direction_pairs"] == 0


def test_the_sign_comes_from_the_data_not_from_which_arm_is_polled_first():
    """`collect/twoarm.py`'s sampler polls prodrome before control, and the
    POD_KILL millisecond trap showed that ordering alone can produce a
    consistent sign. Hurting the treatment arm must therefore flip the sign,
    not merely change the magnitude."""
    p, h = outage_pair(trail=60)
    hurt_control = sensitivity.sweep_point(p, h, 30.0, CONTROL_ARM, tie_band=5.0).iloc[0]
    hurt_treat = sensitivity.sweep_point(p, h, 30.0, TREATMENT_ARM, tie_band=5.0).iloc[0]
    assert hurt_control["recovered_median_delta_s"] < 0
    assert hurt_treat["recovered_median_delta_s"] > 0
    assert hurt_treat["correct_direction_pairs"] == 1


def test_delay_is_clipped_at_the_next_pair_on_the_same_workload():
    """Two pairs back to back on one workload. A 10-minute delay on the first
    must not still be running when the second is injected, or the second pair's
    failure instant is the first pair's manufactured one."""
    serving = [True] * 4 + [False] * 4 + [True] * 10 + [False] * 4 + [True] * 20
    h = pd.concat([health(a, "redis", serving) for a in (TREATMENT_ARM, CONTROL_ARM)],
                  ignore_index=True)
    p = pd.concat([pair("PK_001", "POD_KILL", start=T0 + 4 * STEP),
                   pair("PK_002", "POD_KILL", start=T0 + 18 * STEP)],
                  ignore_index=True)
    out, book = sensitivity.inject_delay(p, h, 600.0, CONTROL_ARM)
    first = book[book["pair_id"] == "PK_001"].iloc[0]
    assert first["clipped"]
    assert first["landed_s"] < 600.0

    # The bound is the second pair's injection start, exactly -- and the first
    # pair's manufactured outage therefore ends before the second begins, so
    # the second pair's failure instant is still its own.
    t_rec = recovery.measure(p.head(2), h).set_index("arm").loc[CONTROL_ARM, "t_recovered"]
    assert t_rec + pd.Timedelta(seconds=first["landed_s"]) <= T0 + 18 * STEP
    after = recovery.measure(p, out).set_index(["pair_id", "arm"])
    assert after.loc[("PK_002", CONTROL_ARM), "t_fail"] == \
        recovery.measure(p, h).set_index(["pair_id", "arm"]).loc[
            ("PK_002", CONTROL_ARM), "t_fail"]


def test_degradation_injection_does_not_move_the_band_under_itself():
    """The band is built from PRE-injection samples. If the injector wrote into
    the baseline the band would rise with the injection and the degraded span
    would not grow -- a positive control that silently measures nothing."""
    lat = [10.0] * 6 + [900.0] * 4 + [10.0] * 40
    h = pd.concat([health(a, "redis", [True] * len(lat), latency=lat)
                   for a in (TREATMENT_ARM, CONTROL_ARM)], ignore_index=True)
    p = pair("CH_001", "CPU_HOG", start=T0 + 6 * STEP)

    base_before = recovery._slice(h, CONTROL_ARM, "redis", hi=T0 + 6 * STEP)
    out, _ = sensitivity.inject_delay(p, h, 30.0, CONTROL_ARM)
    base_after = recovery._slice(out, CONTROL_ARM, "redis", hi=T0 + 6 * STEP)
    assert recovery.latency_band(base_before) == recovery.latency_band(base_after)

    m = recovery.measure(p, out).set_index("arm")
    grew = (m.loc[CONTROL_ARM, "degradation_seconds"]
            - recovery.measure(p, h).set_index("arm").loc[CONTROL_ARM,
                                                          "degradation_seconds"])
    assert grew >= 30.0


# --- the window bound this work uncovered ----------------------------------

def test_open_window_lets_a_later_fault_become_this_pairs_recovery_time():
    """The defect `pair_window_hi` documents, pinned so it cannot come back
    unnoticed.

    One workload, two faults, with a quiet gap between them SHORTER than the
    30s sustained-hold -- which is the case that bites. With the window left
    open the first pair cannot find a long enough quiet run inside its own
    trail, keeps looking, finds one only after the SECOND fault has come and
    gone, and reports that as its recovery: a number spanning both faults,
    where the truth is that this pair never recovered while we were watching.

    Bounded at its own trail the same pair comes back STILL_DEGRADED, which is
    censored and gets excluded. Note which way that cuts: the open window turns
    a censored pair into a fabricated recovery time, so it does not merely add
    noise, it replaces a known-unknown with a wrong number. This is why
    `eval/sensitivity.py` sweeps with the bound on.
    """
    lat = [10.0] * 6 + [900.0] * 4 + [10.0] * 5 + [900.0] * 6 + [10.0] * 30
    h = pd.concat([health(a, "redis", [True] * len(lat), latency=lat)
                   for a in (TREATMENT_ARM, CONTROL_ARM)], ignore_index=True)
    p = pair("CH_001", "CPU_HOG", start=T0 + 6 * STEP)
    p["trail_s"] = 40.0
    p["end_ts"] = T0 + 10 * STEP

    open_win = recovery.measure(p, h).iloc[0]
    bounded = recovery.measure(p, h, window_seconds="trail").iloc[0]
    assert open_win["outcome"] == recovery.DEGRADED
    assert open_win["degradation_seconds"] >= 75.0       # spans the second fault
    assert bounded["outcome"] == recovery.STILL_DEGRADED
    assert bounded["degradation_seconds"] is None


def test_window_bound_defaults_to_open_so_published_numbers_reproduce():
    """The 2026-10-06 campaign was measured with the window open. Changing the
    default would silently restate every number already on the forum."""
    p, h = outage_pair()
    assert recovery.pair_window_hi(p.iloc[0], None) is None


def test_trail_bound_falls_back_to_open_on_a_pairs_csv_without_trail_s():
    """The first live two-arm run predates `trail_s`. It must still load."""
    p, h = outage_pair()
    row = p.iloc[0].copy()
    row["trail_s"] = None
    assert recovery.pair_window_hi(row, "trail") is None


# --- the headline, and the refusal that has to survive it ------------------

def test_minimum_detectable_effect_needs_the_whole_tail_to_pass():
    """A size that passes while a larger one fails is a wobble, not a
    threshold. Reporting it would hand the project a number that does not
    hold at the sizes it actually cares about."""
    curve = pd.DataFrame([
        dict(fault_type="X", kind="outage", manufactured_effect_s=s,
             pairs=10, power=pw, magnitude_error_s=0.0, excluded_censored=0,
             tie_band_s=5.0, recovered_median_delta_s=-s)
        for s, pw in [(5.0, 1.0), (10.0, 0.2), (20.0, 1.0), (40.0, 1.0)]])
    out = sensitivity.minimum_detectable_effect(curve).iloc[0]
    assert out["min_detectable_effect_s"] == 20.0


def test_blindness_is_reported_as_a_value_not_as_an_absence():
    """A fault type the harness cannot see at any size must come back as a
    stated NaN with `detected_any_size=False`, not as a missing row. A missing
    row reads as "not tested"; this is "tested and blind"."""
    curve = pd.DataFrame([
        dict(fault_type="X", kind="degradation", manufactured_effect_s=s,
             pairs=10, power=0.3, magnitude_error_s=99.0, excluded_censored=0,
             tie_band_s=5.0, recovered_median_delta_s=-1.0)
        for s in (10.0, 50.0, 100.0)])
    out = sensitivity.minimum_detectable_effect(curve).iloc[0]
    assert pd.isna(out["min_detectable_effect_s"])
    assert out["detected_any_size"] is False or not out["detected_any_size"]


def test_a_manufactured_effect_never_makes_the_campaign_publishable():
    """The standing refusal. `publishable()` is about whether the EXPERIMENT
    could have produced a comparison, and injecting a difference into the
    health series does not change the answer -- if it did, this module would be
    a way to launder a null run into a result."""
    p, h = outage_pair()
    hurt, _ = sensitivity.inject_delay(p, h, 60.0, CONTROL_ARM)
    atts = recovery.attest(pairs=p, decisions=None, dry_run=True,
                           controller_namespace="prodrome")
    assert not recovery.publishable(atts)


def test_every_curve_row_carries_the_manufactured_size():
    """A reader who sees only the CSV still has to be able to tell that the
    effect was put there on purpose."""
    p, h = outage_pair(trail=60)
    curve = sensitivity.sweep(p, h, sizes=(0.0, 10.0, 30.0), tie_band=5.0)
    assert "manufactured_effect_s" in curve.columns
    assert set(curve["manufactured_effect_s"]) == {0.0, 10.0, 30.0}
    assert set(curve["hurt_arm"]) == {CONTROL_ARM}


def test_sign_test_p_is_exact_and_two_sided():
    assert sensitivity._sign_test_p(9, 9) == pytest.approx(2 / 512)
    assert sensitivity._sign_test_p(5, 10) == pytest.approx(1.0)
    assert sensitivity._sign_test_p(0, 0) is None




# --- the resolution that actually applies ----------------------------------

def test_cadence_is_reported_separately_for_serving_and_not_serving():
    """The live positive control's finding, pinned.

    `collect/twoarm.kubectl_probe` runs `kubectl exec` unconditionally, so when
    a deployment has no pod the probe blocks for its full timeout and the
    sampler slows down by ~2.6x -- exactly while the outage it is measuring is
    happening. A single all-sample median hides that completely, and the tie
    band is derived from it.
    """
    fast, slow = pd.Timedelta(seconds=5), pd.Timedelta(seconds=15)
    ts, t = [], T0
    for step in [fast] * 6 + [slow] * 4 + [fast] * 6:
        ts.append(t)
        t += step
    serving = [True] * 6 + [False] * 4 + [True] * 6
    h = pd.DataFrame([
        dict(ts=ts[i], arm=TREATMENT_ARM, workload="redis",
             ready_replicas=1 if s else 0, desired_replicas=1,
             restarts_total=0, probe_ok=s, probe_latency_ms=10.0)
        for i, s in enumerate(serving)], columns=recovery.HEALTH_COLUMNS)

    c = sensitivity.cadence_by_state(h).set_index("serving")
    assert c.loc[False, "median_gap_s"] > c.loc[True, "median_gap_s"]
    # And the honest tie band is the not-serving one, not the all-sample median.
    assert sensitivity.outage_resolution(h) > recovery.sampling_resolution(h)


def test_outage_resolution_falls_back_when_nothing_ever_stopped_serving():
    """Degradation faults never leave the serving state, so the all-sample
    median IS the right number there and must not be replaced by a NaN."""
    h = health(TREATMENT_ARM, "redis", [True] * 12)
    assert sensitivity.outage_resolution(h) == pytest.approx(
        recovery.sampling_resolution(h))


def main() -> int:
    return pytest.main([__file__, "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
