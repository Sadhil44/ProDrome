"""Offline tests for the LIVE positive control.

`collect/poscontrol.py` is the only file in this repo that deliberately makes
one arm worse than the other. That is a legitimate thing to do -- it is the
only way to find out whether the harness can see a difference at all -- and it
is also exactly the shape of the mistake the whole two-arm design exists to
prevent. So the tests here are weighted toward containment:

  * the asymmetry must be the one that was asked for, and nothing else;
  * both arms must go down TOGETHER, or the structural costs of an outage stop
    cancelling and the "known" delta is not known;
  * the rows must be written so that no downstream reader can mistake a
    manufactured effect for a Prodrome result;
  * and no failure path may leave a namespace at zero replicas.

Nothing here touches a cluster: `scale_fn`, `sleep` and the clock are injected.
That is also the point of those seams existing.
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

from collect import poscontrol  # noqa: E402
from collect.twoarm import ARM_NAMESPACE, CONTROL_ARM, TREATMENT_ARM  # noqa: E402
from eval import recovery  # noqa: E402

T0 = pd.Timestamp("2026-10-13T12:00:00Z")


class FakeCluster:
    """Records every scale call with the simulated time it happened at.

    Simulated, not wall clock: each arm runs on its own thread and the holds
    are the thing under test, so a real `sleep` would make the test both slow
    and timing-dependent. `sleep` advances this arm's own clock instead.
    """

    def __init__(self, fail: set[str] | None = None):
        self.calls: list[tuple[float, str, str, int]] = []
        self.fail = fail or set()
        self._lock = threading.Lock()
        self._t = threading.local()

    @property
    def now(self) -> float:
        return getattr(self._t, "v", 0.0)

    def sleep(self, s: float) -> None:
        self._t.v = self.now + float(s)

    def clock(self) -> pd.Timestamp:
        return T0 + pd.Timedelta(seconds=self.now)

    def scale(self, namespace: str, workload: str, replicas: int) -> bool:
        with self._lock:
            self.calls.append((self.now, namespace, workload, replicas))
        return namespace not in self.fail

    def down_at(self, ns: str) -> float:
        return next(t for t, n, _, r in self.calls if n == ns and r == 0)

    def up_at(self, ns: str) -> float:
        return next(t for t, n, _, r in self.calls if n == ns and r == 1)


# --- the asymmetry is the one that was asked for ---------------------------

def test_only_the_hurt_arm_is_held_longer_and_by_exactly_the_effect():
    fc = FakeCluster()
    poscontrol.hold_both("nginx", base_hold_s=20.0, effect_s=30.0,
                         hurt_arm=CONTROL_ARM, scale_fn=fc.scale,
                         sleep=fc.sleep, clock=fc.clock)
    held = {ns: fc.up_at(ns) - fc.down_at(ns) for ns in ARM_NAMESPACE.values()}
    assert held[ARM_NAMESPACE[TREATMENT_ARM]] == pytest.approx(20.0)
    assert held[ARM_NAMESPACE[CONTROL_ARM]] == pytest.approx(50.0)
    assert (held[ARM_NAMESPACE[CONTROL_ARM]]
            - held[ARM_NAMESPACE[TREATMENT_ARM]]) == pytest.approx(30.0)


def test_hurting_the_treatment_arm_flips_which_arm_is_held_longer():
    """The sign check has to be a real inversion, not a relabelling."""
    fc = FakeCluster()
    poscontrol.hold_both("nginx", 20.0, 30.0, hurt_arm=TREATMENT_ARM,
                         scale_fn=fc.scale, sleep=fc.sleep, clock=fc.clock)
    held = {ns: fc.up_at(ns) - fc.down_at(ns) for ns in ARM_NAMESPACE.values()}
    assert held[ARM_NAMESPACE[TREATMENT_ARM]] == pytest.approx(50.0)
    assert held[ARM_NAMESPACE[CONTROL_ARM]] == pytest.approx(20.0)


def test_zero_effect_holds_both_arms_for_the_same_time():
    """The live null. Any difference measured from a zero-effect pair is the
    harness inventing one."""
    fc = FakeCluster()
    poscontrol.hold_both("nginx", 20.0, 0.0, scale_fn=fc.scale,
                         sleep=fc.sleep, clock=fc.clock)
    held = {ns: fc.up_at(ns) - fc.down_at(ns) for ns in ARM_NAMESPACE.values()}
    assert held[ARM_NAMESPACE[TREATMENT_ARM]] == held[ARM_NAMESPACE[CONTROL_ARM]]


def test_both_arms_go_down_before_either_comes_back():
    """The barrier. If one arm were released before the other was even taken
    down, the pair would not share a cluster state and the structural cost of
    an outage would stop cancelling in the difference."""
    fc = FakeCluster()
    poscontrol.hold_both("nginx", 20.0, 30.0, scale_fn=fc.scale,
                         sleep=fc.sleep, clock=fc.clock)
    downs = [c for c in fc.calls if c[3] == 0]
    first_up = min(c[0] for c in fc.calls if c[3] == 1)
    assert len(downs) == 2
    assert all(t <= first_up for t, _, _, _ in downs)


# --- containment: no failure path leaves an arm at zero --------------------

def test_an_arm_whose_release_fails_is_still_scaled_back_up():
    fc = FakeCluster(fail={ARM_NAMESPACE[CONTROL_ARM]})
    res = poscontrol.hold_both("nginx", 5.0, 5.0, scale_fn=fc.scale,
                               sleep=fc.sleep, clock=fc.clock)
    assert not res[CONTROL_ARM].inject_ok
    assert res[CONTROL_ARM].error
    ups = [c for c in fc.calls
           if c[1] == ARM_NAMESPACE[CONTROL_ARM] and c[3] == 1]
    assert ups, "a failed arm must still have a scale-to-1 attempted"


def test_one_arm_failing_does_not_stop_the_other_from_completing():
    """Same rule `collect/twoarm.inject_both` learned the hard way: a half-pair
    is still worth recording, and recording it as a half-pair is how it gets
    excluded rather than silently counted."""
    fc = FakeCluster(fail={ARM_NAMESPACE[CONTROL_ARM]})
    res = poscontrol.hold_both("nginx", 5.0, 5.0, scale_fn=fc.scale,
                               sleep=fc.sleep, clock=fc.clock)
    assert res[TREATMENT_ARM].inject_ok
    assert not res[CONTROL_ARM].inject_ok


def test_restore_puts_every_arm_back_and_is_idempotent():
    fc = FakeCluster()
    assert all(poscontrol.restore(["nginx"], scale_fn=fc.scale).values())
    assert all(poscontrol.restore(["nginx"], scale_fn=fc.scale).values())
    assert all(r == 1 for _, _, _, r in fc.calls)
    assert {n for _, n, _, _ in fc.calls} == set(ARM_NAMESPACE.values())


# --- containment: the rows cannot be read as a Prodrome result -------------

def rows_for(effect_s: float, hurt_arm: str = CONTROL_ARM) -> pd.DataFrame:
    fc = FakeCluster()
    res = poscontrol.hold_both("nginx", 20.0, effect_s, hurt_arm,
                               scale_fn=fc.scale, sleep=fc.sleep, clock=fc.clock)
    return pd.DataFrame(poscontrol.poscontrol_rows(
        "POSCTL_c030s_001", "nginx", effect_s, 20.0, hurt_arm, res,
        settle=None, trail_s=150.0))


def test_rows_are_marked_not_comparable_so_a_table_reader_cannot_misread_them():
    """The containment that matters most. A reader who sees only the output
    table must still see that this row could not have been a Prodrome win."""
    df = rows_for(30.0)
    assert set(df["fault_type"]) == {poscontrol.FAULT_TYPE}
    assert recovery.lever_for(poscontrol.FAULT_TYPE) == "nothing"
    t = recovery.per_fault_arm_table(
        pd.DataFrame(dict(fault_type=[poscontrol.FAULT_TYPE] * 2,
                          arm=[TREATMENT_ARM, CONTROL_ARM],
                          outcome=[recovery.RECOVERED] * 2,
                          recovery_seconds=[10.0, 40.0],
                          degradation_seconds=[None, None])))
    assert not t["comparable"].any()


def test_rows_record_the_asymmetry_rather_than_leaving_it_in_the_arguments():
    """`duration_s` is per arm and `pattern` carries the size, so the file is
    self-describing: a pairs.csv found a month from now still says what was
    done to which arm."""
    df = rows_for(30.0).set_index("arm")
    assert df.loc[TREATMENT_ARM, "duration_s"] == 20
    assert df.loc[CONTROL_ARM, "duration_s"] == 50
    assert "effect30s" in df.loc[CONTROL_ARM, "pattern"]
    assert f"hurt_{CONTROL_ARM}" in df.loc[CONTROL_ARM, "pattern"]
    assert set(df["trail_s"]) == {150.0}
    assert not df["load_attested"].any()


def test_rows_have_exactly_the_columns_eval_recovery_reads():
    assert list(rows_for(0.0).columns) == recovery.PAIRS_COLUMNS


def test_output_goes_somewhere_the_real_campaign_cannot_pick_it_up():
    """Manufactured pairs and real pairs must never share a directory, or a
    later `python -m eval.recovery --indir data/twoarm` silently includes
    them."""
    from collect import twoarm
    assert poscontrol.OUTDIR != twoarm.OUTDIR
    assert "poscontrol" in poscontrol.OUTDIR


def test_pair_ids_say_which_arm_was_hurt_and_by_how_much():
    assert poscontrol.pair_id_for(30.0, 2, CONTROL_ARM) == "POSCTL_c030s_002"
    assert poscontrol.pair_id_for(30.0, 2, TREATMENT_ARM) == "POSCTL_p030s_002"
    assert (poscontrol.pair_id_for(0.0, 1, CONTROL_ARM)
            != poscontrol.pair_id_for(0.0, 1, TREATMENT_ARM))


# --- end to end against the real measurement -------------------------------

def test_the_harness_recovers_the_manufactured_delta_from_a_simulated_series():
    """Closes the loop offline: holds -> health series -> `recovery.measure`.

    The series is built from the holds this module chose, at the cadence the
    live sampler actually achieves (6.82s, not the 5s it asks for), so the
    quantisation that dominates the noise floor is present here too. The
    recovered delta must be the effect, give or take one sample.
    """
    cadence, base, effect, lead, trail = 6.82, 20.0, 30.0, 45.0, 150.0
    start = T0 + pd.Timedelta(seconds=lead)
    holds = {TREATMENT_ARM: base, CONTROL_ARM: base + effect}

    rows = []
    n = int((lead + base + effect + trail) / cadence)
    for i in range(n):
        ts = T0 + pd.Timedelta(seconds=i * cadence)
        for arm, hold in holds.items():
            down = start <= ts < start + pd.Timedelta(seconds=hold)
            rows.append(dict(ts=ts, arm=arm, workload="nginx",
                             ready_replicas=0 if down else 1,
                             desired_replicas=1, restarts_total=0,
                             probe_ok=not down, probe_latency_ms=700.0))
    health = pd.DataFrame(rows, columns=recovery.HEALTH_COLUMNS)

    pairs = pd.DataFrame(poscontrol.poscontrol_rows(
        "POSCTL_c030s_001", "nginx", effect, base, CONTROL_ARM,
        {a: type("R", (), dict(arm=a, namespace=ARM_NAMESPACE[a],
                               start_ts=start, end_ts=start + pd.Timedelta(seconds=holds[a]),
                               failure_ts=start, restarts_delta=0,
                               inject_ok=True, error=""))()
         for a in (TREATMENT_ARM, CONTROL_ARM)},
        settle=None, trail_s=trail))

    d = recovery.paired_delta_table(recovery.measure(pairs, health),
                                    tie_band=cadence).iloc[0]
    assert d["pairs_in_delta"] == 1
    assert d["prodrome_faster_pairs"] == 1
    assert d["control_faster_pairs"] == 0
    assert d["median_delta_s"] == pytest.approx(-effect, abs=2 * cadence)
    assert not d["comparable"]


def main() -> int:
    return pytest.main([__file__, "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
