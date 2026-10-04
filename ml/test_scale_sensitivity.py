"""Pins the score-before-train fix: the detector must not be scale-invariant.

Before the fix, MetricDetector.update() appended each tick's error to
self.errors and *then* computed _z() against that same distribution, so
every tick was scored against a reference already containing itself. The
observable consequence was that z saturated near 1.91 for every injected
magnitude from 2x to 1e30x, with a sustained ceiling around 3.90 against
Z_THRESHOLD = 4.0 -- Detector.score() was arithmetically incapable of
firing on a step change of any size. Every detector number reported
before 2026-10-03 predates this fix.

This file is the regression guard for that, and for the bound that makes
the fix safe to ship:

  1. z must GROW with the magnitude of the excursion (not saturate).
  2. a large step must fire, and fire on the first scored tick.
  3. the exclusion run must be bounded by SELF_EXCLUDE_CAP, so a genuine
     baseline shift is absorbed instead of alarming forever. This is the
     property cluster's cooldown and collect's harness both depend on:
     firing is guaranteed to stop within SELF_EXCLUDE_CAP ticks of a
     sustained new normal, which is also why a fault window is followed
     by a tail of up to SELF_EXCLUDE_CAP ticks of continued firing.

Deliberately uses synthetic constant-plus-step series rather than
data/healthy/metrics.parquet, which does not exist on disk (the missing
artifact that blocks ml.fit_detector and ml.export_firing_log).

Run: python -m ml.test_scale_sensitivity
"""

from ml.detector import (
    SELF_EXCLUDE_CAP,
    WARMUP_TICKS,
    MetricDetector,
    Z_THRESHOLD,
)

BASE = 100.0
JITTER = 1.0  # a flat series has std 0; give the reference something to measure


def warmed() -> MetricDetector:
    """A detector with a full, healthy reference: alternating +/- jitter so
    the error history has non-degenerate spread."""
    d = MetricDetector()
    for i in range(WARMUP_TICKS * 3):
        d.update(BASE + (JITTER if i % 2 else -JITTER))
    assert len(d.errors) >= WARMUP_TICKS, "reference never filled"
    return d


def z_for_step(multiplier: float) -> float:
    """z on the first tick of a step change `multiplier` x the baseline."""
    d = warmed()
    z, _ = d.update(BASE * multiplier)
    assert z is not None
    return z


def test_z_grows_with_magnitude():
    """The pre-fix detector returned the same z (~1.91) for every magnitude.
    Post-fix, z must be strictly increasing in the size of the excursion."""
    multipliers = [2, 10, 100, 10**6, 10**12, 10**20]
    zs = [z_for_step(m) for m in multipliers]
    for m, z in zip(multipliers, zs):
        print(f"  {m:>8.0e}x -> z = {z:.4g}")
    assert all(b > a for a, b in zip(zs, zs[1:])), (
        f"z is not monotonic in excursion magnitude -- scale invariance is "
        f"back: {zs}"
    )
    assert zs[-1] > 1e6, f"z saturated at {zs[-1]:.4g} on a 1e20x step"
    assert zs[0] > Z_THRESHOLD, (
        f"even a 2x step must clear the threshold on a quiet baseline; got "
        f"z={zs[0]:.4g} vs threshold {Z_THRESHOLD}"
    )


def test_large_step_fires_immediately():
    """A step change must fire on the first tick it is scored, not after the
    reference has been polluted into accepting it."""
    d = warmed()
    z, fired = d.update(BASE * 1000)
    print(f"  first tick of a 1000x step: z={z:.4g} fired={fired}")
    assert fired, f"a 1000x step did not fire (z={z:.4g})"


def test_exclusion_run_is_bounded():
    """A sustained shift must stop firing within SELF_EXCLUDE_CAP ticks.

    Without a cap, withholding every anomalous tick from the reference means
    a deliberate scale-up alarms forever. With it, the reference starts
    absorbing again -- which is exactly why a fault window is followed by a
    firing tail of up to SELF_EXCLUDE_CAP ticks.
    """
    d = warmed()
    fired = [d.update(BASE * 50)[1] for _ in range(SELF_EXCLUDE_CAP * 4)]
    assert fired[0], "the shift never fired at all"
    last = max(i for i, f in enumerate(fired) if f)
    print(f"  sustained 50x shift: fired on {sum(fired)} of {len(fired)} ticks, "
          f"last firing at tick {last}, cap={SELF_EXCLUDE_CAP}")
    assert not any(fired[SELF_EXCLUDE_CAP * 3 :]), (
        "still firing well past SELF_EXCLUDE_CAP ticks of a sustained shift -- "
        "a genuine baseline change would alarm indefinitely"
    )


def test_healthy_baseline_does_not_fire():
    """The flip side: a quiet series must not fire. Guards against 'fixing'
    sensitivity by making everything anomalous."""
    d = warmed()
    fired = [d.update(BASE + (JITTER if i % 2 else -JITTER))[1] for i in range(200)]
    print(f"  200 further healthy ticks: fired {sum(fired)} times")
    assert not any(fired), f"fired {sum(fired)} times on its own baseline"


def main():
    for fn in (
        test_z_grows_with_magnitude,
        test_large_step_fires_immediately,
        test_exclusion_run_is_bounded,
        test_healthy_baseline_does_not_fire,
    ):
        print(f"{fn.__name__}:")
        fn()
        print("  OK\n")
    print("all scale-sensitivity checks passed")


if __name__ == "__main__":
    main()
