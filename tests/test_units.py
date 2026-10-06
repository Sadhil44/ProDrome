"""Unit tests for the individual stages: features, windowing, labelling, detector.

Each maps to a published contract (signal's window-and-feature contract,
SETUP.md §7), a ground rule in SETUP.md §10, or a defect that actually happened
here. Nothing speculative, and no accuracy thresholds -- shapes, orders,
boundaries and the arithmetic itself.

Everything reads data/samples/ or a frame built in the test, so it runs from a
fresh clone (ground rule 8).
"""

import numpy as np
import pandas as pd
import pytest

from control import policy
from ml.dataset import build_dataset
from ml.detector import (
    ERROR_HISTORY,
    MAX_MODE_FRACTION,
    RECENT_WINDOW,
    POST_RESTART_SUPPRESS_TICKS,
    SELF_EXCLUDE_CAP,
    WARMUP_TICKS,
    Z_THRESHOLD,
    Detector,
    MetricDetector,
    WorkloadDetector,
)
from ml.features import METRICS, WINDOW_SIZE, summarize, windows

from conftest import ACTION_VOCABULARY


def _frame(n, start="2026-01-01", workload="redis", values=None):
    """A metrics table of n ticks for one workload, shaped like SETUP.md §7."""
    if values is None:
        values = {m: [float(i) for i in range(n)] for m in METRICS}
    return pd.DataFrame({
        "ts": pd.date_range(start, periods=n, freq="15s", tz="UTC"),
        "workload": [workload] * n,
        **values,
    })


# --- ml/features.py: summarize ----------------------------------------------

def test_summarize_computes_the_five_statistics_it_claims():
    """The 40 features are mean/slope/std/max/last per metric (signal's contract).

    Pinned against hand-computable values: both halves of the pipeline summarize
    a window with this function, so if the arithmetic drifts the detector and
    the classifier still agree with each other and are both wrong together --
    which no cross-stage test can catch. Only this one can.
    """
    values = [float(i + 1) for i in range(WINDOW_SIZE)]  # 1..20, slope exactly 1
    window = pd.DataFrame({m: values for m in METRICS})
    features = summarize(window)

    assert features["cpu_cores_mean"] == pytest.approx(10.5)
    assert features["cpu_cores_slope"] == pytest.approx(1.0)
    assert features["cpu_cores_max"] == pytest.approx(20.0)
    assert features["cpu_cores_last"] == pytest.approx(20.0)


def test_summarize_uses_population_std_not_the_sample_std():
    """np.std (ddof=0), not pandas' Series.std (ddof=1) -- they differ by ~2.6%
    at WINDOW_SIZE=20.

    Worth pinning because the obvious reimplementation in another seat's code is
    the pandas one, and a feature that means two slightly different things on
    either side of a stage boundary is the hardest kind of bug to see.
    """
    values = [float(i + 1) for i in range(WINDOW_SIZE)]
    window = pd.DataFrame({m: values for m in METRICS})
    got = summarize(window)["mem_pct_std"]

    assert got == pytest.approx(np.std(values))
    assert got != pytest.approx(pd.Series(values).std())


def test_summarize_slope_separates_a_rise_from_a_fall():
    """ml/train.py's own sanity check is that mem_pct_slope ranks near the top of
    feature importances. That only means anything if the sign is right: a leak
    climbing and a leak recovering must not summarize the same way.
    """
    rising = pd.DataFrame({m: [float(i) for i in range(WINDOW_SIZE)] for m in METRICS})
    falling = pd.DataFrame({m: [float(WINDOW_SIZE - i) for i in range(WINDOW_SIZE)] for m in METRICS})

    assert summarize(rising)["mem_pct_slope"] > 0
    assert summarize(falling)["mem_pct_slope"] < 0
    assert summarize(rising)["mem_pct_last"] > summarize(rising)["mem_pct_mean"]


# --- ml/features.py: windows ------------------------------------------------

def test_a_series_shorter_than_the_window_yields_no_windows():
    """Edge case with teeth: `range(len(sub) - size + 1)` is empty below size.

    It must stay empty rather than emit a short window, because nothing
    downstream validates length -- ml.classifier.predict() will happily
    summarize a 5-row window and return a confident label
    (see tests/test_degradation.py).
    """
    for n in (0, 1, WINDOW_SIZE - 1):
        assert list(windows(_frame(n), "redis", WINDOW_SIZE)) == []


def test_exactly_window_size_rows_yields_exactly_one_window():
    produced = list(windows(_frame(WINDOW_SIZE), "redis", WINDOW_SIZE))
    assert len(produced) == 1
    assert len(produced[0]) == WINDOW_SIZE


def test_consecutive_windows_overlap_by_all_but_one_tick():
    """The fact ground rule 2 rests on. 19 of 20 ticks shared means a random
    train/test split puts near-duplicates on both sides; this is the measurement
    of "near".
    """
    n = WINDOW_SIZE + 5
    produced = list(windows(_frame(n), "redis", WINDOW_SIZE))
    assert len(produced) == n - WINDOW_SIZE + 1

    for earlier, later in zip(produced, produced[1:]):
        shared = set(earlier["ts"]) & set(later["ts"])
        assert len(shared) == WINDOW_SIZE - 1


def test_a_window_never_mixes_two_workloads():
    """`workload` is a stable name shared across runs (SETUP.md §7), so one
    metrics table holds every workload interleaved. A window that spanned two of
    them would be summarized into meaningless features with no error anywhere.
    """
    a = _frame(WINDOW_SIZE + 3, workload="redis")
    b = _frame(WINDOW_SIZE + 3, workload="nginx")
    mixed = pd.concat([a, b], ignore_index=True)

    for window in windows(mixed, "redis", WINDOW_SIZE):
        assert set(window["workload"]) == {"redis"}


def test_windows_sort_by_timestamp_even_when_the_input_is_not_sorted():
    """A Parquet file is not required to be ordered, and slope is computed
    against tick position -- an out-of-order window inverts the sign of the one
    feature the project leans on hardest.
    """
    frame = _frame(WINDOW_SIZE + 4)
    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)

    for window in windows(shuffled, "redis", WINDOW_SIZE):
        assert list(window["ts"]) == sorted(window["ts"])


def test_gaps_are_forward_filled_and_leading_gaps_become_zero():
    """The missing-value rule is decided once, in windows(), per signal's Part 4.1.

    The existing suite asserts a hole does not reach the model as NaN; this pins
    *what value* it becomes, because ffill-then-zero and zero-everywhere produce
    very different slopes and either would pass a not-NaN check.
    """
    n = WINDOW_SIZE + 2
    frame = _frame(n)
    frame.loc[0, "mem_pct"] = None   # leading gap -> 0.0
    frame.loc[5, "mem_pct"] = None   # interior gap -> the value at tick 4

    first = next(iter(windows(frame, "redis", WINDOW_SIZE)))
    assert first["mem_pct"].iloc[0] == 0.0
    assert first["mem_pct"].iloc[5] == first["mem_pct"].iloc[4]


def test_windowing_does_not_mutate_the_callers_metrics_table():
    """windows() fills missing values in place on its own copy. If that copy ever
    stops being a copy, every later consumer of the same frame silently gets
    gap-filled data -- and collect's metrics table is read by three seats.
    """
    frame = _frame(WINDOW_SIZE + 2)
    frame.loc[3, "mem_pct"] = None

    list(windows(frame, "redis", WINDOW_SIZE))

    assert pd.isna(frame.loc[3, "mem_pct"]), "windows() mutated the input frame"


# --- ml/dataset.py: labelling ------------------------------------------------

@pytest.fixture
def labelled():
    """35 ticks for one workload with a fault over ticks 20-24."""
    frame = _frame(35)
    ts = list(frame["ts"])
    labels = pd.DataFrame([{
        "start_ts": ts[20],
        "end_ts": ts[24],
        "workload": "redis",
        "fault_type": "CPU_HOG",
        "pattern": "constant",
        "run_id": "CPU_HOG_constant_redis_000",
    }])
    return frame, labels, ts


def test_a_window_is_labelled_by_its_last_tick_not_its_first(labelled):
    """Ground rule 3 -- label the full fault trajectory, including the faint
    early windows -- is implemented as "the reference tick is window['ts'].iloc[-1]".

    The consequence is specific and easy to get backwards: a window that is
    mostly healthy but ends one tick inside the fault is a fault window, and a
    window that contains the whole fault but ends after end_ts is NORMAL. Both
    are pinned here because eval/harness.py and ml/export_firing_log.py join the
    detector's firing log on that same reference tick.
    """
    frame, labels, ts = labelled
    df = build_dataset(frame, labels).set_index("ts")

    assert df.loc[ts[20], "label"] == "CPU_HOG"   # only its last tick is in the fault
    assert df.loc[ts[24], "label"] == "CPU_HOG"
    assert df.loc[ts[19], "label"] == "NORMAL"    # one tick short of the fault
    assert df.loc[ts[25], "label"] == "NORMAL"    # contains the whole fault, ends after it


def test_normal_windows_carry_no_run_id_and_fault_windows_always_do(labelled):
    """Splits are by run (SETUP.md §7). A fault window with no run_id cannot be
    placed on either side of the split; a NORMAL window has no run to belong to.
    """
    frame, labels, _ = labelled
    df = build_dataset(frame, labels)

    faults = df[df["label"] != "NORMAL"]
    normals = df[df["label"] == "NORMAL"]
    assert not faults.empty and not normals.empty
    assert faults["run_id"].notna().all()
    assert normals["run_id"].isna().all()


def test_seconds_to_failure_counts_down_to_end_ts(labelled):
    """The regression target, and the axis lead time is measured on. It must be
    non-negative inside a fault and absent outside one -- there is no pending
    failure to count down to during a quiet period.
    """
    frame, labels, ts = labelled
    df = build_dataset(frame, labels).set_index("ts")

    assert df.loc[ts[20], "seconds_to_failure"] == ts[24] - ts[20]
    assert df.loc[ts[24], "seconds_to_failure"] == pd.Timedelta(0)
    assert pd.isna(df.loc[ts[19], "seconds_to_failure"])


def test_build_dataset_emits_the_forty_features_plus_its_own_metadata(labelled):
    frame, labels, _ = labelled
    df = build_dataset(frame, labels)

    feature_cols = [c for c in df.columns if c.endswith(("_mean", "_slope", "_std", "_max", "_last"))]
    assert len(feature_cols) == 40
    assert {"workload", "ts", "label", "run_id", "seconds_to_failure"} <= set(df.columns)


# --- control/policy.py ------------------------------------------------------

def test_every_policy_action_is_in_the_vocabulary_cluster_implements():
    """The policy table is the contract between diagnosis and cluster: each value
    has to be an action a controller can execute. A new lever added here without
    a controller branch is a decision that silently does nothing.
    """
    for label, entry in policy.POLICY.items():
        assert entry["action"] in ACTION_VOCABULARY, f"{label} -> unimplementable {entry['action']!r}"


def test_a_row_with_no_confidence_threshold_can_only_mean_do_nothing():
    """NORMAL and UNKNOWN have min_confidence=None, which decide() reads as
    "no bar to clear". Any row with a None threshold therefore acts
    unconditionally -- which is only safe while the action is 'nothing'.
    """
    for label, entry in policy.POLICY.items():
        if entry["min_confidence"] is None:
            assert entry["action"] == "nothing", f"{label} acts unconditionally"


@pytest.mark.parametrize("label", ["CPU_HOG", "MEMORY_LEAK", "DISK_STRESS"])
def test_the_confidence_threshold_is_inclusive(label):
    """The boundary is >=, not >. Pinned because a future refactor that flips it
    changes behaviour on exactly the confidence values the table names, and
    nothing else in the suite would notice.
    """
    bar = policy.POLICY[label]["min_confidence"]
    assert policy.decide(label, bar) == policy.POLICY[label]["action"]
    assert policy.decide(label, bar - 0.01) == "nothing"


# --- ml/detector.py ---------------------------------------------------------

def _warm(detector, ticks=WARMUP_TICKS + 90, seed=0):
    """Stream healthy-looking noise through update() -- for this detector,
    streaming healthy data through once *is* the fit (signal's Part 2.5)."""
    rng = np.random.default_rng(seed)
    for value in 100.0 + rng.normal(0.0, 1.0, ticks):
        detector.update(float(value))
    return detector


def test_a_metric_does_not_score_until_it_is_calibrated():
    """WARMUP_TICKS exists so a detector cannot fire off two observations.

    The boundary sits two ticks later than the constant suggests, for two
    separate reasons that each cost one tick:
      - the first observation produces no error at all (nothing to compare to);
      - update() now scores BEFORE folding the tick into the reference, so the
        tick being scored no longer counts toward its own calibration.
    The second is new, and deliberate: it is the fix for update() having been
    arithmetically unable to fire (see
    test_update_fires_in_proportion_to_how_extreme_the_anomaly_is).
    """
    det = MetricDetector()
    for value in range(WARMUP_TICKS + 1):
        assert det.update(float(value)) == (None, False)
    assert det.update(float(WARMUP_TICKS + 1))[0] is not None


def test_neither_path_lets_an_anomalous_tick_redefine_typical():
    """Ground rule #1 at the unit level: a fault tick must not redefine typical.

    score_only() never folds a tick in. update() folds in only ticks that scored
    healthy, and withholds anomalous ones — that withholding is the fix for a
    fault that runs for minutes training the detector to accept it.

    So for an ANOMALOUS tick the two paths now agree, which is the point. They
    still differ for a healthy one (below) and at the exclusion cap, which
    test_update_resumes_learning_after_the_exclusion_cap covers.
    """
    det = _warm(MetricDetector(), ticks=WARMUP_TICKS + 5)
    before = len(det.errors)
    assert before < ERROR_HISTORY, "reference deque is full; this test could not tell growth from a drop"

    det.score_only(1e6)
    assert len(det.errors) == before, "score_only() must never extend the reference"

    _, fired = det.update(1e6)
    assert fired, "a 1e6 excursion should score anomalous; if not, the fix regressed"
    assert len(det.errors) == before, "update() must withhold an anomalous tick from the reference"


def test_update_still_learns_from_a_healthy_tick():
    """The counterpart: withholding anomalies must not stop ordinary learning, or
    the reference freezes and the detector never adapts to anything.
    """
    det = _warm(MetricDetector(), ticks=WARMUP_TICKS + 5)
    before = len(det.errors)

    # A value close to the prediction scores low and must be folded in.
    _, fired = det.update(det.prediction * 1.001)
    assert not fired
    assert len(det.errors) == before + 1


def test_update_resumes_learning_after_the_exclusion_cap():
    """Withholding anomalous ticks forever means a genuine baseline shift — a
    deliberate scale-up, traffic that doubles for good — alarms indefinitely and
    is never learned. SELF_EXCLUDE_CAP bounds the run of exclusions.
    """
    det = _warm(MetricDetector(), ticks=WARMUP_TICKS + 5)
    before = len(det.errors)

    for _ in range(SELF_EXCLUDE_CAP):
        det.update(1e6)
    assert len(det.errors) == before, "should still be withholding inside the cap"

    det.update(1e6)
    assert len(det.errors) == before + 1, "past the cap, the reference must start accepting again"


def test_a_non_finite_reading_is_ignored_rather_than_poisoning_the_prediction():
    """Prometheus can hand back NaN for a scrape that missed. One NaN folded into
    an EWMA makes every later prediction NaN, and a NaN z-score never fires
    again -- a permanently blind detector with no error anywhere.
    """
    det = _warm(MetricDetector())
    prediction = det.prediction

    assert det.update(float("nan")) == (None, False)
    assert det.update(float("inf")) == (None, False)
    assert det.prediction == prediction


def test_constant_metrics_are_dropped_from_the_active_set(healthy_metrics):
    """`restarts` never moves in healthy data, so its error distribution has zero
    spread and STD_FLOOR would be doing all the work -- any change at all scores
    as infinitely anomalous. The variance and mode-fraction guards drop it.
    """
    det = Detector.fit_healthy(healthy_metrics)
    assert det.workloads, "expected one WorkloadDetector per sample workload"

    for workload, wd in det.workloads.items():
        assert "restarts" not in wd.metrics, f"{workload} kept a constant metric"
        assert wd.metrics, f"{workload} has no active metrics at all"
        mode_fraction = (healthy_metrics[healthy_metrics["workload"] == workload]["restarts"] == 0).mean()
        assert mode_fraction > MAX_MODE_FRACTION


def test_a_restart_is_suppressed_for_the_full_window_and_then_lifts(healthy_metrics):
    """signal's Part 4.4. A restart looks like a severe anomaly -- memory drops to
    near zero, CPU spikes on startup -- so without suppression the controller
    "fixes" its own restart by restarting again. That is the infinite loop.

    This is the fresh-clone runnable counterpart to ml/test_restart_suppression.py,
    which reads the gitignored data/healthy/ and therefore cannot run for anyone
    without a collection handoff. It asserts only the half that can be asserted:
    that suppression holds for POST_RESTART_SUPPRESS_TICKS and then releases.
    The other half of that script -- that the restart shape WOULD have fired
    without suppression -- does not hold through update() on any data, synthetic
    or real; see test_update_cannot_fire_on_an_arbitrarily_extreme_anomaly.
    """
    workload = sorted(healthy_metrics["workload"].unique())[0]
    det = Detector.fit_healthy(healthy_metrics)
    wd = det.workloads[workload]
    baseline = healthy_metrics[healthy_metrics["workload"] == workload].iloc[-1]

    restart_tick = {m: float(baseline[m]) for m in wd.metrics}
    for key, factor in (("mem_bytes", 0.05), ("mem_pct", 0.05), ("cpu_cores", 4.0)):
        if key in restart_tick:
            restart_tick[key] = float(baseline[key]) * factor

    wd.on_restart()
    assert wd.suppress_ticks_left == POST_RESTART_SUPPRESS_TICKS

    fired = [wd.update(restart_tick)[1] for _ in range(POST_RESTART_SUPPRESS_TICKS)]
    assert not any(fired), "fired inside the post-restart suppression window"
    assert wd.suppress_ticks_left == 0, "suppression must lift, not latch"


def test_on_restart_clears_the_per_metric_state_it_is_documented_to_clear():
    """Suppression alone is not enough: the pre-restart baseline describes a
    process that no longer exists, so the reference has to be rebuilt too.
    """
    wd = WorkloadDetector(["cpu_cores"])
    _warm(wd.detectors["cpu_cores"])
    wd.streak = 3

    wd.on_restart()

    assert wd.streak == 0
    assert wd.detectors["cpu_cores"].prediction is None
    assert len(wd.detectors["cpu_cores"].errors) == 0


def test_update_fires_in_proportion_to_how_extreme_the_anomaly_is():
    """Regression for the desensitization defect: update() used to be unable to
    fire on a step change of ANY magnitude.

    The cause was ordering. update() appended this tick's error to the reference
    distribution and then computed _z() against that same distribution, so every
    tick was scored against a reference containing itself. That made the score
    scale-invariant: a 10x excursion and a 10^12x excursion both produced
    z = 1.91, with a single-tick ceiling of about
    (1/RECENT_WINDOW)*sqrt(ERROR_HISTORY) ~ 2.0 and a sustained burst peaking
    near 3.9 -- all under Z_THRESHOLD = 4.0. Detector.score() is the only public
    scoring API and the one the pickled artifact hands to the controller, so the
    live gate never opened and nothing downstream of it ever ran. That was the
    whole of the observed "3 firings across 160 real fault runs".

    update() now scores BEFORE folding the tick in. This test pins the property
    that was missing: the score must grow with the anomaly.
    """
    scores = {}
    for magnitude in (1e3, 1e6, 1e12):
        z, fired = _warm(MetricDetector()).update(magnitude)
        scores[magnitude] = z
        assert fired, f"a {magnitude:g} excursion must fire"
        assert z > Z_THRESHOLD

    # The specific thing that was broken: bigger anomaly, bigger score. Under the
    # old ordering these were all equal to within 0.1%.
    assert scores[1e12] > scores[1e6] > scores[1e3]

    # And sustaining it must not decay below the threshold the way it used to.
    sustained = _warm(MetricDetector())
    assert sustained.update(1e12)[1], "first sustained tick must fire"


def test_score_only_does_fire_on_the_same_extreme_anomaly():
    """The other side of the test above: the arithmetic is fine, the ceiling comes
    entirely from training on the tick being scored. Same warmup, same input,
    score_only() instead of update() -- and it fires immediately.
    """
    det = _warm(MetricDetector())
    z, fired = det.score_only(100.0 * 10**12)
    assert fired, "score_only must fire on an excursion update() cannot see"
    assert z > Z_THRESHOLD
