"""Positive control for the two-arm harness: can it SEE an effect that is there?

-------------------------------------------------------------------------------
WHY THIS FILE EXISTS
-------------------------------------------------------------------------------

The 63-pair two-arm campaign of 2026-10-06 came back null -- median delta
+0.00s, 7 pairs favouring prodrome against 9 favouring control, sign test
p=0.80 -- and that was the correct answer, because `control/controller.py` ran
with `DRY_RUN=True` and `load_attested=False`, so the two arms genuinely were
the same system. A harness that reported a difference there would have been
broken in exactly the way that looks like a discovery.

But a null run only rules out a harness that INVENTS differences. It says
nothing about a harness that is BLIND to them. Until somebody shows this
pipeline recovering an effect of known size, every future "Prodrome did not
beat stock Kubernetes" is uninterpretable: indistinguishable from an instrument
that cannot see.

This module is the missing half. It takes the REAL health series from a
finished campaign, holds one arm's recovery back by a known number of seconds,
and reruns the unmodified `eval.recovery` pipeline on the result. The true
delta is N by construction. The question is what comes out.

-------------------------------------------------------------------------------
WHAT THIS TESTS, AND WHAT IT DOES NOT -- READ THIS BEFORE QUOTING A NUMBER
-------------------------------------------------------------------------------

This is a REPLAY-LEVEL injection. It exercises:

    health.csv -> serving()/latency_band() -> outage_outcome()/
    degradation_outcome() -> first_sustained() -> measure() ->
    paired_delta_table() -> the tie band and the directional counts

i.e. the ANALYSIS half, end to end, including every judgement call in it. It
does NOT exercise the COLLECTION half -- `collect/twoarm.py`'s sampler, the
`kubectl` round-trip, the injection barrier, the per-arm probe. A number from
this module is a statement about the analysis, and `collect/poscontrol.py` is
the live counterpart that covers collection as well.

The injection is made in the RAW HEALTH SERIES, not in the measured deltas.
That distinction is the whole value of the file: adding N to a column of
`measured.csv` would prove only that subtraction works. Flipping health bits
forces `first_sustained`, the 30s hold, the outcome classifier, the per-arm
latency band, the skew gate and the tie band all to re-derive the number from
samples -- which is where a blind instrument would actually be blind, because
the sampler's achieved cadence quantises every recovery time it produces.

-------------------------------------------------------------------------------
THIS IS NOT EVIDENCE THAT PRODROME WORKS
-------------------------------------------------------------------------------

Stated once in code so it cannot be lost in a summary: the effect injected here
is MANUFACTURED. It is a property of this module, not of `control/`. Nothing in
here may be reported as a Prodrome result, and the sweep output carries
`manufactured_effect_s` on every row so a reader who sees only the table still
sees that. `eval.recovery.publishable()` is unaffected and still refuses the
underlying campaign, which is correct.

-------------------------------------------------------------------------------
HOW THE DELAY IS INJECTED
-------------------------------------------------------------------------------

Per pair, in the target arm's own slice of the health series:

  OUTAGE faults (POD_KILL, MEMORY_LEAK)
    The samples in [t_recovered, t_recovered + N) are rewritten to
    not-serving (`ready_replicas=0`, `probe_ok=False`). The failure instant
    does not move -- it is still the first not-serving sample at or after
    injection -- so the pair's recovery time grows by N and nothing else
    changes. This is literally "the pod took N seconds longer to come back."

  DEGRADATION faults (CPU_HOG, DISK_STRESS)
    The samples in [t_recovered, t_recovered + N) get their
    `probe_latency_ms` pushed above that arm's own pre-injection band, so the
    degraded span grows by N. The band itself is computed from pre-injection
    samples, which this module never touches, so the band does not move under
    its own injection.

Both are clipped at the next pair's injection start for the same arm and
workload, so one pair's manufactured delay can never leak into the next pair's
measurement window. `clipped_pairs` in the sweep output counts any that hit
that bound; a non-zero count means the requested N did not fully land and the
row understates the harness.

-------------------------------------------------------------------------------
WHAT "RELIABLY RECOVERS" MEANS HERE
-------------------------------------------------------------------------------

`DetectionCriterion` is deliberately three conditions, not one, because each
alone has a cheap failure mode:

  1. DIRECTION.  At least `min_power` of the pairs in the delta must land on
     the correct side of the tie band. A median that moves while half the
     pairs disagree is not a detection, it is a shifted distribution with no
     per-pair signal.
  2. MAGNITUDE.  The recovered median delta must be within `mag_tol` of -N.
     Direction alone would pass a harness that notices SOMETHING happened and
     reports the wrong size, which for a project that has to decide whether an
     improvement is worth shipping is not a detection either.
  3. NO CENSORING BLOWUP.  Pairs lost to NOT_RECOVERED/STILL_DEGRADED must
     stay under `max_excluded`. At large N the delay runs past the end of the
     trailing observation window and pairs start dropping out -- the harness
     "loses" the effect by discarding exactly the pairs that have it, which is
     a real and reportable failure mode, not a detection.

The smallest N meeting all three, for a fault type, is that fault type's
MINIMUM DETECTABLE EFFECT. It is the number any future claim about that fault
type has to clear before the instrument could have seen it.

Usage:
    python -m eval.sensitivity --indir data/twoarm
    python -m eval.sensitivity --indir data/twoarm --sizes 0,5,10,20,40,80
    python -m eval.sensitivity --indir data/twoarm --arm prodrome   # sign check
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eval import recovery
from eval.recovery import (CONTROL_ARM, DEGRADATION, OUTAGE, RECOVERY_KIND,
                           TREATMENT_ARM)

# Default sweep. Deliberately dense below the measured noise floor (p90 |delta|
# of 25.9s / 78.9s / 21.9s / 6.8s per fault type) and sparse above it: the
# interesting question is where the curve turns on, and above the floor every
# harness that is not broken detects everything.
DEFAULT_SIZES = (0.0, 2.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0,
                 50.0, 60.0, 75.0, 90.0, 120.0)

# How far above the band a manufactured degraded sample is pushed. Any value
# strictly above the band would do; a multiple rather than a constant keeps it
# above the band for a workload whose baseline latency is large (postgres sits
# an order of magnitude above redis), and the result is read only through a
# `> band` comparison, so the magnitude is never itself a measurement.
DEGRADED_BAND_MULTIPLE = 4.0


@dataclass(frozen=True)
class DetectionCriterion:
    """What counts as the harness having recovered the injected effect.

    Defaults are set, not guessed: `min_power=0.9` because a 10%
    wrong-side rate on a manufactured effect is already generous for an
    instrument that is supposed to be measuring a real one; `mag_tol=None`
    resolves at call time to one achieved sampling interval, because a
    recovery time quantised at 6.8s cannot be expected to report a magnitude
    to better than that and demanding more would fail the harness for the
    sampler's limitation rather than for blindness.
    """
    min_power: float = 0.9
    mag_tol: float | None = None          # None -> one achieved sample interval
    max_excluded: float = 0.1             # fraction of pairs lost to censoring


# --- the injection ----------------------------------------------------------

def _next_start(pairs: pd.DataFrame, arm: str, workload: str,
                after: pd.Timestamp) -> pd.Timestamp | None:
    """Start of the next pair on the same arm and workload, or None.

    The clip bound. Without it a large manufactured delay on one pair would
    still be sitting in the health series when the next pair on that workload
    opens its (upper-unbounded) measurement window, and the next pair's
    failure instant would be this pair's manufactured one.
    """
    later = pairs[(pairs["arm"] == arm) & (pairs["workload"] == workload)
                  & (pairs["start_ts"] > after)]["start_ts"]
    return later.min() if len(later) else None


def inject_delay(pairs: pd.DataFrame, health: pd.DataFrame, seconds: float,
                 arm: str = CONTROL_ARM,
                 hold_seconds: float = recovery.HOLD_SECONDS,
                 window_seconds: float | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold `arm` back by `seconds` on every pair, in the raw health series.

    Returns the modified health frame and a per-pair bookkeeping frame: where
    the delay landed, how much of it landed, and why any of it did not. The
    bookkeeping is per pair rather than a running total because the two reasons
    a delay fails to land -- no recovery instant to delay, and the next pair on
    the same workload starting first -- are both concentrated in particular
    fault types, and a single campaign-wide counter would hide that.

    Not in-place. The caller sweeps many sizes against one loaded campaign and
    every size must start from the same untouched series.
    """
    out = health.copy()
    # Normalise once. The files on disk carry `probe_ok` as the string "True",
    # and a later `.loc[...] = False` would otherwise leave a column of mixed
    # bool and str that `serving()` only survives because it stringifies.
    out["probe_ok"] = out["probe_ok"].astype(str).str.lower().isin(
        ["true", "1", "yes"])
    out["ready_replicas"] = pd.to_numeric(out["ready_replicas"], errors="coerce")
    out["probe_latency_ms"] = pd.to_numeric(out["probe_latency_ms"], errors="coerce")

    book = []
    for _, p in pairs[pairs["arm"] == arm].iterrows():
        kind = RECOVERY_KIND.get(p["fault_type"])
        if kind is None:
            continue
        rec = dict(pair_id=p["pair_id"], fault_type=p["fault_type"],
                   workload=p["workload"], requested_s=float(seconds),
                   landed_s=0.0, rows_rewritten=0, applied=False,
                   clipped=False, no_recovery=False)
        book.append(rec)
        if seconds <= 0:
            continue

        t0, wl = p["start_ts"], p["workload"]
        obs_hi = recovery.pair_window_hi(p, window_seconds)
        base = recovery._slice(out, arm, wl, hi=t0)
        win = recovery._slice(out, arm, wl, lo=t0, hi=obs_hi)
        if kind == OUTAGE:
            res = recovery.outage_outcome(win, t0, hold_seconds)
        else:
            res = recovery.degradation_outcome(win, t0, base, hold_seconds)
        t_rec = res.get("t_recovered")
        if t_rec is None:
            rec["no_recovery"] = True
            continue

        hi = t_rec + pd.Timedelta(seconds=seconds)
        # Two bounds, both hard. The next pair on this arm and workload, so a
        # manufactured delay can never become the next pair's fault; and the
        # pair's own observation window when one is set, so the injection
        # cannot write outside the span the measurement is allowed to read.
        for bound in (_next_start(pairs, arm, wl, t0), obs_hi):
            if bound is not None and bound < hi:
                hi, rec["clipped"] = bound, True

        sel = ((out["arm"] == arm) & (out["workload"] == wl)
               & (out["ts"] >= t_rec) & (out["ts"] < hi))
        n = int(sel.sum())
        if not n:
            continue
        if kind == OUTAGE:
            out.loc[sel, "ready_replicas"] = 0
            out.loc[sel, "probe_ok"] = False
        else:
            band = recovery.latency_band(base)
            if band is None:                 # no band -> the pair is NO_DATA anyway
                continue
            out.loc[sel, "probe_latency_ms"] = band * DEGRADED_BAND_MULTIPLE
        rec.update(applied=True, rows_rewritten=n,
                   landed_s=max(0.0, (hi - t_rec).total_seconds()))
    return out, pd.DataFrame(book)


# --- one point on the curve -------------------------------------------------

def _sign_test_p(n_correct: int, n_directional: int) -> float | None:
    """Two-sided exact binomial sign test at p=0.5, no scipy.

    Computed over DIRECTIONAL pairs only -- pairs inside the tie band carry no
    sign and padding them in either direction would be inventing data. With
    every directional pair on one side this is just 2 * 0.5**n, which is the
    number worth reporting.
    """
    n = int(n_directional)
    if n == 0:
        return None
    from math import comb
    k = int(n_correct)
    lo, hi = min(k, n - k), max(k, n - k)
    tail = sum(comb(n, i) for i in range(0, lo + 1))
    tail += sum(comb(n, i) for i in range(hi, n + 1))
    return min(1.0, tail / float(2 ** n))


def sweep_point(pairs: pd.DataFrame, health: pd.DataFrame, seconds: float,
                arm: str = CONTROL_ARM, tie_band: float | None = None,
                hold_seconds: float = recovery.HOLD_SECONDS,
                window_seconds: float | str | None = None) -> pd.DataFrame:
    """Inject `seconds` into `arm`, remeasure, and return one row per fault type.

    `recovered_median_delta_s` is signed the way `paired_delta_table` signs it:
    negative means the treatment arm was faster. Delaying the CONTROL arm by N
    should therefore produce -N, and delaying the TREATMENT arm by N should
    produce +N -- which is why `--arm prodrome` is worth running once: a
    harness that reports a magnitude but gets the sign from the arm it polled
    first rather than from the data would pass the control-arm sweep.
    """
    hurt, book = inject_delay(pairs, health, seconds, arm, hold_seconds,
                              window_seconds)
    tie = recovery.sampling_resolution(hurt) if tie_band is None else tie_band
    measured = recovery.measure(pairs, hurt, hold_seconds=hold_seconds,
                                window_seconds=window_seconds)
    deltas = recovery.paired_delta_table(measured, tie_band=tie)

    # Which direction a correct detection points, given which arm we hurt.
    favoured = "prodrome_faster_pairs" if arm == CONTROL_ARM else "control_faster_pairs"
    wrong = "control_faster_pairs" if arm == CONTROL_ARM else "prodrome_faster_pairs"
    expected_sign = -1.0 if arm == CONTROL_ARM else 1.0

    by_fault = (book.groupby("fault_type") if len(book) else None)
    rows = []
    for _, d in deltas.iterrows():
        n_in = int(d["pairs_in_delta"])
        n_right, n_wrong = int(d[favoured]), int(d[wrong])
        n_dir = n_right + n_wrong
        med = d["median_delta_s"]
        b = (by_fault.get_group(d["fault_type"])
             if by_fault is not None and d["fault_type"] in by_fault.groups
             else pd.DataFrame(columns=book.columns))
        # What the injection ACTUALLY delivered, not what it was asked for. A
        # delay clipped by the next pair is a smaller effect, and comparing the
        # recovered median against the requested size would then blame the
        # harness for a shortfall the injector caused.
        landed = float(b.loc[b["applied"], "landed_s"].median()) if b["applied"].any() else 0.0
        rows.append(dict(
            fault_type=d["fault_type"], kind=d["kind"],
            manufactured_effect_s=float(seconds), hurt_arm=arm,
            delivered_median_s=round(landed, 1),
            expected_median_delta_s=round(expected_sign * float(seconds), 1),
            pairs=int(d["pairs"]), pairs_in_delta=n_in,
            excluded=int(d["excluded"]), excluded_censored=int(d["excluded_censored"]),
            tie_band_s=round(float(tie), 2),
            recovered_median_delta_s=med,
            magnitude_error_s=(None if med is None
                               else round(abs(float(med) - expected_sign * seconds), 1)),
            correct_direction_pairs=n_right, wrong_direction_pairs=n_wrong,
            tied_pairs=int(d["tied_pairs"]),
            power=round(n_right / n_in, 3) if n_in else None,
            sign_test_p=_sign_test_p(n_right, n_dir),
            injected_pairs=int(b["applied"].sum()) if len(b) else 0,
            clipped_pairs=int(b["clipped"].sum()) if len(b) else 0,
            no_recovery_pairs=int(b["no_recovery"].sum()) if len(b) else 0,
        ))
    return pd.DataFrame(rows)


def sweep(pairs: pd.DataFrame, health: pd.DataFrame,
          sizes=DEFAULT_SIZES, arm: str = CONTROL_ARM,
          tie_band: float | None = None,
          hold_seconds: float = recovery.HOLD_SECONDS,
          window_seconds: float | str | None = None) -> pd.DataFrame:
    """The curve: every size against the same untouched campaign.

    The tie band is pinned to the UNMODIFIED series' achieved cadence rather
    than recomputed per size. Recomputing would let the injection move the
    yardstick it is being measured against -- rewriting health bits does not
    change sample timestamps, so it should not change the band, and pinning it
    makes that an invariant instead of a hope.
    """
    tie = recovery.sampling_resolution(health) if tie_band is None else tie_band
    return pd.concat([sweep_point(pairs, health, s, arm, tie, hold_seconds,
                                  window_seconds)
                      for s in sizes], ignore_index=True)


# --- the resolution that actually applies -----------------------------------

def cadence_by_state(health: pd.DataFrame) -> pd.DataFrame:
    """Sampling interval split by whether the workload was serving.

    `recovery.sampling_resolution()` takes the median interval over ALL
    samples, and the overwhelming majority of samples are healthy ones. That
    is the wrong population. Every recovery time is a difference between two
    instants that both sit INSIDE an outage, so the resolution that bounds the
    measurement is the cadence achieved while NOT serving -- and the live
    positive control showed those are not the same number:

        data/twoarm-poscontrol   serving 5.60s   NOT serving 14.77s
        data/twoarm              serving 6.80s   NOT serving  7.58s

    The cause is in `collect/twoarm.kubectl_probe`: it runs
    `kubectl exec deploy/<w> -- true` unconditionally, and when the deployment
    has no pod there is nothing to exec into, so the call blocks for its full
    15s timeout. The sampler therefore slows down by 2.6x exactly while the
    event it exists to measure is happening. `data/twoarm` shows it mildly
    because MEMORY_LEAK leaves a crashlooping pod present to exec into;
    `data/twoarm-poscontrol` scales to zero, so no pod exists for the whole
    hold and every probe pays the timeout.

    Consequence for the tie band: `TIE_BAND_SECONDS` is calibrated from the
    all-sample median, so it UNDERSTATES the quantisation that applies at the
    two instants the measurement is made from. The live null pair proves the
    cost -- see `eval/README.md`.
    """
    h = health.sort_values(["arm", "workload", "ts"]).copy()
    h["gap"] = h.groupby(["arm", "workload"])["ts"].diff().dt.total_seconds()
    h["serving"] = recovery.serving(h).values
    out = (h.groupby("serving")["gap"]
           .agg(samples="count", median_gap_s="median", max_gap_s="max")
           .reset_index())
    out["median_gap_s"] = out["median_gap_s"].round(2)
    out["max_gap_s"] = out["max_gap_s"].round(2)
    return out


def outage_resolution(health: pd.DataFrame) -> float:
    """The honest tie band: the cadence achieved while NOT serving.

    Falls back to `recovery.sampling_resolution()` when a run contains no
    not-serving samples at all, which is the degradation-fault case -- there
    the probe keeps answering, so the all-sample median IS the right number.
    """
    c = cadence_by_state(health)
    bad = c[~c["serving"].astype(bool)]
    if not len(bad) or pd.isna(bad["median_gap_s"].iloc[0]):
        return recovery.sampling_resolution(health)
    return max(float(bad["median_gap_s"].iloc[0]),
               recovery.sampling_resolution(health))


# --- the live counterpart ---------------------------------------------------

PATTERN_RE = re.compile(r"effect(?P<size>-?\d+)s_hurt_(?P<arm>\w+)")


def live_curve(pairs: pd.DataFrame, health: pd.DataFrame,
               tie_band: float | None = None,
               hold_seconds: float = recovery.HOLD_SECONDS,
               window_seconds: float | str = "trail") -> pd.DataFrame:
    """The same curve, from `collect/poscontrol.py`'s LIVE pairs.

    Nothing is injected here -- the effect is already in the cluster data. The
    size and the hurt arm are read back out of the `pattern` column, which
    `poscontrol.poscontrol_rows` writes precisely so that this is possible from
    the file alone rather than from the command line that produced it.

    One row per effect size, and each size is measured on ONLY its own pairs:
    pooling them would average a 0s group into a 30s one and produce a curve
    with no point on it.

    `window_seconds` defaults to `"trail"` here, not to the campaign's open
    window. A positive-control run has several pairs on one workload minutes
    apart, which is exactly the geometry in which an open window lets one
    pair's outage become the next pair's -- and these pairs exist to measure
    the instrument, so they get the instrument's honest setting.
    """
    tie = recovery.sampling_resolution(health) if tie_band is None else tie_band
    rows = []
    for pattern, g in pairs.groupby("pattern", dropna=False):
        m = PATTERN_RE.search(str(pattern))
        if m is None:
            continue
        size, arm = float(m.group("size")), m.group("arm")
        measured = recovery.measure(g, health, hold_seconds=hold_seconds,
                                    window_seconds=window_seconds)
        d = recovery.paired_delta_table(measured, tie_band=tie)
        if not len(d):
            continue
        d = d.iloc[0]
        favoured = ("prodrome_faster_pairs" if arm == CONTROL_ARM
                    else "control_faster_pairs")
        wrong = ("control_faster_pairs" if arm == CONTROL_ARM
                 else "prodrome_faster_pairs")
        sign = -1.0 if arm == CONTROL_ARM else 1.0
        n_in = int(d["pairs_in_delta"])
        n_right, n_wrong = int(d[favoured]), int(d[wrong])
        med = d["median_delta_s"]
        rows.append(dict(
            fault_type=d["fault_type"], kind=d["kind"],
            manufactured_effect_s=size, hurt_arm=arm,
            delivered_median_s=size,          # the cluster delivered it, not us
            expected_median_delta_s=round(sign * size, 1),
            pairs=int(d["pairs"]), pairs_in_delta=n_in,
            excluded=int(d["excluded"]), excluded_censored=int(d["excluded_censored"]),
            tie_band_s=round(float(tie), 2), recovered_median_delta_s=med,
            magnitude_error_s=(None if med is None
                               else round(abs(float(med) - sign * size), 1)),
            correct_direction_pairs=n_right, wrong_direction_pairs=n_wrong,
            tied_pairs=int(d["tied_pairs"]),
            power=round(n_right / n_in, 3) if n_in else None,
            sign_test_p=_sign_test_p(n_right, n_right + n_wrong),
            injected_pairs=int(d["pairs"]), clipped_pairs=0, no_recovery_pairs=0,
        ))
    return (pd.DataFrame(rows).sort_values("manufactured_effect_s")
            .reset_index(drop=True))


# --- the headline number ----------------------------------------------------

def minimum_detectable_effect(curve: pd.DataFrame,
                              criterion: DetectionCriterion | None = None,
                              tie_band: float | None = None) -> pd.DataFrame:
    """Smallest injected effect each fault type reliably recovers.

    "Smallest meeting the criterion" is taken as the smallest size from which
    EVERY larger size in the sweep also passes. A single size that happens to
    pass while the next one up fails is noise in the curve, not a threshold,
    and reporting it would hand the project a number that does not hold.
    """
    criterion = criterion or DetectionCriterion()
    rows = []
    for fault, g in curve.groupby("fault_type", dropna=False):
        g = g.sort_values("manufactured_effect_s").reset_index(drop=True)
        tol = criterion.mag_tol
        if tol is None:
            tol = float(g["tie_band_s"].max() if tie_band is None else tie_band)
        n_pairs = int(g["pairs"].max())

        def direction_ok(r) -> bool:
            """Did the harness point the right way, on enough pairs, without
            throwing away the pairs that carry the effect?"""
            if r["manufactured_effect_s"] <= 0:
                return False                   # a null point can never "detect"
            if r["power"] is None or r["power"] < criterion.min_power:
                return False
            lost = r["excluded_censored"] / r["pairs"] if r["pairs"] else 1.0
            return lost <= criterion.max_excluded

        def magnitude_ok(r) -> bool:
            return (r["magnitude_error_s"] is not None
                    and not pd.isna(r["magnitude_error_s"])
                    and float(r["magnitude_error_s"]) <= tol)

        def first_stable(flags: list[bool]) -> float | None:
            """Smallest size from which every LARGER size also passes.

            A size that passes while the next one up fails is a wobble in the
            curve, not a threshold, and quoting it would hand the project a
            number that does not hold.
            """
            for i in range(len(flags)):
                if flags[i] and all(flags[i:]):
                    return float(g.loc[i, "manufactured_effect_s"])
            return None

        dir_ok = [direction_ok(r) for _, r in g.iterrows()]
        both_ok = [d and magnitude_ok(r) for d, (_, r) in zip(dir_ok, g.iterrows())]

        # Signed bias on the sizes where detection worked at all: positive means
        # the harness reports the gap as BIGGER than it is. It is not noise --
        # a recovery is only observed at the first sample after it happens, and
        # the 30s sustained-hold then pushes it further.
        det = g[[d and r["recovered_median_delta_s"] is not None
                 for d, (_, r) in zip(dir_ok, g.iterrows())]]
        bias = None
        if len(det):
            over = (det["recovered_median_delta_s"].abs()
                    - det["manufactured_effect_s"])
            bias = round(float(over.median()), 1)

        rows.append(dict(
            fault_type=fault, kind=g.loc[0, "kind"], pairs=n_pairs,
            tie_band_s=round(float(g["tie_band_s"].max()), 2),
            magnitude_tolerance_s=round(tol, 2),
            min_power=criterion.min_power,
            # The headline: smallest effect the harness reliably points at.
            min_detectable_effect_s=first_stable(dir_ok),
            # The stricter one: smallest effect it also SIZES to within a
            # sampling interval. Always >= the above, and the gap between them
            # is the band where the harness can tell you a difference exists
            # but not how big it is.
            min_accurate_effect_s=first_stable(both_ok),
            median_overstatement_s=bias,
            detected_any_size=bool(any(dir_ok)),
            largest_size_tried_s=float(g["manufactured_effect_s"].max()),
        ))
    return pd.DataFrame(rows).sort_values("fault_type").reset_index(drop=True)


# --- reporting --------------------------------------------------------------

BANNER = """
POSITIVE CONTROL -- THE EFFECT BELOW WAS PUT THERE ON PURPOSE, by
{source}. It is a property of the positive control, not of control/.
Nothing here is evidence that Prodrome works; it is evidence about whether
the instrument can see.
"""


def render(curve: pd.DataFrame, mde: pd.DataFrame, arm: str,
           window_seconds: float | str | None = None,
           live: bool = False) -> str:
    cols = ["fault_type", "manufactured_effect_s", "delivered_median_s",
            "expected_median_delta_s", "recovered_median_delta_s",
            "magnitude_error_s", "pairs_in_delta", "correct_direction_pairs",
            "wrong_direction_pairs", "tied_pairs", "power", "sign_test_p",
            "excluded_censored", "injected_pairs", "clipped_pairs"]
    source = ("collect/poscontrol.py, live in the cluster" if live
              else "eval/sensitivity.py, rewritten into a finished run's health.csv")
    half = ("LIVE: COLLECTION + ANALYSIS, every component a real campaign uses"
            if live else
            "REPLAY: ANALYSIS half only -- collection is NOT exercised;\n"
            " collect/poscontrol.py is the live counterpart")
    out = [BANNER.format(source=source), f"arm held back: {arm}   ({half})",
           f"measurement window: {window_seconds if window_seconds is not None else 'OPEN (campaign default -- see recovery.pair_window_hi)'}\n",
           "SENSITIVITY CURVE", "-" * 70,
           curve[cols].to_string(index=False), "",
           "MINIMUM DETECTABLE EFFECT", "-" * 70,
           mde.to_string(index=False), ""]
    blind = mde[mde["min_detectable_effect_s"].isna()]
    if len(blind):
        out.append("BLIND at every size tried: "
                   + ", ".join(blind["fault_type"]) + "\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="data/twoarm")
    ap.add_argument("--outdir", default=None,
                    help="where to write sensitivity-curve.csv / mde.csv "
                         "(default: --indir)")
    ap.add_argument("--sizes", default=None,
                    help="comma-separated seconds, e.g. 0,5,10,20,40")
    ap.add_argument("--arm", choices=list(recovery.ARMS), default=CONTROL_ARM,
                    help="which arm to hold back. control -> a manufactured "
                         "Prodrome win; prodrome -> the sign check.")
    ap.add_argument("--min-power", type=float, default=0.9)
    ap.add_argument("--window", default=None,
                    help="close each pair's measurement window: 'trail' "
                         "(end_ts + trail_s, the span actually watched) or a "
                         "number of seconds after start_ts. Default: open, "
                         "which is what the campaign was measured with.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filenames, so a second sweep "
                         "does not overwrite the first")
    ap.add_argument("--tie", default="auto",
                    help="tie band: 'auto' = recovery.sampling_resolution() "
                         "(the all-sample median, which is what the campaign "
                         "used), 'outage' = the cadence achieved while NOT "
                         "serving (the honest one -- see cadence_by_state), or "
                         "a number of seconds.")
    ap.add_argument("--live", action="store_true",
                    help="--indir holds collect/poscontrol.py output: read the "
                         "effect back out of it instead of injecting one. This "
                         "is the COLLECTION + ANALYSIS result.")
    args = ap.parse_args()

    window = args.window
    if window is not None and window != "trail":
        window = float(window)

    pairs, health = recovery.load(args.indir)
    if args.tie == "auto":
        tie = recovery.sampling_resolution(health)
    elif args.tie == "outage":
        tie = outage_resolution(health)
    else:
        tie = float(args.tie)
    # The not-serving row is the resolution that actually bounds a recovery
    # time; the all-sample median the tie band is usually taken from is
    # dominated by healthy samples. See `cadence_by_state`.
    print("cadence by state:")
    print(cadence_by_state(health).to_string(index=False))
    print(f"tie band in use: {tie:.2f}s  (--tie {args.tie})")
    print()

    if args.live:
        # No injection: read the effect back out of pairs written by
        # collect/poscontrol.py, which already put it into the cluster.
        curve = live_curve(pairs, health, tie_band=tie,
                           window_seconds="trail" if window is None else window)
        mde = minimum_detectable_effect(
            curve, DetectionCriterion(min_power=args.min_power))
        arm = curve["hurt_arm"].iloc[0] if len(curve) else "?"
        print(render(curve, mde, arm, window or "trail", live=True))
        outdir = Path(args.outdir or args.indir)
        outdir.mkdir(parents=True, exist_ok=True)
        tag = f"-{args.tag}" if args.tag else "-live"
        curve.to_csv(outdir / f"sensitivity-curve{tag}.csv", index=False)
        mde.to_csv(outdir / f"sensitivity-mde{tag}.csv", index=False)
        print(f"wrote {outdir / f'sensitivity-curve{tag}.csv'} and "
              f"{outdir / f'sensitivity-mde{tag}.csv'}")
        return

    sizes = (DEFAULT_SIZES if args.sizes is None
             else tuple(float(s) for s in args.sizes.split(",")))
    curve = sweep(pairs, health, sizes, arm=args.arm, tie_band=tie,
                  window_seconds=window)
    mde = minimum_detectable_effect(
        curve, DetectionCriterion(min_power=args.min_power))
    print(render(curve, mde, args.arm, window))

    outdir = Path(args.outdir or args.indir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    curve.to_csv(outdir / f"sensitivity-curve{tag}.csv", index=False)
    mde.to_csv(outdir / f"sensitivity-mde{tag}.csv", index=False)
    print(f"wrote {outdir / f'sensitivity-curve{tag}.csv'} and "
          f"{outdir / f'sensitivity-mde{tag}.csv'}")


if __name__ == "__main__":
    main()
