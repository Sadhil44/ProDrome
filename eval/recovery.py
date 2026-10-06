"""Control-arm recovery measurement -- SETUP.md ground rule 4, the half
`eval/harness.py` explicitly does not cover.

`harness.py` answers "did the detector see it coming," single-arm. This module
answers the other question, the one that makes any remediation claim mean
anything: **we recovered the service, and stock Kubernetes didn't.** It turns
two arms' health series into a recovery-time table, per arm, per fault type,
and it refuses to call that table a comparison when the experiment it came
from could not have produced one.

Nothing here touches a cluster. The cluster-touching half is
`collect/twoarm.py`, which injects the paired faults and writes the two files
this module reads. The split is deliberate: every definition below is a
judgement call that has to be arguable offline, and it is all unit-tested
against fixtures in `eval/test_recovery.py`.

-------------------------------------------------------------------------------
RECOVERY TIME IS NOT ONE NUMBER. The landmine, stated once:
-------------------------------------------------------------------------------

`data/chaos/runs.csv`'s `failure_ts` is non-null only for MEMORY_LEAK (30/30)
and POD_KILL (10/10). It is null for all 90 DISK_STRESS and all 30 CPU_HOG
runs, because those two faults never actually fail -- `end_ts` is just when
`stress-ng` stopped. A recovery time measured from `end_ts` for CPU_HOG or
DISK_STRESS measures when the injector got bored, not when the service got
better, and averaging it with POD_KILL's genuine outage produces a number with
no referent at all.

So this module reports two quantities with two different names, and there is
no code path that pools them:

  OUTAGE faults       POD_KILL, MEMORY_LEAK
    The service stops serving. `recovery_seconds` = first not-serving tick ->
    first tick of a sustained-serving run. A real wall-clock outage.

  DEGRADATION faults  CPU_HOG, DISK_STRESS
    The service keeps serving, worse. There is no outage to time, so
    `recovery_seconds` is None by construction and `degradation_seconds`
    measures the span where the readiness-probe latency sat outside that
    arm's own pre-injection band.

`RECOVERY_KIND` is the whole of that decision and it is keyed by fault type,
not inferred from whether `failure_ts` happens to be populated -- inferring it
would silently reclassify a fault the day the injector's luck changed.

-------------------------------------------------------------------------------
WHY PROBE LATENCY, AND WHAT IT IS NOT
-------------------------------------------------------------------------------

The frozen metrics contract (SETUP.md 7) has no serving-health column:
`restarts` is a 1-minute `changes()` transient, not readiness, and a pod can be
Ready with terrible latency or not-Ready with none of the eight metrics moving.
Rather than change a contract four seats consume, `collect/twoarm.py` samples
readiness and the wall time of the readiness probe itself, on both arms, from
the same loop at the same cadence. That symmetry is the point: a difference
between arms cannot come from having measured them differently.

Probe latency is a WEAK SLI and the report says so. It is a `redis-cli ping` /
`pg_isready` / `GET /`, not user traffic, so a degradation number from it is a
lower bound on user impact and a null result is not evidence of no impact. The
honest fix is a real latency SLI from the k6 load generator, which would be a
new column, not a new definition -- see `degradation_outcome`.

-------------------------------------------------------------------------------
TWO OF FOUR FAULT TYPES CANNOT SHOW A DIFFERENCE, BY DESIGN
-------------------------------------------------------------------------------

Read `control/policy.py` before reading any result:

    CPU_HOG      -> scale_out        Prodrome has a lever
    MEMORY_LEAK  -> rolling_restart  Prodrome has a lever
    DISK_STRESS  -> alert_only       Prodrome deliberately does nothing
    POD_KILL     -> not in the table -> UNKNOWN -> "nothing"

For DISK_STRESS and POD_KILL the treatment arm is specified to behave exactly
like the control arm, so a tie is the designed outcome and is not a negative
result. `lever_for()` puts that in the table beside every row, because a reader
who averages four rows gets "Prodrome ties stock Kubernetes" out of a design
where two rows could never have moved.

Usage:
    python -m eval.recovery                       # reads data/twoarm/
    python -m eval.recovery --indir data/twoarm
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Read-only: the policy table is diagnosis's file. Imported so the "could this
# fault type ever have shown a difference" column is derived from the real
# lookup table rather than from a copy of it that can drift.
from control.policy import POLICY

TREATMENT_ARM = "prodrome"
CONTROL_ARM = "control"
ARMS = (TREATMENT_ARM, CONTROL_ARM)

# Per fault type, which quantity is even defined. See the module docstring.
# Keyed by fault type on purpose, NOT inferred from failure_ts being populated.
OUTAGE = "outage"
DEGRADATION = "degradation"
RECOVERY_KIND = {
    "POD_KILL": OUTAGE,
    "MEMORY_LEAK": OUTAGE,
    "CPU_HOG": DEGRADATION,
    "DISK_STRESS": DEGRADATION,
}

# Outcomes. Deliberately not booleans and deliberately never None-as-zero: a
# fault that never caused an outage is the case Prodrome is SUPPOSED to produce,
# and scoring it as "recovered in 0s" would hand the treatment arm a free win
# while scoring it as a miss would hand one to the control arm.
RECOVERED = "RECOVERED"              # went down, came back -- has a number
NO_OUTAGE = "NO_OUTAGE"              # never stopped serving (prevented, or too weak)
NOT_RECOVERED = "NOT_RECOVERED"      # went down, still down when we stopped watching
DEGRADED = "DEGRADED"                # latency left its band and came back
NO_DEGRADATION = "NO_DEGRADATION"    # latency never left its band
STILL_DEGRADED = "STILL_DEGRADED"    # left its band, had not returned
NO_DATA = "NO_DATA"                  # no health samples for this arm/pair at all

# A serving blip of one sample is a probe that lost a race, not an outage, and
# a recovery that lasts one sample is a pod about to CrashLoop again. Both
# endpoints therefore require the state to HOLD. The recovery instant is the
# FIRST sample of the sustained run, not the last -- so the hold requirement
# cannot inflate a recovery time, only disqualify a flap.
HOLD_SECONDS = 30.0

# Max tolerated difference between the two arms' actual injection start times.
# Both arms must take their fault under the same background load; a pair whose
# arms started a minute apart saw two different cluster states and is evidence
# of nothing. Flagged per pair, excluded from the paired delta, never silently
# averaged in.
SKEW_TOLERANCE_SECONDS = 5.0

# Degradation band: median + MAD_K * MAD of that arm's own pre-injection probe
# latency, with a floor so a workload whose baseline latency is near-constant
# (MAD ~ 0) does not get a zero-width band that every sample "exceeds".
MAD_K = 6.0
LATENCY_BAND_FLOOR_MS = 25.0
MIN_BASELINE_SAMPLES = 6

# Health sampling cadence the runner aims for. Lives here rather than in
# collect/twoarm.py because it is not just a collection knob: it sets the
# RESOLUTION of every recovery time, so the measurement side has to know it.
HEALTH_SAMPLE_SECONDS = 5.0

# Differences smaller than the sampling resolution are a tie, not a win.
#
# This is not conservatism, it is a bug that already happened. The first live
# two-arm POD_KILL pair came back prodrome 15.2097s vs control 15.2036s and the
# table duly reported "control faster: 1 pair" -- on a 6 millisecond difference,
# measured by a sampler whose achieved cadence was 6.4 seconds. The sign of a
# delta three orders of magnitude below the sample interval is noise, and
# counting it as a directional win is how a null experiment grows a result.
#
# `sampling_resolution()` recovers the ACHIEVED cadence from the health series
# so the band self-calibrates rather than trusting the interval we asked for.
TIE_BAND_SECONDS = HEALTH_SAMPLE_SECONDS

PAIRS_COLUMNS = ["pair_id", "arm", "namespace", "fault_type", "workload", "pattern",
                 "duration_s", "start_ts", "end_ts", "failure_ts", "restarts_delta",
                 "inject_ok", "skew_s", "load_attested"]
HEALTH_COLUMNS = ["ts", "arm", "workload", "ready_replicas", "desired_replicas",
                  "restarts_total", "probe_ok", "probe_latency_ms"]

MEASURED_COLUMNS = ["pair_id", "arm", "fault_type", "workload", "pattern", "kind",
                    "outcome", "t_fail", "t_recovered", "recovery_seconds",
                    "degradation_seconds", "samples", "skew_ok"]


# --- loading ----------------------------------------------------------------

def load(indir: str | Path = "data/twoarm") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two files `collect/twoarm.py` writes.

    Separate from every measurement function so the measurement is testable
    from constructed frames -- the testing seat's standing request that
    entrypoints take explicit data rather than whatever is on disk.
    """
    indir = Path(indir)
    pairs = pd.read_csv(indir / "pairs.csv",
                        parse_dates=["start_ts", "end_ts", "failure_ts"])
    health = pd.read_csv(indir / "health.csv", parse_dates=["ts"])
    return pairs, health


# --- the core primitive -----------------------------------------------------

def first_sustained(ts: pd.Series, ok: pd.Series, hold_seconds: float = HOLD_SECONDS):
    """First timestamp in `ts` that begins an unbroken `ok` run covering at
    least `hold_seconds`.

    Returns the START of the run. This is the one piece of arithmetic every
    number in this module rests on, which is why it is a named function with
    its own tests rather than three inline loops.

    A run at the very end of the series that is unbroken but shorter than
    `hold_seconds` does NOT count: we did not watch long enough to know it
    held, and treating "we stopped looking" as "it recovered" is exactly the
    bias that makes a control-arm comparison flattering. The caller reports
    that case as NOT_RECOVERED / STILL_DEGRADED, i.e. censored, rather than
    imputing a time.
    """
    ts = pd.Series(pd.to_datetime(ts, utc=True)).reset_index(drop=True)
    ok = pd.Series(ok).astype(bool).reset_index(drop=True)
    if len(ts) == 0:
        return None

    run_start = None
    for i in range(len(ts)):
        if not ok.iloc[i]:
            run_start = None
            continue
        if run_start is None:
            run_start = ts.iloc[i]
        if (ts.iloc[i] - run_start).total_seconds() >= hold_seconds:
            return run_start
    return None


def _slice(health: pd.DataFrame, arm: str, workload: str,
           lo: pd.Timestamp | None = None, hi: pd.Timestamp | None = None) -> pd.DataFrame:
    sub = health[(health["arm"] == arm) & (health["workload"] == workload)]
    if lo is not None:
        sub = sub[sub["ts"] >= lo]
    if hi is not None:
        sub = sub[sub["ts"] <= hi]
    return sub.sort_values("ts").reset_index(drop=True)


def serving(health_slice: pd.DataFrame) -> pd.Series:
    """The arm-blind SLI: is this workload serving at this sample?

    `ready_replicas >= 1` AND the readiness probe actually answered. Both arms
    run byte-identical probes (`attest_manifest_parity` verifies that), so this
    predicate means the same thing on both sides -- which is the only reason
    the two arms' numbers can be subtracted.
    """
    ready = pd.to_numeric(health_slice["ready_replicas"], errors="coerce").fillna(0) >= 1
    probe = health_slice["probe_ok"].astype(str).str.lower().isin(["true", "1", "yes"])
    return (ready & probe).reset_index(drop=True)


# --- outcomes ---------------------------------------------------------------

def outage_outcome(health_slice: pd.DataFrame, t0: pd.Timestamp,
                   hold_seconds: float = HOLD_SECONDS) -> dict:
    """Recovery time for a fault that actually takes the service down.

    `t0` is the injection start, not `failure_ts`: we find the failure instant
    in the health series itself rather than trusting the injector's note. For
    POD_KILL the two agree. For MEMORY_LEAK the injector's `failure_ts` is a
    best-of-three guess (restart count, mem pegged, stress-ng dying early --
    see `collect/chaos.py`) and, more to the point, it is written by the arm's
    own injection code; deriving the failure instant from the symmetric health
    series instead keeps both arms judged by one yardstick.
    """
    if health_slice.empty:
        return dict(outcome=NO_DATA, t_fail=None, t_recovered=None,
                    recovery_seconds=None, degradation_seconds=None, samples=0)

    ok = serving(health_slice)
    ts = health_slice["ts"].reset_index(drop=True)
    down = ts[~ok]
    down = down[down >= t0]
    if down.empty:
        # Never stopped serving. For MEMORY_LEAK on the treatment arm this is
        # the win condition -- the restart landed before the OOMKill, so there
        # was no outage to recover from. It is NOT a recovery time of zero, and
        # it is counted in its own column.
        return dict(outcome=NO_OUTAGE, t_fail=None, t_recovered=None,
                    recovery_seconds=None, degradation_seconds=None, samples=len(ts))

    t_fail = down.min()
    after = health_slice[health_slice["ts"] >= t_fail]
    t_rec = first_sustained(after["ts"], serving(after), hold_seconds)
    if t_rec is None:
        return dict(outcome=NOT_RECOVERED, t_fail=t_fail, t_recovered=None,
                    recovery_seconds=None, degradation_seconds=None, samples=len(ts))
    return dict(outcome=RECOVERED, t_fail=t_fail, t_recovered=t_rec,
                recovery_seconds=(t_rec - t_fail).total_seconds(),
                degradation_seconds=None, samples=len(ts))


def latency_band(baseline: pd.DataFrame) -> float | None:
    """Upper edge of "normal" probe latency, from THIS arm's own samples before
    THIS pair's injection.

    Per-arm and per-pair on purpose. A shared or cross-arm baseline would let a
    standing difference between the namespaces (different node, warmer page
    cache, whatever) show up as a recovery-time result, which is the single
    easiest way to manufacture a win that isn't one.
    """
    vals = pd.to_numeric(baseline.get("probe_latency_ms"), errors="coerce").dropna()
    if len(vals) < MIN_BASELINE_SAMPLES:
        return None
    med = float(vals.median())
    mad = float((vals - med).abs().median())
    return med + max(MAD_K * mad, LATENCY_BAND_FLOOR_MS)


def degradation_outcome(health_slice: pd.DataFrame, t0: pd.Timestamp,
                        baseline: pd.DataFrame,
                        hold_seconds: float = HOLD_SECONDS) -> dict:
    """Degraded-time for a fault that never takes the service down.

    CPU_HOG and DISK_STRESS keep every probe answering, so `outage_outcome`
    would return NO_OUTAGE for every run on both arms and the row would read
    like a tie when nothing was measured at all. This instead times how long
    probe latency sat above the arm's own pre-injection band.

    `recovery_seconds` stays None here, permanently and by construction: this
    is a different quantity and the per-fault table keeps it in a different
    column so no aggregation can mix the two.

    Weak-SLI caveat, repeated because it governs how the number may be read: a
    readiness probe is not user traffic. A NO_DEGRADATION row means the probe
    did not notice, not that users would not have. Replacing this with a k6
    request-latency column from `collect/load/` is the upgrade path and needs
    no change to any definition above.
    """
    if health_slice.empty:
        return dict(outcome=NO_DATA, t_fail=None, t_recovered=None,
                    recovery_seconds=None, degradation_seconds=None, samples=0)

    band = latency_band(baseline)
    if band is None:
        return dict(outcome=NO_DATA, t_fail=None, t_recovered=None,
                    recovery_seconds=None, degradation_seconds=None,
                    samples=len(health_slice))

    lat = pd.to_numeric(health_slice["probe_latency_ms"], errors="coerce")
    ts = health_slice["ts"].reset_index(drop=True)
    bad = (lat > band).fillna(False).reset_index(drop=True)
    hit = ts[bad & (ts >= t0)]
    if hit.empty:
        return dict(outcome=NO_DEGRADATION, t_fail=None, t_recovered=None,
                    recovery_seconds=None, degradation_seconds=None, samples=len(ts))

    t_deg = hit.min()
    after = health_slice[health_slice["ts"] >= t_deg].reset_index(drop=True)
    within = ~(pd.to_numeric(after["probe_latency_ms"], errors="coerce") > band).fillna(False)
    t_rec = first_sustained(after["ts"], within, hold_seconds)
    if t_rec is None:
        return dict(outcome=STILL_DEGRADED, t_fail=t_deg, t_recovered=None,
                    recovery_seconds=None, degradation_seconds=None, samples=len(ts))
    return dict(outcome=DEGRADED, t_fail=t_deg, t_recovered=t_rec,
                recovery_seconds=None,
                degradation_seconds=(t_rec - t_deg).total_seconds(), samples=len(ts))


# --- per-pair-arm measurement ----------------------------------------------

def measure(pairs: pd.DataFrame, health: pd.DataFrame,
            hold_seconds: float = HOLD_SECONDS,
            skew_tolerance: float = SKEW_TOLERANCE_SECONDS) -> pd.DataFrame:
    """One row per (pair, arm). The pairs file has two rows per `pair_id`, one
    per arm, which is what makes the comparison paired rather than two
    independent campaigns compared after the fact.
    """
    rows = []
    for _, p in pairs.iterrows():
        kind = RECOVERY_KIND.get(p["fault_type"])
        if kind is None:
            raise ValueError(
                f"fault_type {p['fault_type']!r} has no entry in RECOVERY_KIND. "
                "Add one deliberately -- a new fault type has to declare whether it "
                "produces an outage or a degradation before it can be measured.")

        t0 = p["start_ts"]
        baseline = _slice(health, p["arm"], p["workload"], hi=t0)
        window = _slice(health, p["arm"], p["workload"], lo=t0)
        if kind == OUTAGE:
            res = outage_outcome(window, t0, hold_seconds)
        else:
            res = degradation_outcome(window, t0, baseline, hold_seconds)

        skew = pd.to_numeric(pd.Series([p.get("skew_s")]), errors="coerce").iloc[0]
        rows.append(dict(pair_id=p["pair_id"], arm=p["arm"], fault_type=p["fault_type"],
                         workload=p["workload"], pattern=p.get("pattern"), kind=kind,
                         skew_ok=bool(pd.isna(skew) or abs(skew) <= skew_tolerance),
                         **res))
    return pd.DataFrame(rows, columns=MEASURED_COLUMNS)


# --- reporting --------------------------------------------------------------

def lever_for(fault_type: str) -> str:
    """What `control/policy.py` would actually do about this class -- and so
    whether a difference between the arms was ever possible.

    POD_KILL has no row in POLICY, so it falls through to UNKNOWN -> "nothing":
    the treatment arm is specified to do exactly what stock Kubernetes does.
    DISK_STRESS maps to alert_only, which is also nothing, on purpose. Those
    two rows are designed ties, and that belongs in the table next to them.
    """
    entry = POLICY.get(fault_type)
    if entry is None:
        return POLICY["UNKNOWN"]["action"]
    return entry["action"]


def per_fault_arm_table(measured: pd.DataFrame) -> pd.DataFrame:
    """Ground rule 6: per fault type, per arm, never one aggregate.

    Counts are reported beside every median because the counts are the result
    for the outage faults. "MEMORY_LEAK: prodrome 0 outages in 15 pairs,
    control 12 outages, median recovery 41s" is a stronger and more honest
    claim than any median the treatment arm could produce, and a table that
    only showed medians would hide it entirely.
    """
    rows = []
    for (fault, arm), g in measured.groupby(["fault_type", "arm"], dropna=False):
        kind = RECOVERY_KIND.get(fault)
        rec = g["recovery_seconds"].dropna()
        deg = g["degradation_seconds"].dropna()
        rows.append(dict(
            fault_type=fault, kind=kind, arm=arm, lever=lever_for(fault),
            comparable=lever_for(fault) not in ("nothing", "alert_only"),
            pairs=len(g),
            n_outage=int((g["outcome"] == RECOVERED).sum() + (g["outcome"] == NOT_RECOVERED).sum()),
            n_no_outage=int((g["outcome"] == NO_OUTAGE).sum()),
            n_degraded=int((g["outcome"] == DEGRADED).sum()),
            n_no_degradation=int((g["outcome"] == NO_DEGRADATION).sum()),
            n_censored=int((g["outcome"] == NOT_RECOVERED).sum()
                           + (g["outcome"] == STILL_DEGRADED).sum()),
            n_no_data=int((g["outcome"] == NO_DATA).sum()),
            median_recovery_s=round(float(rec.median()), 1) if len(rec) else None,
            p90_recovery_s=round(float(rec.quantile(0.9)), 1) if len(rec) else None,
            median_degradation_s=round(float(deg.median()), 1) if len(deg) else None,
        ))
    out = pd.DataFrame(rows)
    return out.sort_values(["fault_type", "arm"]).reset_index(drop=True)


def sampling_resolution(health: pd.DataFrame) -> float:
    """The cadence the health sampler ACHIEVED, not the one it was asked for.

    Every recovery time is quantised by this, so it is the floor on any
    difference the harness can honestly claim. Measured live against this
    cluster it came out at 6.4s against a requested 5.0s -- each `kubectl` call
    costs 600-900ms of startup and API round-trip, and that overhead is the
    real limit on this measurement's precision.

    Returns the worst (largest) per-arm median interval, so the band is set by
    the arm we watched least closely rather than by an average that hides it.
    """
    worst = 0.0
    for _, g in health.groupby("arm"):
        d = g.sort_values("ts")["ts"].diff().dt.total_seconds().dropna()
        if len(d):
            worst = max(worst, float(d.median()))
    return worst or HEALTH_SAMPLE_SECONDS


def paired_delta_table(measured: pd.DataFrame,
                       tie_band: float = TIE_BAND_SECONDS) -> pd.DataFrame:
    """The actual claim, computed pairwise.

    Two arms that took the same fault at the same instant under the same load
    share everything except Prodrome, so the per-pair difference removes the
    between-pair variance that an unpaired comparison of medians leaves in. A
    pair contributes only if BOTH arms produced the same measurable outcome
    and the injection skew was inside tolerance -- anything else goes in the
    `excluded` count with its reason, never into the delta.

    `n_prevented_by_prodrome` is reported separately and is not a time at all:
    pairs where the control arm had an outage and the treatment arm did not.
    For MEMORY_LEAK that is the entire thesis of the project, and it cannot be
    expressed as a difference of two durations because one of them does not
    exist.

    Note which way the exclusions bias the answer. A pair where the control arm
    is NOT_RECOVERED -- still down when we stopped watching -- has no number to
    subtract, so it leaves the delta even though it is the control arm failing
    as badly as it can. Every exclusion here therefore discards a Prodrome
    advantage rather than manufacturing one, which is the safe direction, but
    it means the delta UNDERSTATES the gap and the counts beside it are not
    decoration. `excluded_censored` makes that visible instead of letting those
    pairs vanish into a single `excluded` total.
    """
    rows = []
    for fault, g in measured.groupby("fault_type", dropna=False):
        kind = RECOVERY_KIND.get(fault)
        col = "recovery_seconds" if kind == OUTAGE else "degradation_seconds"
        wide = g.pivot_table(index="pair_id", columns="arm",
                             values=col, aggfunc="first", dropna=False)
        outc = g.pivot_table(index="pair_id", columns="arm",
                             values="outcome", aggfunc="first", dropna=False)
        skew_ok = g.groupby("pair_id")["skew_ok"].all()

        for arm in ARMS:
            if arm not in wide.columns:
                wide[arm] = pd.NA
            if arm not in outc.columns:
                outc[arm] = pd.NA

        usable = (wide[TREATMENT_ARM].notna() & wide[CONTROL_ARM].notna()
                  & skew_ok.reindex(wide.index).fillna(False))
        deltas = (pd.to_numeric(wide.loc[usable, TREATMENT_ARM], errors="coerce")
                  - pd.to_numeric(wide.loc[usable, CONTROL_ARM], errors="coerce"))

        prevented = int((outc[TREATMENT_ARM].isin([NO_OUTAGE, NO_DEGRADATION])
                         & outc[CONTROL_ARM].isin([RECOVERED, NOT_RECOVERED,
                                                   DEGRADED, STILL_DEGRADED])).sum())
        inverse = int((outc[CONTROL_ARM].isin([NO_OUTAGE, NO_DEGRADATION])
                       & outc[TREATMENT_ARM].isin([RECOVERED, NOT_RECOVERED,
                                                   DEGRADED, STILL_DEGRADED])).sum())
        rows.append(dict(
            fault_type=fault, kind=kind, lever=lever_for(fault),
            comparable=lever_for(fault) not in ("nothing", "alert_only"),
            pairs=int(len(wide)), pairs_in_delta=int(usable.sum()),
            excluded=int(len(wide) - usable.sum()),
            excluded_skew=int((~skew_ok.reindex(wide.index).fillna(False)).sum()),
            # Pairs dropped because an arm was censored (still down / still
            # degraded when we stopped watching). Surfaced on its own because
            # these are mostly control-arm failures, i.e. discarded Prodrome
            # wins -- an `excluded` total alone would hide that.
            excluded_censored=int((outc[TREATMENT_ARM].isin([NOT_RECOVERED, STILL_DEGRADED])
                                   | outc[CONTROL_ARM].isin([NOT_RECOVERED,
                                                             STILL_DEGRADED])).sum()),
            tie_band_s=round(tie_band, 2),
            median_delta_s=round(float(deltas.median()), 1) if len(deltas) else None,
            # Directional counts respect the tie band: a pair only counts for an
            # arm if it beat the other by more than the measurement resolution.
            prodrome_faster_pairs=int((deltas < -tie_band).sum()) if len(deltas) else 0,
            control_faster_pairs=int((deltas > tie_band).sum()) if len(deltas) else 0,
            tied_pairs=int((deltas.abs() <= tie_band).sum()) if len(deltas) else 0,
            n_prevented_by_prodrome=prevented,
            n_prevented_in_control_only=inverse,
        ))
    return pd.DataFrame(rows).sort_values("fault_type").reset_index(drop=True)


# --- attestation: is this experiment even capable of a comparison? ---------

@dataclass(frozen=True)
class Attestation:
    name: str
    ok: bool
    detail: str


def _normalise_manifest(text: str) -> list[str]:
    """Strip what is SUPPOSED to differ between the arms (the namespace) and
    what carries no behaviour (comments, blank lines), so what remains is
    exactly the behaviour the two arms share.

    Text-normalised rather than YAML-parsed on purpose: `pyyaml` is not in
    `requirements.txt`, and ground rule 8 says a result has to reproduce from a
    fresh clone. Adding a dependency so a check can run is the wrong trade when
    the check is a diff of two files generated from each other by `sed` (see
    `docs/guides/shravan.md` 3.6).
    """
    out = []
    for line in text.splitlines():
        line = re.sub(r"\s+#.*$", "", line)
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        line = re.sub(r"namespace:\s*control\s*$", "namespace: prodrome", line)
        line = re.sub(r"name:\s*control\s*$", "name: prodrome", line)
        out.append(line.rstrip())
    return out


def attest_manifest_parity(treatment_yaml: str | Path = "infra/workloads.yaml",
                           control_yaml: str | Path = "infra/workloads-control.yaml") -> Attestation:
    """Identical workloads, identical limits, identical probes -- or the
    comparison is between two different systems.

    This is the cheapest and most load-bearing check in the module: resource
    limits set the OOMKill threshold and the probes set how fast Kubernetes
    notices anything, so a one-line drift in either file silently becomes the
    "result". Both files are cluster's (`infra/`), read-only from here; a
    failure is a message for that seat, not something to fix in place.
    """
    a = Path(treatment_yaml)
    b = Path(control_yaml)
    if not a.exists() or not b.exists():
        return Attestation("manifest-parity", False, f"missing {a if not a.exists() else b}")
    la, lb = _normalise_manifest(a.read_text()), _normalise_manifest(b.read_text())
    if la == lb:
        return Attestation("manifest-parity", True,
                           f"{a.name} == {b.name} ({len(la)} significant lines) "
                           "after normalising namespace/comments")
    diff = [f"  -{x}" for x in la if x not in lb][:4] + [f"  +{y}" for y in lb if y not in la][:4]
    return Attestation("manifest-parity", False,
                       "arms are NOT identical; comparison invalid:\n" + "\n".join(diff))


def attest_arm_isolation(controller_namespace: str) -> Attestation:
    """The controller must not be able to act on the control arm.

    `control/controller.py` pins `NAMESPACE` at module scope, so this reads the
    value rather than guessing. If the controller ever becomes namespace-
    configurable, the two-arm runner has to assert the control arm is excluded,
    and this check is where that will fail loudly.
    """
    if controller_namespace == CONTROL_ARM:
        return Attestation("arm-isolation", False,
                           f"controller NAMESPACE is {controller_namespace!r} -- it would "
                           "remediate the control arm, so there is no control arm")
    return Attestation("arm-isolation", True,
                       f"controller acts on {controller_namespace!r} only; "
                       f"{CONTROL_ARM!r} is untouched")


def attest_treatment_arm_live(decisions: pd.DataFrame | None,
                              dry_run: bool | None = None) -> Attestation:
    """The treatment arm has to have actually DONE something.

    The failure this exists to prevent: run the two-arm experiment with
    `control/controller.py`'s `DRY_RUN = True`, get two statistically
    indistinguishable arms, and report "Prodrome ties stock Kubernetes" when
    what actually happened is that Prodrome was never switched on. In shadow
    mode `execute_action()` returns "dry-run" before touching the API, so the
    treatment arm IS a second control arm and every row is a tautology.

    Checks the decision log's `mode` column (SETUP.md 7) because that is the
    recorded fact, falling back to the passed `DRY_RUN` flag when there is no
    log yet. A shadow-mode run is still worth collecting -- it is the null
    experiment, and a null experiment that yields a difference means something
    is wrong with the harness -- but it must not be published as a comparison.
    """
    if decisions is not None and len(decisions):
        if "mode" not in decisions.columns:
            return Attestation("treatment-arm-live", False,
                               "decision log has no `mode` column, so shadow cannot be "
                               "told from live (SETUP.md 7)")
        modes = set(decisions["mode"].dropna().astype(str).str.lower())
        executed = int((decisions.get("result", pd.Series(dtype=str))
                        .astype(str) == "executed").sum())
        if modes == {"shadow"}:
            return Attestation("treatment-arm-live", False,
                               f"every one of {len(decisions)} decision rows is mode=shadow: "
                               "the treatment arm took no action, so any tie is a tautology")
        if executed == 0:
            return Attestation("treatment-arm-live", False,
                               f"modes={sorted(modes)} but 0 rows with result=executed: "
                               "no remediation was actually applied")
        return Attestation("treatment-arm-live", True,
                           f"{executed} executed action(s) across {len(decisions)} decisions, "
                           f"modes={sorted(modes)}")
    if dry_run is None:
        return Attestation("treatment-arm-live", False,
                           "no decision log and no DRY_RUN value -- cannot show the "
                           "treatment arm did anything")
    if dry_run:
        return Attestation("treatment-arm-live", False,
                           "control/controller.py DRY_RUN=True: execute_action() returns "
                           "'dry-run' before touching the API, so the treatment arm is a "
                           "second control arm")
    return Attestation("treatment-arm-live", True, "DRY_RUN=False and no log read yet")


def attest_pair_skew(pairs: pd.DataFrame,
                     tolerance: float = SKEW_TOLERANCE_SECONDS) -> Attestation:
    """Both arms took the fault at the same moment, under the same load."""
    skew = pd.to_numeric(pairs.get("skew_s"), errors="coerce").abs().dropna()
    if skew.empty:
        return Attestation("pair-skew", False,
                           "no skew_s recorded -- cannot show the arms were simultaneous")
    bad = int((skew > tolerance).sum())
    n_pairs = pairs["pair_id"].nunique()
    if bad:
        return Attestation("pair-skew", False,
                           f"{bad}/{len(skew)} arm-rows exceeded {tolerance}s skew "
                           f"(max {skew.max():.1f}s); those pairs are excluded from the delta")
    return Attestation("pair-skew", True,
                       f"{n_pairs} pairs, max skew {skew.max():.1f}s <= {tolerance}s")


def attest_load_parity(pairs: pd.DataFrame) -> Attestation:
    """Identical load against both namespaces, per `docs/guides/shaurya.md`
    Part 4: "both arms need identical conditions or the comparison is
    meaningless." The runner records an attestation rather than inferring it,
    because an idle control arm recovers faster for reasons that have nothing
    to do with Prodrome.
    """
    col = pairs.get("load_attested")
    if col is None:
        return Attestation("load-parity", False, "no load_attested column recorded")
    flags = col.astype(str).str.lower().isin(["true", "1", "yes"])
    if flags.all():
        return Attestation("load-parity", True, f"load attested on all {len(flags)} arm-rows")
    return Attestation("load-parity", False,
                       f"{int((~flags).sum())}/{len(flags)} arm-rows ran without attested "
                       "load on both arms")


def attest(pairs: pd.DataFrame | None = None,
           decisions: pd.DataFrame | None = None,
           controller_namespace: str | None = None,
           dry_run: bool | None = None) -> list[Attestation]:
    """Every precondition a recovery-time comparison needs, each independently
    checkable. `publishable()` is the AND of them."""
    out = [attest_manifest_parity()]
    if controller_namespace is not None:
        out.append(attest_arm_isolation(controller_namespace))
    out.append(attest_treatment_arm_live(decisions, dry_run))
    if pairs is not None and len(pairs):
        out.append(attest_pair_skew(pairs))
        out.append(attest_load_parity(pairs))
    return out


def publishable(attestations: list[Attestation]) -> bool:
    return all(a.ok for a in attestations)


def render(per_fault: pd.DataFrame, deltas: pd.DataFrame,
           attestations: list[Attestation]) -> str:
    """The report, with its own caveats attached. Ground rule 7: say what it
    cannot do, in the report itself -- not in a README somebody will not open
    next to the number they are quoting."""
    ok = publishable(attestations)
    lines = ["=== control-arm attestation ==="]
    for a in attestations:
        lines.append(f"  [{'ok ' if a.ok else 'FAIL'}] {a.name}: {a.detail}")
    lines.append("")
    lines.append("PUBLISHABLE AS A CONTROL-ARM COMPARISON: "
                 + ("yes" if ok else "NO -- the numbers below describe this run, "
                                     "they do not compare Prodrome to stock Kubernetes"))
    lines += ["", "=== recovery per fault type, per arm (ground rule 6) ===",
              per_fault.to_string(index=False) if len(per_fault) else "  (no pairs measured)",
              "", "=== paired delta, treatment minus control (negative = Prodrome faster) ===",
              deltas.to_string(index=False) if len(deltas) else "  (no pairs measured)",
              "", "=== what this cannot tell you (ground rule 7) ===",
              "  - DISK_STRESS maps to alert_only and POD_KILL has no POLICY row, so for both",
              "    the treatment arm is SPECIFIED to behave like the control arm. A tie there is",
              "    the design, not a result. Only CPU_HOG and MEMORY_LEAK can move.",
              "  - recovery_seconds (POD_KILL, MEMORY_LEAK) and degradation_seconds (CPU_HOG,",
              "    DISK_STRESS) are different quantities and are never pooled. CPU_HOG and",
              "    DISK_STRESS have no failure instant at all -- runs.csv's failure_ts is null for",
              "    all 120 of them -- so they have no recovery time to report.",
              "  - degradation is measured from readiness-probe latency, not user traffic. It is a",
              "    lower bound on impact; NO_DEGRADATION means the probe did not notice.",
              "  - NO_OUTAGE is counted, never scored as 0s. A prevented failure has no recovery",
              "    time, and the prevented count is the claim for MEMORY_LEAK.",
              "  - POD_KILL is instant: no lead time is possible, so no remediation can beat",
              "    kubelet's own restart. It is in the table as a sanity floor, not a target.",
              "  - every recovery time is quantised by the health sampling cadence, which is",
              "    dominated by kubectl startup (600-900ms per call). Differences at or below",
              "    tie_band_s are reported as ties; the harness cannot resolve them."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="data/twoarm",
                    help="where collect/twoarm.py wrote pairs.csv and health.csv")
    ap.add_argument("--decisions", default="data/decisions/log.csv")
    ap.add_argument("--out", default="eval/recovery-results.csv")
    args = ap.parse_args()

    indir = Path(args.indir)
    if not (indir / "pairs.csv").exists():
        # Not an error: this is the expected state until a cluster exists. Print
        # the attestations that CAN be checked offline, which is most of them,
        # so the preconditions are known good before anyone spends 2.5 hours.
        from control import controller
        atts = attest(controller_namespace=controller.NAMESPACE, dry_run=controller.DRY_RUN)
        print(f"no {indir}/pairs.csv -- no two-arm run has happened yet.\n")
        print(render(pd.DataFrame(), pd.DataFrame(), atts))
        print("\nTo produce one:  python collect/twoarm.py --campaign   (needs a live cluster)")
        return

    pairs, health = load(indir)
    decisions = None
    dpath = Path(args.decisions)
    if dpath.exists():
        decisions = pd.read_csv(dpath)

    from control import controller
    measured = measure(pairs, health)
    per_fault = per_fault_arm_table(measured)
    # Tie band from the cadence actually achieved, not the one requested.
    resolution = sampling_resolution(health)
    deltas = paired_delta_table(measured, tie_band=resolution)
    print(f"health sampling resolution achieved: {resolution:.2f}s "
          f"(requested {HEALTH_SAMPLE_SECONDS:.1f}s) -- differences at or below "
          f"this are reported as ties, not wins\n")
    atts = attest(pairs=pairs, decisions=decisions,
                  controller_namespace=controller.NAMESPACE, dry_run=controller.DRY_RUN)

    print(render(per_fault, deltas, atts))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    per_fault.to_csv(args.out, index=False)
    measured.to_csv(indir / "measured.csv", index=False)
    deltas.to_csv(indir / "paired-deltas.csv", index=False)
    print(f"\nwrote {args.out}, {indir}/measured.csv, {indir}/paired-deltas.csv")


if __name__ == "__main__":
    main()
