"""Evaluation harness -- Part 7 of docs/guides/shaurya.md.

Fits Sagar's detector on data/healthy (ground rule #1: healthy data only --
this is exactly the fit_healthy() case, since healthy and chaos are separate
files with no time overlap), replays it against data/chaos, and reports the
headline numbers per fault type, never as one aggregate (guide 7.3):

    lead time     -- seconds from first firing to the failure instant
    recall        -- fraction of injected faults detected before failure
    precision     -- fraction of firings that were real faults
    fp/hour       -- precision translated into an operational number

"Failure instant" is data/chaos/runs.csv's failure_ts where it exists (the
OOMKill for MEMORY_LEAK, the kill for POD_KILL) and end_ts otherwise
(CPU_HOG / DISK_STRESS don't have a real failure to count down to -- see
collect/chaos.py's docstring; lead time for those two is really "time
before injection ended," not "time before a crash").

NOT computed here: recovery time, ours vs control. That needs Shravan's
live controller and identical faults injected into BOTH namespaces
simultaneously (Phase 4) -- this harness covers what's measurable now,
Phase 2/3, single-arm, detector-only.

Does NOT use ml.replay.replay(): that calls Detector.score() ->
WorkloadDetector.update() on every tick regardless of fault status, which is
fine over one isolated fault but wrong over this file's 100 back-to-back
faults -- each one keeps training the "healthy" reference, so the detector
progressively desensitizes to everything after it (verified by tracing a
DISK_STRESS window: an isolated copy of the same detector fires correctly,
z up to ~4; scored via the real replay() path deep into the file it doesn't
fire at all). This module's replay_no_leakage() instead mirrors
ml.replay.train_and_replay()'s fault-aware branching (score_only() inside a
labeled window, update() outside it) and additionally calls on_restart()
after any run that actually restarted the pod -- simulating the reset a
live controller would apply, which nothing else provides since Shravan's
loop isn't wired up yet.

Usage:
    python -m eval.harness
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.detector import Detector
from ml.replay import fault_mask, firing_events

HEALTHY_METRICS = Path("data/healthy/metrics.parquet")
CHAOS_METRICS = Path("data/chaos/metrics.parquet")
CHAOS_LABELS = Path("data/chaos/labels.csv")
CHAOS_RUNS = Path("data/chaos/runs.csv")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    healthy = pd.read_parquet(HEALTHY_METRICS)
    chaos = pd.read_parquet(CHAOS_METRICS)
    labels = pd.read_csv(CHAOS_LABELS, parse_dates=["start_ts", "end_ts"])
    runs = pd.read_csv(CHAOS_RUNS, parse_dates=["start_ts", "end_ts", "failure_ts"])
    return healthy, chaos, labels, runs


def replay_no_leakage(det: Detector, chaos: pd.DataFrame, labels: pd.DataFrame,
                      runs: pd.DataFrame) -> pd.DataFrame:
    """Replay an already-healthy-fit detector over a long, multi-fault chaos
    file WITHOUT training on fault ticks and WITH the post-restart reset a
    live controller would apply.

    ml.replay.replay() calls Detector.score() -> WorkloadDetector.update() on
    every tick, fault or not -- correct for one isolated fault (see the trace
    in this module's history), but over a file with 100 faults back-to-back
    it keeps folding each fault's abnormal values into the "healthy"
    reference, so the detector desensitizes more with every fault it walks
    through and z-scores collapse by the time it reaches later runs. Ground
    rule #1 says fault ticks must never update the baseline; this mirrors
    ml.replay.train_and_replay's fault-aware branching (score_only() inside a
    labeled window, update() outside it) but starting from a detector already
    fit on a separate healthy file, and additionally calls on_restart() after
    any run that actually restarted the pod -- the reset a live controller
    would trigger, which an offline multi-fault replay has no other way to
    get since Shravan's loop isn't wired up yet (guide Part 4.5).
    """
    is_fault = fault_mask(chaos, labels)
    restarted = runs[runs.get("restarts_delta", 0) > 0] if "restarts_delta" in runs else runs.iloc[0:0]

    records = []
    for workload, group in chaos.groupby("workload"):
        group = group.sort_values("ts")
        wd = det.workloads[workload]
        flags = is_fault.loc[group.index]
        pending_ends = sorted(restarted[restarted["workload"] == workload]["end_ts"])
        next_end = pending_ends.pop(0) if pending_ends else None

        for (_, row), is_f in zip(group.iterrows(), flags):
            values = {m: row[m] for m in wd.metrics}
            z_scores, fired = wd.score_only(values) if is_f else wd.update(values)
            finite = [z for z in z_scores.values() if z is not None]
            records.append((row["ts"], workload, max(finite) if finite else 0.0, fired))
            while next_end is not None and row["ts"] >= next_end:
                det.on_restart(workload)
                next_end = pending_ends.pop(0) if pending_ends else None

    return pd.DataFrame(records, columns=["ts", "workload", "score", "fired"])


def fit_and_replay(healthy: pd.DataFrame, chaos: pd.DataFrame, labels: pd.DataFrame,
                   runs: pd.DataFrame) -> tuple[Detector, pd.DataFrame]:
    det = Detector.fit_healthy(healthy)
    log = replay_no_leakage(det, chaos, labels, runs)
    return det, log


def _failure_instant(run: pd.Series) -> pd.Timestamp:
    return run["failure_ts"] if pd.notna(run["failure_ts"]) else run["end_ts"]


# How long a restart's on_restart() reset makes the NEXT fault invisible:
# WARMUP_TICKS (30) to refill the wiped error history, or
# POST_RESTART_SUPPRESS_TICKS (32) of explicit fired-suppression, whichever
# binds -- 15 minutes covers either at 15s ticks with margin.
RESTART_DEAD_ZONE = pd.Timedelta(minutes=15)


def _contaminated(run: pd.Series, restarts: pd.DataFrame) -> bool:
    """True if a same-workload restart happened recently enough before this
    run that the detector's post-restart reset (guide 4.4) was still active --
    i.e. this run can't fairly be blamed on weak detection, only on timing."""
    prior = restarts[
        (restarts["workload"] == run["workload"]) & (restarts["end_ts"] < run["start_ts"])
    ]
    if prior.empty:
        return False
    return (run["start_ts"] - prior["end_ts"].max()) < RESTART_DEAD_ZONE


def per_run_detection(runs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per fault run: was it detected before failure, how early, and
    whether it landed inside another fault's post-restart dead zone."""
    restarts = runs[runs.get("restarts_delta", 0) > 0] if "restarts_delta" in runs else runs.iloc[0:0]
    rows = []
    for _, run in runs[runs["run_type"] == "fault"].iterrows():
        failure = _failure_instant(run)
        onsets = events[
            (events["workload"] == run["workload"])
            & (events["onset_ts"] >= run["start_ts"])
            & (events["onset_ts"] <= failure)
        ]
        detected = not onsets.empty
        lead_s = (failure - onsets["onset_ts"].min()).total_seconds() if detected else None
        rows.append(dict(run_id=run["run_id"], fault_type=run["fault_type"],
                         workload=run["workload"], detected=detected, lead_seconds=lead_s,
                         contaminated=_contaminated(run, restarts)))
    return pd.DataFrame(rows)


def per_fault_table(detection: pd.DataFrame) -> pd.DataFrame:
    """Guide 7.3's table: never one aggregate row. Reports recall twice --
    raw, and again excluding runs contaminated by a preceding restart's dead
    zone (guide 5.3's "if a number looks off, check for leakage" applied to
    the timing side rather than the labeling side)."""
    def agg(g):
        return pd.Series(dict(
            runs=len(g),
            recall=round(g["detected"].mean(), 3),
            median_lead_s=round(g["lead_seconds"].median(), 1),
            contaminated=int(g["contaminated"].sum()),
        ))

    raw = detection.groupby("fault_type").apply(agg, include_groups=False)
    clean = (detection[~detection["contaminated"]].groupby("fault_type")
             .apply(agg, include_groups=False))

    out = raw.reset_index().rename(columns={"recall": "recall_raw"})
    out["recall_excl_dead_zone"] = out["fault_type"].map(clean["recall"])
    out["runs"] = out["runs"].astype(int)
    out["contaminated"] = out["contaminated"].astype(int)
    out["recovery_ours_s"] = "N/A"
    out["recovery_control_s"] = "N/A"
    return out[["fault_type", "runs", "contaminated", "recall_raw",
               "recall_excl_dead_zone", "median_lead_s", "recovery_ours_s", "recovery_control_s"]]


def false_positive_stats(events: pd.DataFrame, healthy: pd.DataFrame,
                         runs: pd.DataFrame) -> dict:
    """Precision and FP/hour, measured against KNOWN-clean time: the whole
    healthy dataset plus the campaign's dedicated CLEAN_* windows (guide
    5.5: "the clean runs are how you measure false positives"). Firings
    inside any fault window count as true positives; everything else, real
    firings during clean time, counts as false."""
    clean_runs = runs[runs["run_type"] == "clean"]
    fault_runs = runs[runs["run_type"] == "fault"]

    def during_a_fault(onset_ts, workload) -> bool:
        rows = fault_runs[fault_runs["workload"] == workload]
        return ((rows["start_ts"] <= onset_ts) & (onset_ts <= rows["end_ts"])).any()

    tp = fp = 0
    for _, ev in events.iterrows():
        if during_a_fault(ev["onset_ts"], ev["workload"]):
            tp += 1
        else:
            fp += 1

    healthy_hours = (healthy["ts"].max() - healthy["ts"].min()).total_seconds() / 3600
    clean_hours = clean_runs["end_ts"].sub(clean_runs["start_ts"]).dt.total_seconds().sum() / 3600
    total_clean_hours = healthy_hours + clean_hours

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    fp_per_hour = fp / total_clean_hours if total_clean_hours else float("nan")
    return dict(true_positive_events=tp, false_positive_events=fp,
               precision=round(precision, 3), fp_per_hour=round(fp_per_hour, 3),
               clean_hours_measured=round(total_clean_hours, 2))


def main() -> None:
    healthy, chaos, labels, runs = load()
    print(f"healthy: {healthy.shape}  chaos: {chaos.shape}  "
          f"fault runs: {(runs.run_type == 'fault').sum()}  clean: {(runs.run_type == 'clean').sum()}")

    det, log = fit_and_replay(healthy, chaos, labels, runs)
    events = firing_events(log)
    print(f"detector fired {len(events)} times across the chaos window")

    detection = per_run_detection(runs, events)
    table = per_fault_table(detection)
    print("\n--- per-fault-type results (guide 7.3) ---")
    print(table.to_string(index=False))

    fp_stats = false_positive_stats(events, healthy, runs)
    print("\n--- precision / false positives (guide 7.1) ---")
    for k, v in fp_stats.items():
        print(f"  {k}: {v}")

    print("\nrecovery time (ours vs control): N/A -- needs the live controller "
          "and faults injected into both namespaces (Phase 4).")

    out = Path("eval/results.csv")
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
