"""Replay harness (Part 4.2).

Two ways to run a detector against stored data:

- replay(): score an ALREADY-FIT detector against a file. Only correct
  when that file doesn't overlap in time with whatever the detector was
  fit on (e.g. fit on data/healthy/*, replay data/chaos/*) -- otherwise
  you re-walk ticks the detector already trained on, restarting from a
  point in time the detector's internal state has moved past. Requires
  `labels` for any file with more than one fault: see the fixed-bug note
  on replay() itself for why.

- train_and_replay(): for a SINGLE file with healthy and fault ticks
  interleaved in time (like the sample fixture) -- one continuous pass
  in timestamp order, training (update()) on healthy ticks and only
  scoring (score_only()) on ticks inside a labeled fault window. Ground
  rule #1 still holds (fault ticks never update the baseline); this just
  avoids replaying the same already-covered range twice.
"""

import pandas as pd

from ml.detector import ALPHA, K_OF_N_METRICS, N_CONSECUTIVE, Z_THRESHOLD, Detector


def replay(detector, metrics: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Score every tick in `metrics` against `detector`, one workload at a
    time, oldest first. `detector` must already be fit -- this only scores,
    never trains on a labeled fault tick.

    Bug fixed here, credit to eval/harness.py's replay_no_leakage() for
    catching it: this used to call Detector.score() -> WorkloadDetector.
    update() on every tick regardless of fault status. Fine over one
    isolated fault, wrong over any file with several faults back-to-back
    (a real chaos campaign, not just the tiny bundled sample) -- each
    fault's abnormal values got folded into the "healthy" reference, so
    the detector progressively desensitized to everything after it
    (verified by tracing a DISK_STRESS window: fires correctly in
    isolation, z~4; scored deep into a multi-fault file via the old path,
    doesn't fire at all). Branches exactly like train_and_replay() now:
    score_only() inside a labeled window, update() outside it.

    Returns one row per tick: ts, workload, score, fired.
    """
    is_fault = fault_mask(metrics, labels)

    records = []
    for workload, group in metrics.groupby("workload"):
        group = group.sort_values("ts")
        wd = detector.workloads[workload]
        fault_flags = is_fault.loc[group.index]

        for (_, row), is_f in zip(group.iterrows(), fault_flags):
            values = {m: row[m] for m in wd.metrics}
            z_scores, fired = wd.score_only(values) if is_f else wd.update(values)

            finite = [z for z in z_scores.values() if z is not None]
            score = max(finite) if finite else 0.0
            records.append((row["ts"], workload, score, fired))

    return pd.DataFrame(records, columns=["ts", "workload", "score", "fired"])


def fault_mask(metrics: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """True for every tick that falls inside some labeled fault window, or
    has a nonzero restart count.

    The label window alone isn't quite enough: a MEMORY_LEAK's restarts
    flag stays set for a few ticks past the breach, and the label's
    end_ts is the breach instant itself -- so some restart-flagged ticks
    land just after end_ts and would otherwise leak into "healthy" data.
    restarts only ever moves on a fault (see generate.py's docstring), so
    excluding any nonzero restart tick directly closes that gap.
    """
    mask = pd.Series(False, index=metrics.index)
    for _, lab in labels.iterrows():
        rows = (
            (metrics["workload"] == lab["workload"])
            & (metrics["ts"] >= lab["start_ts"])
            & (metrics["ts"] <= lab["end_ts"])
        )
        mask |= rows
    mask |= metrics["restarts"] != 0
    return mask


def train_and_replay(
    metrics: pd.DataFrame,
    labels: pd.DataFrame,
    alpha: float = ALPHA,
    threshold: float = Z_THRESHOLD,
    k: int = K_OF_N_METRICS,
    n: int = N_CONSECUTIVE,
) -> tuple[Detector, pd.DataFrame]:
    """One continuous pass over `metrics` in timestamp order: healthy ticks
    train the detector (update()), ticks inside a labeled fault window are
    only scored (score_only()). Fixes the discontinuity from fitting on a
    healthy subset and then separately replaying the whole file again.

    Returns (fitted detector, log) where log has one row per tick:
    ts, workload, score, fired.
    """
    is_fault = fault_mask(metrics, labels)
    healthy = metrics[~is_fault]
    det = Detector._init_workloads(healthy, alpha, threshold, k, n)

    records = []
    for workload, group in metrics.groupby("workload"):
        group = group.sort_values("ts")
        wd = det.workloads[workload]
        fault_flags = is_fault.loc[group.index]

        for (_, row), is_f in zip(group.iterrows(), fault_flags):
            values = {m: row[m] for m in wd.metrics}
            z_scores, fired = wd.score_only(values) if is_f else wd.update(values)

            finite = [z for z in z_scores.values() if z is not None]
            score = max(finite) if finite else 0.0
            records.append((row["ts"], workload, score, fired))

    log = pd.DataFrame(records, columns=["ts", "workload", "score", "fired"])
    return det, log


def firing_events(log: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive fired=True ticks per workload into discrete
    events, keeping each streak's onset timestamp and peak score.

    A sustained fire during one fault is one event, not one row per tick --
    that's what lead-time measurement needs (Part 5.1).
    """
    events = []
    for workload, group in log.groupby("workload"):
        group = group.sort_values("ts").reset_index(drop=True)
        in_streak = False
        onset_ts = None
        peak_score = None

        for _, row in group.iterrows():
            if row["fired"] and not in_streak:
                in_streak = True
                onset_ts = row["ts"]
                peak_score = row["score"]
            elif row["fired"] and in_streak:
                peak_score = max(peak_score, row["score"])
            elif not row["fired"] and in_streak:
                events.append((workload, onset_ts, peak_score))
                in_streak = False

        if in_streak:
            events.append((workload, onset_ts, peak_score))

    return pd.DataFrame(events, columns=["workload", "onset_ts", "peak_score"])
