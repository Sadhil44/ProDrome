"""The detector (Signal). See docs/guides/sagar.md Parts 2-4.

Per-metric EWMA-of-error detectors, voted k-of-n across metrics per
workload. Trains only on whatever it's fed via update() -- callers must
only ever call update() with healthy data (ground rule #1 / Part 1.3).
There is no gradient descent: streaming healthy data through once *is*
the fit (Part 2.5).

Fault ticks should be scored with score_only(), which never touches the
running prediction or error history. This matters for offline evaluation
against a single file with faults interleaved in time (see
ml.replay.train_and_replay): fitting on the healthy subset and then
separately replaying the whole file from scratch creates a discontinuity
at the replay restart point, since fit_healthy() ends calibrated to the
END of the healthy timeline, not the start.
"""

import pickle
from collections import deque

import numpy as np
import pandas as pd

from ml.features import METRICS

# Chosen via the Part 5.1 sweep against the sample fixture: best recall
# among the configs at fp_per_hour=0.00 (see docs/guides/sagar.md Part
# 5.2 for the balanced-vs-precision writeup). Caveat: the sample is tiny
# enough that several configs hit zero measured false positives, so this
# isn't yet a validated precision/recall tradeoff -- revisit once
# Shaurya's real healthy/chaos data lands.
ALPHA = 0.1                    # EWMA prediction responsiveness
ERROR_HISTORY = 100            # ticks of error history kept per metric
WARMUP_TICKS = 30              # ticks before a metric starts scoring
RECENT_WINDOW = 5              # "recent" errors compared against the full history
Z_THRESHOLD = 2.5              # std devs of error above typical -> anomalous
STD_FLOOR = 1e-6               # avoids divide-by-zero on constant metrics

K_OF_N_METRICS = 3             # how many metrics must be anomalous at once
N_CONSECUTIVE = 1              # how many consecutive ticks that must hold
MIN_HEALTHY_VARIANCE = 1e-9    # metrics below this (e.g. restarts) are dropped
POST_RESTART_SUPPRESS_TICKS = 32  # ~8 minutes at 15s ticks (Part 4.4)

# A fault that only ever disturbs one metric (e.g. a pure CPU hog) can
# never clear a k>=2 vote no matter how extreme it is. Escape hatch: a
# single metric far enough past the normal threshold fires on its own,
# without needing others to agree.
EXTREME_MULTIPLIER = 2.0


class MetricDetector:
    """Tracks one metric for one workload: an EWMA predictor plus its error history."""

    def __init__(self, alpha: float = ALPHA, threshold: float = Z_THRESHOLD):
        self.alpha = alpha
        self.threshold = threshold
        self.prediction = None
        self.errors = deque(maxlen=ERROR_HISTORY)          # healthy-only reference distribution
        self.recent_errors = deque(maxlen=RECENT_WINDOW)   # always-live, tracks any observed tick

    def _z(self) -> float | None:
        """z-score of the recent window against the healthy-only reference."""
        if len(self.errors) < WARMUP_TICKS:
            return None
        errs = np.array(self.errors)
        mean_all = errs.mean()
        std_all = errs.std() + STD_FLOOR
        recent_mean = np.array(self.recent_errors).mean()
        return (recent_mean - mean_all) / std_all

    def _observe(self, value) -> float | None:
        """Update the prediction and the always-live recent-error window
        for any observed value, healthy or fault. Returns the tick's
        error, or None if this is the first-ever observation (nothing to
        compare against yet) or the value is non-finite."""
        if value is None or not np.isfinite(value):
            return None
        if self.prediction is None:
            self.prediction = float(value)
            return None

        error = abs(value - self.prediction)
        self.prediction = self.alpha * value + (1 - self.alpha) * self.prediction
        self.recent_errors.append(error)
        return error

    def update(self, value) -> tuple[float | None, bool]:
        """Observe this tick, folding its error into both the recent
        window and the long-term reference -- this IS the training step,
        so only ever call this on healthy data."""
        error = self._observe(value)
        if error is None:
            return None, False

        self.errors.append(error)
        z = self._z()
        return z, z is not None and z > self.threshold

    def score_only(self, value) -> tuple[float | None, bool]:
        """Observe this tick for prediction/recency purposes, but never let
        it redefine what's 'typical' -- the long-term reference is left
        untouched. Use on ticks known to be inside a fault window
        (ground rule #1)."""
        error = self._observe(value)
        if error is None:
            return None, False

        z = self._z()
        return z, z is not None and z > self.threshold


class WorkloadDetector:
    """k-of-n vote across a workload's active per-metric detectors."""

    def __init__(
        self,
        metrics: list[str],
        k: int = K_OF_N_METRICS,
        n: int = N_CONSECUTIVE,
        alpha: float = ALPHA,
        threshold: float = Z_THRESHOLD,
        extreme_multiplier: float = EXTREME_MULTIPLIER,
    ):
        self.metrics = metrics
        self.k = min(k, len(metrics))
        self.n = n
        self.alpha = alpha
        self.threshold = threshold
        self.extreme_threshold = threshold * extreme_multiplier
        self.detectors = {m: MetricDetector(alpha, threshold) for m in metrics}
        self.streak = 0
        self.suppress_ticks_left = 0

    def on_restart(self):
        """Reset all per-metric state and suppress firing for ~8 minutes (Part 4.4)."""
        self.detectors = {m: MetricDetector(self.alpha, self.threshold) for m in self.metrics}
        self.streak = 0
        self.suppress_ticks_left = POST_RESTART_SUPPRESS_TICKS

    def _vote(self, z_scores: dict) -> bool:
        anomalous_count = sum(1 for z in z_scores.values() if z is not None and z > self.threshold)
        extreme = any(z is not None and z > self.extreme_threshold for z in z_scores.values())
        return anomalous_count >= self.k or extreme

    def _advance(self, is_anomalous: bool) -> bool:
        self.streak = self.streak + 1 if is_anomalous else 0
        if self.suppress_ticks_left > 0:
            self.suppress_ticks_left -= 1
            return False
        return self.streak >= self.n

    def update(self, values: dict) -> tuple[dict, bool]:
        z_scores = {m: self.detectors[m].update(values.get(m))[0] for m in self.metrics}
        fired = self._advance(self._vote(z_scores))
        return z_scores, fired

    def score_only(self, values: dict) -> tuple[dict, bool]:
        z_scores = {m: self.detectors[m].score_only(values.get(m))[0] for m in self.metrics}
        fired = self._advance(self._vote(z_scores))
        return z_scores, fired


class Detector:
    """One WorkloadDetector per workload. This is the object that gets pickled
    and handed to Shravan (Part 4.5)."""

    def __init__(self):
        self.workloads: dict[str, WorkloadDetector] = {}

    @classmethod
    def _init_workloads(
        cls, healthy: pd.DataFrame, alpha: float, threshold: float, k: int, n: int
    ) -> "Detector":
        """Construct one WorkloadDetector per workload, choosing active metrics
        from healthy-data variance (the zero-variance guard). Feeds no data --
        callers decide how ticks get fed in."""
        det = cls()
        for workload, group in healthy.groupby("workload"):
            active_metrics = [
                m for m in METRICS if group[m].astype(float).var() > MIN_HEALTHY_VARIANCE
            ]
            det.workloads[workload] = WorkloadDetector(
                active_metrics, k=k, n=n, alpha=alpha, threshold=threshold
            )
        return det

    @classmethod
    def fit_healthy(
        cls,
        healthy: pd.DataFrame,
        alpha: float = ALPHA,
        threshold: float = Z_THRESHOLD,
        k: int = K_OF_N_METRICS,
        n: int = N_CONSECUTIVE,
    ) -> "Detector":
        """healthy must contain zero fault ticks -- see ground rule #1.

        For a single file with faults interleaved in time (like the sample
        fixture), use ml.replay.train_and_replay instead of fit_healthy +
        replay: fitting here ends calibrated to the END of the healthy
        timeline, so a separate replay pass that restarts from the
        beginning creates an artificial discontinuity. This method is for
        the real case where healthy and chaos come from separate files
        with no time overlap.
        """
        det = cls._init_workloads(healthy, alpha, threshold, k, n)
        for workload, group in healthy.groupby("workload"):
            group = group.sort_values("ts")
            wd = det.workloads[workload]
            for _, row in group.iterrows():
                wd.update({m: row[m] for m in wd.metrics})
        return det

    def score(self, workload: str, values) -> tuple[float, bool]:
        """values: array of 8 readings in ml.features.METRICS order.
        Returns (score, fired) -- score is the max z-score among this
        workload's active metrics this tick (0.0 while any are warming up)."""
        wd = self.workloads[workload]
        values_by_metric = dict(zip(METRICS, values))
        z_scores, fired = wd.update(values_by_metric)

        finite = [z for z in z_scores.values() if z is not None]
        score = max(finite) if finite else 0.0
        return score, fired

    def on_restart(self, workload: str):
        self.workloads[workload].on_restart()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "Detector":
        with open(path, "rb") as f:
            return pickle.load(f)
