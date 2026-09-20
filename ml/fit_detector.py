"""Fit the detector on healthy data and pickle it (Part 4.5).

Prefers the real healthy baseline (data/healthy/metrics.parquet) over
the bundled synthetic sample, same "files not services" preference
ml/classifier.py uses for chaos data (SETUP.md S8).

Run: python -m ml.fit_detector
"""

from pathlib import Path

import pandas as pd

from ml.detector import Detector
from ml.replay import fault_mask

DETECTOR_PATH = Path("ml/detector.pkl")
HEALTHY_METRICS = Path("data/healthy/metrics.parquet")
SAMPLE_METRICS = Path("data/samples/metrics.parquet")
SAMPLE_LABELS = Path("data/samples/labels.csv")


def load_healthy() -> tuple[pd.DataFrame, Path]:
    if HEALTHY_METRICS.exists():
        return pd.read_parquet(HEALTHY_METRICS), HEALTHY_METRICS

    # The bundled sample interleaves faults with healthy ticks in one file --
    # filter to healthy-only before fitting (ground rule #1).
    metrics = pd.read_parquet(SAMPLE_METRICS)
    labels = pd.read_csv(SAMPLE_LABELS, parse_dates=["start_ts", "end_ts"])
    return metrics[~fault_mask(metrics, labels)], SAMPLE_METRICS


def fit_and_save() -> Detector:
    healthy, source = load_healthy()
    det = Detector.fit_healthy(healthy)
    det.save(DETECTOR_PATH)

    print(f"fit on {source} ({len(healthy)} healthy ticks), saved to {DETECTOR_PATH}")
    for workload, wd in det.workloads.items():
        print(f"  {workload}: {wd.metrics}")
    return det


def _load_or_fit() -> Detector:
    if DETECTOR_PATH.exists():
        return Detector.load(DETECTOR_PATH)
    return fit_and_save()


detector = _load_or_fit()


if __name__ == "__main__":
    fit_and_save()
