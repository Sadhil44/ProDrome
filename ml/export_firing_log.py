"""Export the detector's real-data firing log for Sadhil (Part 7).

"His firing timestamps... he replays the detector over stored data and
gives you a file of firings; you bucket your accuracy against them."
This is that file: the tuned detector (ml.fit_detector's shipped
config), replayed against real data/chaos/metrics.parquet, fit on real
data/healthy/metrics.parquet -- not the synthetic sample.

Columns: ts, workload, score, fired. Join on (workload, ts) against a
window's reference tick (ml.dataset's ref_ts = window["ts"].iloc[-1])
to get "had the detector already fired as of this window."

Run: python -m ml.export_firing_log
"""

from pathlib import Path

import pandas as pd

from ml.fit_detector import detector
from ml.replay import firing_events, replay

CHAOS_METRICS = Path("data/chaos/metrics.parquet")
CHAOS_LABELS = Path("data/chaos/labels.csv")
OUT_LOG = Path("data/chaos/firing_log.csv")
OUT_EVENTS = Path("data/chaos/firing_events.csv")


def main():
    chaos_metrics = pd.read_parquet(CHAOS_METRICS)
    chaos_labels = pd.read_csv(CHAOS_LABELS, parse_dates=["start_ts", "end_ts"])

    log = replay(detector, chaos_metrics, chaos_labels)
    log.to_csv(OUT_LOG, index=False)

    events = firing_events(log)
    events.to_csv(OUT_EVENTS, index=False)

    print(f"wrote {OUT_LOG}: {len(log)} ticks, {log['fired'].sum()} fired")
    print(f"wrote {OUT_EVENTS}: {len(events)} discrete firing events")


if __name__ == "__main__":
    main()
