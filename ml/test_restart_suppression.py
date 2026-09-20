"""Verifies Part 4.4's post-restart guard actually does something.

A pod restart looks like a severe anomaly to the detector (memory drops
to near-zero, CPU spikes during startup) -- without suppression this
fires, the controller "fixes" it by restarting again, and that's the
infinite loop Part 4.4 warns about. This constructs one synthetic
restart on top of real healthy data and checks both sides: it WOULD
fire without on_restart(), and does NOT fire for the suppression
window when on_restart() is called at the right moment.
"""

import numpy as np
import pandas as pd

from ml.detector import POST_RESTART_SUPPRESS_TICKS, Detector

WORKLOAD = "nginx"
RESTART_TICKS = 10  # startup burst: memory near-zero, CPU spiking


def make_restart_sequence(baseline_row, n=RESTART_TICKS):
    """n ticks shaped like a pod restart: memory collapses, CPU spikes."""
    rows = []
    for i in range(n):
        row = dict(baseline_row)
        row["mem_bytes"] = baseline_row["mem_bytes"] * 0.05
        row["mem_pct"] = baseline_row["mem_pct"] * 0.05
        row["cpu_cores"] = baseline_row["cpu_cores"] * 4.0
        rows.append(row)
    return rows


def main():
    healthy = pd.read_parquet("data/healthy/metrics.parquet")
    det = Detector.fit_healthy(healthy)
    wd_metrics = det.workloads[WORKLOAD].metrics

    baseline_row = healthy[healthy["workload"] == WORKLOAD].iloc[-1]
    restart_rows = make_restart_sequence(baseline_row)

    # Scenario A: no on_restart() called -- does it fire on the restart shape?
    det_a = Detector.fit_healthy(healthy)
    wd_a = det_a.workloads[WORKLOAD]
    fired_a = [wd_a.update({m: row[m] for m in wd_metrics})[1] for row in restart_rows]
    print("without on_restart():", fired_a)
    print("  -> fired at least once:", any(fired_a))

    # Scenario B: on_restart() called right as the restart begins.
    det_b = Detector.fit_healthy(healthy)
    wd_b = det_b.workloads[WORKLOAD]
    wd_b.on_restart()
    fired_b = [wd_b.update({m: row[m] for m in wd_metrics})[1] for row in restart_rows]
    print()
    print(f"with on_restart() (suppression window = {POST_RESTART_SUPPRESS_TICKS} ticks):")
    print("  fired during the restart itself:", fired_b)
    print("  -> suppressed throughout:", not any(fired_b))

    # Suppression should also hold for ticks after the restart sequence,
    # up to POST_RESTART_SUPPRESS_TICKS total, then lift.
    remaining = POST_RESTART_SUPPRESS_TICKS - RESTART_TICKS
    post_restart_fired = [
        wd_b.update({m: baseline_row[m] for m in wd_metrics})[1] for _ in range(remaining)
    ]
    print(f"  fired in the next {remaining} (still-suppressed, back-to-normal) ticks:", post_restart_fired)
    print("  -> still suppressed:", not any(post_restart_fired))
    print("  suppress_ticks_left after that:", wd_b.suppress_ticks_left)


if __name__ == "__main__":
    main()
