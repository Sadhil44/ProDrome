"""The plot that sells the idea (Part 5.5).

For one ramping MEMORY_LEAK: memory usage, the detector's score, the
moment it fired, and the moment the failure actually occurred, all on
one time axis. Uses the chosen configuration from Part 5.2.1.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ml.replay import firing_events, train_and_replay

OUT_PATH = "ml/memory_leak_example.png"
CONTEXT = pd.Timedelta(minutes=3)  # padding shown before/after the fault window


def main():
    metrics = pd.read_parquet("data/samples/metrics.parquet")
    labels = pd.read_csv("data/samples/labels.csv", parse_dates=["start_ts", "end_ts"])

    ramps = labels[(labels["fault_type"] == "MEMORY_LEAK") & (labels["pattern"] == "ramp")]
    leak = ramps[ramps["workload"] == "redis"].iloc[0]  # the one run this config actually detects
    workload = leak["workload"]

    det, log = train_and_replay(metrics, labels)
    events = firing_events(log)
    workload_events = events[events["workload"] == workload]

    window_start = leak["start_ts"] - CONTEXT
    window_end = leak["end_ts"] + CONTEXT

    m = metrics[(metrics["workload"] == workload) & (metrics["ts"].between(window_start, window_end))]
    s = log[(log["workload"] == workload) & (log["ts"].between(window_start, window_end))]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(m["ts"], m["mem_pct"] * 100, color="tab:blue", label="mem_pct")
    ax1.set_ylabel("Memory (% of limit)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(s["ts"], s["score"], color="tab:orange", label="detector score")
    ax2.set_ylabel("Detector score (z)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.axvline(leak["end_ts"], color="black", linestyle="--", label="failure (OOMKill)")

    onset = workload_events[workload_events["onset_ts"].between(window_start, window_end)]
    if not onset.empty:
        ax1.axvline(onset.iloc[0]["onset_ts"], color="tab:red", linestyle=":", label="detector fired")

    fig.suptitle(f"{workload}: ramping MEMORY_LEAK, run {leak['run_id']}")
    fig.autofmt_xdate()

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")
    if not onset.empty:
        lead_s = (leak["end_ts"] - onset.iloc[0]["onset_ts"]).total_seconds()
        print(f"fired at {onset.iloc[0]['onset_ts']}, {lead_s:.0f}s before failure at {leak['end_ts']}")
    else:
        print("did not fire within the plotted window")


if __name__ == "__main__":
    main()
