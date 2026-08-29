"""EWMA baseline detector (Signal).

Tracks a per-workload, per-metric exponentially weighted moving average
and flags ticks that deviate too far from it. Trains only on whatever
data it's pointed at -- callers must only ever point this at healthy
data (see SETUP.md ground rule #1). Baseline stats are not updated on
anomalous ticks, so a sustained fault can't drag the "normal" baseline
up and blind the detector to itself.
"""

import pandas as pd

METRICS = ["cpu_cores", "mem_pct"]
ALPHA = 0.05      # EWMA smoothing factor: lower = slower-moving baseline
K = 3.5           # z-score threshold for "anomalous"
WARMUP = 20       # ticks used to seed level/var before scoring starts
DEBOUNCE = 2      # consecutive anomalous ticks required to set fired=1


def _score_workload(g: pd.DataFrame) -> pd.DataFrame:
    g = g.reset_index(drop=True)
    n = len(g)

    z_cols = {m: [float("nan")] * n for m in METRICS}
    fired = [0] * n
    streak = 0

    level = {}
    var = {}

    for m in METRICS:
        warm = g[m].iloc[:WARMUP]
        level[m] = warm.mean()
        var[m] = warm.var(ddof=0) or 1e-9  # guard against a zero-variance warmup

    for i in range(n):
        if i < WARMUP:
            continue

        any_anomalous = False
        for m in METRICS:
            x = g[m].iloc[i]
            z = (x - level[m]) / (var[m] ** 0.5)
            z_cols[m][i] = z

            if abs(z) > K:
                any_anomalous = True
            else:
                # Only fold typical points into the running baseline.
                level[m] = ALPHA * x + (1 - ALPHA) * level[m]
                var[m] = ALPHA * (x - level[m]) ** 2 + (1 - ALPHA) * var[m]

        streak = streak + 1 if any_anomalous else 0
        fired[i] = 1 if streak >= DEBOUNCE else 0

    out = g[["ts", "workload"]].copy()
    for m in METRICS:
        out[f"{m}_z"] = z_cols[m]
    out["fired"] = fired
    return out


def compute_ewma_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["workload", "ts"])
    parts = [_score_workload(g) for _, g in df.groupby("workload")]
    return pd.concat(parts, ignore_index=True)


def main():
    df = pd.read_parquet("data/samples/synthetic/metrics.parquet")
    scores = compute_ewma_scores(df)

    fire_rate = scores["fired"].mean()
    print(f"scored {len(scores)} rows across {df['workload'].nunique()} workloads")
    print(f"fired rate: {fire_rate:.4%}  (expect ~0% on healthy-only data)")
    print()
    print("per-workload fired counts:")
    print(scores.groupby("workload")["fired"].sum())


if __name__ == "__main__":
    main()
