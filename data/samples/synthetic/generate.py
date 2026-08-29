"""Synthetic metrics + labels generator -- Prodrome Phase 0.

This is the Phase 0 deliverable from `docs/guides/shaurya.md` Part 0: the
artifact that unblocks Signal (Sagar) and Diagnosis (Sadhil) before any
cluster exists. It emits the two data contracts from SETUP.md section 7 and
nothing else, because nothing downstream reads anything else:

  data/synthetic/metrics.parquet   ts, workload, cpu_cores, mem_bytes, mem_pct,
                                   net_rx, net_tx, fs_reads, fs_writes, restarts
                                   -- one row per workload per 15s tick
  data/synthetic/labels.csv        start_ts, end_ts, workload, fault_type,
                                   pattern, run_id  -- one row per injected fault

A small slice of each is also written to data/samples/ (committed) -- that
doubles as the fixture SETUP.md section 3/4 promises everyone.

The two properties that matter (realism is not the point -- shape is):

  1. Healthy data VARIES. A compressed diurnal cycle (one "day" = 10 min,
     guide Part 4.2) plus gaussian noise. If healthy data were flat, Sagar's
     detector would learn "normal == flat" and fire on the first real traffic
     change. Catch that here, not on real data.
  2. Faults come in `constant` AND `ramp` patterns (guide Part 5.4). A ramp
     is a staircase toward failure -- the only kind of fault where predicting
     means anything. If every fault were a step, Sadhil's model would never
     see a slide.

Deterministic (fixed seed): reproducible from a fresh clone.

Known simplifications -- this file is disposable, Phase 2 real data replaces it:
  * `end_ts` is the OOMKill instant for MEMORY_LEAK. CPU_HOG and DISK_STRESS
    don't kill anything, so for them it's just the end of injection, and
    seconds-to-failure derived from it is really seconds-to-injection-end.
  * `restarts` only ever moves on a MEMORY_LEAK breach.
  * Downstream gets "healthy only" (SETUP.md ground rule 1) by dropping rows
    that fall inside any labels row -- no schema change needed.

Read the labels CSV back with parse_dates so the timestamps compare correctly:
    pd.read_csv("data/synthetic/labels.csv", parse_dates=["start_ts", "end_ts"])

Run from the repo root:  python data/samples/synthetic/generate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
TICK = pd.Timedelta(seconds=15)
START = pd.Timestamp("2026-01-01T00:00:00Z")

DAY_TICKS = 40          # a simulated day compressed into 10 minutes (guide Part 4.2)
DIURNAL_AMP = 0.25      # peak load is +/-25% around the daily mean

LEAD_CLEAN = 60         # healthy ticks before the first fault -- detector needs a clean baseline
RUN_TICKS = 20          # a fault episode is 5 minutes (20 x 15s), same as the real chaos plan
GAP_TICKS = 36          # ~9 minutes of recovery between episodes
TRAIL_CLEAN = 40
RUNS_PER_COMBO = 6      # 6 runs of each (workload x fault x pattern) -> Sadhil can split by run

FAULT_TYPES = ["CPU_HOG", "MEMORY_LEAK", "DISK_STRESS"]
PATTERNS = ["constant", "ramp"]
COMBOS = [(f, p) for f in FAULT_TYPES for p in PATTERNS]
RUNS_PER_WORKLOAD = len(COMBOS) * RUNS_PER_COMBO
N_TICKS = LEAD_CLEAN + RUNS_PER_WORKLOAD * (RUN_TICKS + GAP_TICKS) + TRAIL_CLEAN

# Committed fixture = the lead-in + exactly one round of every combo, cut in the
# recovery gap before round 2 so no fault run straddles the boundary.
SAMPLE_TICKS = LEAD_CLEAN + len(COMBOS) * (RUN_TICKS + GAP_TICKS) - GAP_TICKS // 2

METRIC_COLUMNS = ["ts", "workload", "cpu_cores", "mem_bytes", "mem_pct",
                  "net_rx", "net_tx", "fs_reads", "fs_writes", "restarts"]
LABEL_COLUMNS = ["start_ts", "end_ts", "workload", "fault_type", "pattern", "run_id"]

# Per-workload baselines. Plausible for each service's typical profile, not measured.
# mem_limit is the OOMKill ceiling -- what makes a MEMORY_LEAK a failure rather than a graph.
WORKLOAD_PROFILES = {
    "redis": dict(cpu=0.15, mem=512 * 1024 * 1024, mem_limit=1024 * 1024 * 1024,
                  net_rx=50_000, net_tx=80_000, fs_reads=5, fs_writes=20),
    "nginx": dict(cpu=0.25, mem=128 * 1024 * 1024, mem_limit=256 * 1024 * 1024,
                  net_rx=200_000, net_tx=180_000, fs_reads=10, fs_writes=2),
    "postgres": dict(cpu=0.40, mem=1024 * 1024 * 1024, mem_limit=2048 * 1024 * 1024,
                     net_rx=30_000, net_tx=25_000, fs_reads=150, fs_writes=90),
}


def _diurnal(t: np.ndarray) -> np.ndarray:
    """Load multiplier over tick indices: starts at the nightly trough, ~1 +/- DIURNAL_AMP."""
    return 1.0 + DIURNAL_AMP * np.sin(2 * np.pi * t / DAY_TICKS - np.pi / 2)


def _healthy(rng: np.random.Generator, prof: dict, n: int) -> dict:
    """A full healthy timeline per metric: baseline * diurnal cycle * gaussian noise."""
    t = np.arange(n)
    d = _diurnal(t)
    return dict(
        cpu_cores=np.clip(prof["cpu"] * d * (1 + rng.normal(0, 0.08, n)), 1e-4, None),
        # memory tracks load only weakly
        mem_bytes=np.clip(prof["mem"] * (1 + 0.10 * (d - 1)) * (1 + rng.normal(0, 0.03, n)), 1.0, None),
        net_rx=np.clip(prof["net_rx"] * d * (1 + rng.normal(0, 0.15, n)), 0, None),
        net_tx=np.clip(prof["net_tx"] * d * (1 + rng.normal(0, 0.15, n)), 0, None),
        fs_reads=rng.poisson(np.clip(prof["fs_reads"] * d, 0.1, None)).astype(float),
        fs_writes=rng.poisson(np.clip(prof["fs_writes"] * d, 0.1, None)).astype(float),
        restarts=np.zeros(n),
    )


def _intensity(pattern: str, w: int) -> np.ndarray:
    """Fault intensity over the window, in [0.2, 1.0].

    constant -> full intensity from tick 0 (a step).
    ramp     -> five stairs of increasing intensity (a slide toward failure).
    """
    if pattern == "constant":
        return np.ones(w)
    stair = np.floor(np.linspace(0, 5, w, endpoint=False)) / 4.0  # 0, .25, .5, .75, 1.0
    return 0.2 + 0.8 * stair


def _apply_fault(arr: dict, prof: dict, s: int, fault: str, pattern: str,
                 rng: np.random.Generator) -> tuple[int, int]:
    """Overwrite ticks [s, s+RUN_TICKS) with a fault trajectory. Returns (start_idx, end_idx)."""
    w = RUN_TICKS
    idx = np.arange(s, s + w)
    g = _intensity(pattern, w)
    end_idx = s + w - 1

    if fault == "CPU_HOG":
        # spinning threads: several extra cores on top of baseline
        arr["cpu_cores"][idx] += 2.5 * g * (1 + rng.normal(0, 0.05, w))

    elif fault == "MEMORY_LEAK":
        limit = prof["mem_limit"]
        # Both patterns fill memory over minutes, then OOMKill -- never an instant breach,
        # so there is a full trajectory to label (guide Part 6.1). constant == a steep
        # linear climb (allocation at full rate from t0); ramp == a staircase.
        if pattern == "constant":
            frac = np.linspace(0.60, 1.12, w)
        else:
            frac = 0.60 + 0.50 * _intensity("ramp", w)     # 0.60 .. 1.10 in five stairs
        leaked = np.maximum(arr["mem_bytes"][idx], frac * limit) * (1 + rng.normal(0, 0.02, w))
        arr["mem_bytes"][idx] = leaked
        breached = np.where(arr["mem_bytes"][idx] >= limit)[0]
        if breached.size:
            first = int(idx[breached[0]])
            end_idx = first                    # failure instant
            arr["restarts"][first] = 1         # OOMKill -- exactly on end_idx, so a
            #                                    [start_ts, end_ts] exclusion catches it
            #                                    and no restart tick leaks into "healthy"

    elif fault == "DISK_STRESS":
        arr["fs_writes"][idx] += 5000 * g * (1 + rng.normal(0, 0.08, w))
        arr["fs_reads"][idx] += 800 * g * (1 + rng.normal(0, 0.08, w))
        arr["cpu_cores"][idx] += 0.3 * g                                      # iowait

    return s, end_idx


def gen_workload(rng: np.random.Generator, name: str, prof: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = N_TICKS
    arr = _healthy(rng, prof, n)

    # Schedule: RUNS_PER_COMBO rounds, each round one run of every combo in shuffled order,
    # so fault types interleave instead of sitting in blocks.
    order = []
    for _ in range(RUNS_PER_COMBO):
        order.extend(COMBOS[k] for k in rng.permutation(len(COMBOS)))

    labels = []
    seen = {c: 0 for c in COMBOS}
    cursor = LEAD_CLEAN
    for fault, pattern in order:
        start_idx, end_idx = _apply_fault(arr, prof, cursor, fault, pattern, rng)
        i = seen[(fault, pattern)]
        seen[(fault, pattern)] += 1
        labels.append(dict(
            start_ts=START + start_idx * TICK,
            end_ts=START + end_idx * TICK,
            workload=name,
            fault_type=fault,
            pattern=pattern,
            run_id=f"{fault}_{pattern}_{name}_{i:03d}",
        ))
        cursor += RUN_TICKS + GAP_TICKS

    mem_bytes = np.clip(arr["mem_bytes"], 1, None).astype(np.int64)
    metrics = pd.DataFrame({
        "ts": pd.date_range(START, periods=n, freq="15s"),
        "workload": name,
        "cpu_cores": np.clip(arr["cpu_cores"], 0, None).round(4),
        "mem_bytes": mem_bytes,
        "mem_pct": (mem_bytes / prof["mem_limit"]).round(4),
        "net_rx": np.clip(arr["net_rx"], 0, None).astype(np.int64),
        "net_tx": np.clip(arr["net_tx"], 0, None).astype(np.int64),
        "fs_reads": np.clip(arr["fs_reads"], 0, None).astype(np.int64),
        "fs_writes": np.clip(arr["fs_writes"], 0, None).astype(np.int64),
        "restarts": np.clip(arr["restarts"], 0, None).astype(np.int64),
    })
    return metrics, pd.DataFrame(labels, columns=LABEL_COLUMNS)


def _sanity_check(metrics: pd.DataFrame, labels: pd.DataFrame) -> None:
    """The guide Part 3.4 checks, run before anything trusts the output."""
    assert list(metrics.columns) == METRIC_COLUMNS, metrics.columns.tolist()
    assert list(labels.columns) == LABEL_COLUMNS, labels.columns.tolist()
    assert set(metrics["workload"]) == set(WORKLOAD_PROFILES), set(metrics["workload"])
    assert not metrics.drop(columns=["ts"]).isna().any().any(), "NaN in metrics"
    assert metrics["mem_pct"].max() >= 1.0, "no MEMORY_LEAK ever breaches the limit"
    assert (metrics["restarts"] > 0).any(), "restarts never fires"

    per_combo = labels.groupby(["workload", "fault_type", "pattern"]).size()
    assert (per_combo == RUNS_PER_COMBO).all(), per_combo.to_dict()
    assert labels["run_id"].is_unique, "run_id collisions"
    assert labels["start_ts"].min() >= metrics["ts"].min()
    assert labels["end_ts"].max() <= metrics["ts"].max()
    assert (labels["start_ts"] <= labels["end_ts"]).all()

    print(f"  workloads      {sorted(metrics['workload'].unique())}")
    print(f"  metric rows    {len(metrics)}  ({N_TICKS} ticks x {len(WORKLOAD_PROFILES)} workloads)")
    print(f"  fault runs     {len(labels)}  ({RUNS_PER_COMBO}/combo x {len(WORKLOAD_PROFILES)} x {len(COMBOS)} combos)")
    print(f"  span           {metrics['ts'].min()} .. {metrics['ts'].max()}")
    print(f"  mem_pct        {metrics['mem_pct'].min():.3f} .. {metrics['mem_pct'].max():.3f}")
    print(f"  cpu_cores      {metrics['cpu_cores'].min():.3f} .. {metrics['cpu_cores'].max():.3f}")
    faulty = 0
    for _, r in labels.iterrows():
        m = metrics[(metrics.workload == r.workload) & (metrics.ts >= r.start_ts) & (metrics.ts <= r.end_ts)]
        faulty += len(m)
    print(f"  faulty ticks   {faulty} / {len(metrics)}  ({faulty / len(metrics):.1%})")


def _write(metrics: pd.DataFrame, labels: pd.DataFrame, mpath: str, lpath: str, tag: str) -> None:
    Path(mpath).parent.mkdir(parents=True, exist_ok=True)
    Path(lpath).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(mpath, index=False)
    labels.to_csv(lpath, index=False)
    print(f"{tag:9} {mpath}  {metrics.shape}")
    print(f"{tag:9} {lpath}  {labels.shape}")


def main() -> None:
    rng = np.random.default_rng(SEED)

    mframes, lframes = [], []
    for name, prof in WORKLOAD_PROFILES.items():
        m, l = gen_workload(rng, name, prof)
        mframes.append(m)
        lframes.append(l)

    metrics = pd.concat(mframes, ignore_index=True).sort_values(["ts", "workload"]).reset_index(drop=True)
    labels = pd.concat(lframes, ignore_index=True).sort_values(["start_ts", "workload"]).reset_index(drop=True)

    _sanity_check(metrics, labels)

    _write(metrics, labels,
           "data/synthetic/metrics.parquet", "data/synthetic/labels.csv", "full")

    cutoff = START + SAMPLE_TICKS * TICK
    _write(metrics[metrics["ts"] < cutoff].reset_index(drop=True),
           labels[labels["start_ts"] < cutoff].reset_index(drop=True),
           "data/samples/metrics.parquet", "data/samples/labels.csv", "sample")


if __name__ == "__main__":
    main()
