"""Generate the healthy-only sample metrics fixture.

Deterministic (fixed seed) so the fixture is reproducible from a fresh
clone. Produces stationary, noisy-but-normal behavior only -- no step
functions, no injected faults. Detector training must never see faults
mixed into "healthy" data (see SETUP.md ground rule #1).

Run: python data/samples/synthetic/generate.py
"""

import numpy as np
import pandas as pd

SEED = 0
TICK_SECONDS = 15
N_TICKS = 500  # ~2 hours per workload

# Per-workload baselines: (cpu_cores, mem_bytes, net_rx, net_tx, fs_reads, fs_writes)
# Chosen to be plausible for each service's typical profile, not measured.
WORKLOAD_PROFILES = {
    "redis": dict(cpu=0.15, mem=512 * 1024 * 1024, mem_limit=1024 * 1024 * 1024,
                  net_rx=50_000, net_tx=80_000, fs_reads=5, fs_writes=20),
    "nginx": dict(cpu=0.25, mem=128 * 1024 * 1024, mem_limit=256 * 1024 * 1024,
                  net_rx=200_000, net_tx=180_000, fs_reads=10, fs_writes=2),
    "postgres": dict(cpu=0.40, mem=1024 * 1024 * 1024, mem_limit=2048 * 1024 * 1024,
                  net_rx=30_000, net_tx=25_000, fs_reads=150, fs_writes=90),
}


def gen_workload(rng: np.random.Generator, name: str, profile: dict, start_ts: pd.Timestamp) -> pd.DataFrame:
    ts = pd.date_range(start=start_ts, periods=N_TICKS, freq=f"{TICK_SECONDS}s")

    cpu_cores = np.clip(rng.normal(profile["cpu"], profile["cpu"] * 0.08, N_TICKS), 0, None)
    mem_bytes = np.clip(rng.normal(profile["mem"], profile["mem"] * 0.05, N_TICKS), 0, None).astype(np.int64)
    mem_pct = mem_bytes / profile["mem_limit"]
    net_rx = np.clip(rng.normal(profile["net_rx"], profile["net_rx"] * 0.15, N_TICKS), 0, None).astype(np.int64)
    net_tx = np.clip(rng.normal(profile["net_tx"], profile["net_tx"] * 0.15, N_TICKS), 0, None).astype(np.int64)
    fs_reads = rng.poisson(profile["fs_reads"], N_TICKS).astype(np.int64)
    fs_writes = rng.poisson(profile["fs_writes"], N_TICKS).astype(np.int64)

    # Restarts: 0 almost always, rare organic blip (not fault-correlated).
    restarts = np.zeros(N_TICKS, dtype=np.int64)
    blip_idx = rng.integers(0, N_TICKS, size=max(1, N_TICKS // 250))
    restarts[blip_idx] = 1

    return pd.DataFrame({
        "ts": ts,
        "workload": name,
        "cpu_cores": cpu_cores.round(4),
        "mem_bytes": mem_bytes,
        "mem_pct": mem_pct.round(4),
        "net_rx": net_rx,
        "net_tx": net_tx,
        "fs_reads": fs_reads,
        "fs_writes": fs_writes,
        "restarts": restarts,
    })


def main():
    rng = np.random.default_rng(SEED)
    start_ts = pd.Timestamp("2026-01-01T00:00:00Z")

    frames = [gen_workload(rng, name, profile, start_ts) for name, profile in WORKLOAD_PROFILES.items()]
    df = pd.concat(frames, ignore_index=True).sort_values(["ts", "workload"]).reset_index(drop=True)

    out_path = "data/samples/synthetic/metrics.parquet"
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {df.shape}")


if __name__ == "__main__":
    main()
