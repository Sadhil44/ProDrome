# data/samples/

Small committed fixtures so everyone can run everything without waiting on another
workstream's output — see [SETUP.md](../../SETUP.md) §3 and §8.

This directory is the one exception to the `data/*` gitignore rule.

## Contents

| File | Schema | Rows |
|---|---|---|
| `metrics.parquet` | `ts, workload, cpu_cores, mem_bytes, mem_pct, net_rx, net_tx, fs_reads, fs_writes, restarts` (SETUP.md §7) | ~1.1k — 3 workloads × ~380 ticks |
| `labels.csv` | `start_ts, end_ts, workload, fault_type, pattern, run_id` (SETUP.md §7) | 18 — one run of each (workload × fault × pattern) combo |

Both are a slice of the larger synthetic set written to `data/synthetic/` (gitignored).
Regenerate everything with:

```bash
python data/samples/synthetic/generate.py
```

## Notes for consumers

- Read the labels CSV with `pd.read_csv("labels.csv", parse_dates=["start_ts", "end_ts"])`
  or timestamp comparisons against the metrics table will fail.
- **Healthy-only data** (detector training, SETUP.md ground rule #1): drop every metrics
  row that falls inside a `labels` row for its workload. What's left is healthy by construction.
- `fault_type` is one of `CPU_HOG`, `MEMORY_LEAK`, `DISK_STRESS`; `pattern` is `constant` or `ramp`.
- This is Phase 0 synthetic data — realistic in *shape*, not in absolute numbers. Phase 2
  replaces it with a real Prometheus export at the same schema.
