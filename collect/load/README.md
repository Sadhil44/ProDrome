# collect/load/

Diurnal load generation — Part 4 of `docs/guides/shaurya.md`.

Why it exists: a detector trained on **constant** load learns that "normal" is
flat, then fires on the first real traffic swing (guide 4.1). The variation is
the training signal. So healthy data must be collected *with* load running, and
the load must vary on a realistic rhythm.

## The rhythm

One simulated day is compressed into **10 minutes**: quiet night → morning ramp
→ midday dip → evening peak → night. Twelve hours of wall-clock ≈ 70 days.
All three workloads follow the same shape (nginx smoothly via k6, redis/postgres
as a coarse staircase).

| File | Role |
|---|---|
| `k6-diurnal.js` | HTTP load for one nginx service. `TARGET` + `DAYS` env vars. |
| `rhythm.sh` | `redis-benchmark` / `pgbench` load for one workload in one namespace, via `kubectl exec`. Runs until killed. |
| `run-load.sh` | Orchestrator. Launches all of the above for redis/nginx/postgres × `prodrome` + `control`. Self-terminates when the k6 runs finish. |

## Run it

```bash
# smoke test (~30 min)
DAYS=3 bash collect/load/run-load.sh

# the real overnight run (~14h)
DAYS=84 bash collect/load/run-load.sh
```

Keep the Mac awake for the whole run — `caffeinate -i -w $$` in another tab.
The Prometheus port-forward from guide 2.2 must also be up (the scraper needs it,
not the load generator).

When the run is done, export the window:

```bash
python collect/scrape.py --start <run-start-ISO> --end <run-end-ISO> \
    --out data/healthy/metrics.parquet
```

## Caveats: in-container generators

`rhythm.sh` runs `redis-benchmark` / `pgbench` *inside* the workload container:

- A small slice of that container's measured CPU is generator overhead rather
  than served work. Acceptable for a synthetic testbed, same tradeoff the
  stress-ng chaos injector makes (guide 5.2).
- Client↔server traffic is loopback inside the container's network namespace, so
  it never crosses the pod's `eth0`. **`net_rx` / `net_tx` for redis and postgres
  stay near zero even under load.** That's fine — no fault type is
  network-based (CPU_HOG / MEMORY_LEAK / DISK_STRESS / POD_KILL) — but it is a
  fidelity gap versus the synthetic data, which gave those columns non-zero
  values. nginx is unaffected: k6 runs on the host, so its traffic is real pod
  network I/O.
