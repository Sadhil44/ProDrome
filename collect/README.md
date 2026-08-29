# collect/

Metric scraper, chaos runner, load generation. Owned by Shaurya — see [SETUP.md](../SETUP.md) §6.

## Scripts

| Path | Part | What |
|---|---|---|
| `scrape.py` | 3.2 | Export a time window of the 8 metrics from Prometheus → Parquet. `--minutes N` or `--start/--end`. |
| `load/run-load.sh` | 4 | Diurnal load against all 6 workloads (prodrome + control). `DAYS=84` ≈ 14h. See `load/README.md`. |
| `chaos.py` | 5 | Inject CPU_HOG / MEMORY_LEAK / DISK_STRESS / POD_KILL via `kubectl exec … stress-ng`; write `data/chaos/labels.csv`. `--campaign` or `--one`. |

### The two collection runs

Both need the Prometheus port-forward loop (guide §2.2) up, and the load
generator running so metrics vary. **Before each run, reset the pods** so restart
counts start at 0 and there's no leftover memory pressure:

```bash
kubectl rollout restart deployment -n prodrome && kubectl rollout status deployment -n prodrome
kubectl rollout restart deployment -n control  && kubectl rollout status deployment -n control
```

1. **Healthy** (`data/healthy/metrics.parquet` → Sagar). Load only, no chaos.
   Run `run-load.sh` overnight, then `scrape.py --start <after-reset> --end <now>`.
2. **Chaos** (`data/chaos/{metrics.parquet,labels.csv}` → Sadhil). Load + chaos.
   Keep load running, `python collect/chaos.py --campaign` (~2.5h), then run the
   `scrape.py` line it prints.

MEMORY_LEAK note: on this cluster (cgroup v2) the OOM killer usually reaps
stress-ng rather than the container, so `restarts` fires inconsistently for
MEMORY_LEAK runs — the rising `mem_pct` trajectory is the reliable signal, and
that's what the models key on.

---

## The eight metrics — FROZEN

This list is the contract for the scraper's output columns. Per
`docs/guides/shaurya.md` §3.1 it is frozen: changing it means every model retrains.
Each query below was verified against the live cluster (k8s v1.37, kube-prometheus-stack)
and returns exactly **3 series** for `namespace="prodrome"` — one per workload.

```promql
# 1  cpu_cores   — CPU cores in use
rate(container_cpu_usage_seconds_total{namespace="prodrome",container!=""}[1m])

# 2  mem_bytes   — working set in bytes
container_memory_working_set_bytes{namespace="prodrome",container!=""}

# 3  mem_pct     — memory as a fraction of the limit  (the most useful one)
container_memory_working_set_bytes{namespace="prodrome",container!=""}
  / on(namespace,pod,container)
kube_pod_container_resource_limits{namespace="prodrome",resource="memory"}

# 4  net_rx      — receive bytes/sec, summed across interfaces
sum by (namespace,pod) (
  rate(container_network_receive_bytes_total{namespace="prodrome"}[1m])
)

# 5  net_tx      — transmit bytes/sec, summed across interfaces
sum by (namespace,pod) (
  rate(container_network_transmit_bytes_total{namespace="prodrome"}[1m])
)

# 6  fs_reads    — disk read bytes/sec
rate(container_fs_reads_bytes_total{namespace="prodrome",container!=""}[1m])

# 7  fs_writes   — disk write bytes/sec
rate(container_fs_writes_bytes_total{namespace="prodrome",container!=""}[1m])

# 8  restarts    — cumulative container restarts
kube_pod_container_status_restarts_total{namespace="prodrome"}
```

Stable workload name (`redis` / `nginx` / `postgres`, per §3.3): take it from the
`container` label, which is already clean. Queries 4 and 5 drop `container` in the
`sum`, so for those two derive the name from `pod` by stripping the two hash
segments (`redis-65c4779958-zgprh` → `redis`).

---

## NOTE — these queries deviate from `docs/guides/shaurya.md` §3.1

The guide predates a live cluster. Two queries there do not work as written on
k8s v1.37 and were adjusted here. **This README is authoritative; the guide is not.**

| # | Guide §3.1 | Here | Why |
|---|---|---|---|
| 3 `mem_pct` | divides by `container_spec_memory_limit_bytes` | divides by `kube_pod_container_resource_limits{resource="memory"}` with an explicit `on(...)` join | `container_spec_memory_limit_bytes` is **not emitted** by this cluster's cAdvisor (dropped under cgroup v2 / recent kubelet). The guide's query returns zero series. The limit now comes from kube-state-metrics instead; the join is needed because the two metrics carry different label sets. |
| 4, 5 `net_rx` / `net_tx` | bare `rate(...)` | wrapped in `sum by (namespace,pod)` | Network counters are per **interface**, so the bare query returns 9 series per pod (27 total), not 1. Summing collapses to one value per workload, which is what the guide's own §1.5 note intends. |

Queries 1, 2, 6, 7, 8 are unchanged from the guide.

**Downstream is unaffected.** Sagar and Sadhil consume the output Parquet
(`ts, workload, cpu_cores, mem_bytes, mem_pct, net_rx, net_tx, fs_reads, fs_writes, restarts`
— SETUP.md §7). Column names and meanings are identical; only the PromQL that
produces them changed. `mem_pct` is still "working set ÷ memory limit", 0–1.

Platform pair: if you write your own memory-headroom queries for the controller,
use `kube_pod_container_resource_limits`, not `container_spec_memory_limit_bytes`.
