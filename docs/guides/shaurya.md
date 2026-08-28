# Shaurya - Observability, Chaos and Evaluation

## Prodrome: a complete beginner's guide

You produce the data everyone else consumes, and the numbers that say whether any of it worked. This guide assumes you've never used Prometheus and explains monitoring from zero.

---

## Part 1 — Understanding monitoring

### 1.1 What a metric actually is

A **metric** is a number measured repeatedly over time. "Memory used by the redis container" sampled every 15 seconds gives you a **time series** — a sequence of (timestamp, value) pairs.

Everything in this project is time series. Your job is to produce good ones.

### 1.2 Prometheus pulls; it doesn't receive

Most beginners assume monitoring works by applications *sending* metrics somewhere. Prometheus works the opposite way.

Every monitored thing exposes an HTTP endpoint — usually `/metrics` — that returns its current numbers as plain text. Prometheus has a list of targets and **visits each one on a schedule** (every 15 seconds for us), reads the numbers, and stores them with a timestamp.

This is called the **pull model**, and it has a consequence that matters to you: if a container dies, Prometheus simply gets nothing on the next scrape. There's a gap in the data. Gaps are information, not errors.

### 1.3 We instrument nothing

Here's the part that makes this project practical: **Kubernetes already emits everything we need.**

**cAdvisor** is built into every Kubernetes node. It watches every container's cgroup — the kernel mechanism enforcing resource limits — and reports CPU, memory, network, and disk for all of them. No code changes, no libraries, no cooperation from the application.

**kube-state-metrics** is a small service that reports Kubernetes' own state: how many pods exist, how many times each has restarted, whether probes are passing.

Between them you get every metric this project uses, on any workload, with zero instrumentation. **This is also the product pitch** — point Prodrome at any deployment and it works.

### 1.4 Counters versus gauges — the one thing that trips everyone

This distinction will silently break half your metrics if you get it wrong.

A **gauge** is a current value that goes up and down. Memory in bytes. Pods running. You read it directly and it means something.

A **counter** only ever increases. It's cumulative since the process started. `container_cpu_usage_seconds_total` is the total CPU-seconds consumed since the container booted.

If you graph a counter raw, you get a line that climbs forever. It's meaningless. You need the **rate** of change:

```
rate(container_cpu_usage_seconds_total[1m])
```

This means: over a 1-minute window, how fast is this counter increasing? For CPU-seconds-per-second, the answer is **CPU cores in use**. That's the number you actually want.

**How to tell them apart:** if the metric name ends in `_total`, it's a counter and needs `rate()`. Nearly always true.

Get this wrong and Sagar's detector will behave bizarrely on Saturday and neither of you will know why.

### 1.5 Labels

A metric name isn't unique. `container_memory_working_set_bytes` exists for every container in the cluster. **Labels** distinguish them:

```
container_memory_working_set_bytes{namespace="prodrome", pod="redis-7d9f", container="redis"}
```

You filter with them:

```
container_memory_working_set_bytes{namespace="prodrome", container!=""}
```

`!=` means "not equal." `container!=""` excludes rows where the container label is empty — those are pod-level aggregate rows that would double-count.

> **Quirk that will confuse you:** network metrics are reported on the *pod*, not the container. So `container!=""` silently excludes them. If your network columns come back completely empty, this is why. Drop that filter for the two network queries only.

### 1.6 PromQL — the three things you need

You do not need to learn PromQL properly. You need three things.

**Select with filters:**
```
container_memory_working_set_bytes{namespace="prodrome"}
```

**Rate a counter:**
```
rate(container_cpu_usage_seconds_total{namespace="prodrome"}[1m])
```

**Divide two metrics** (they must share labels):
```
container_memory_working_set_bytes{namespace="prodrome",container!=""}
  / container_spec_memory_limit_bytes{namespace="prodrome",container!=""}
```

That last one gives **memory as a fraction of the limit**, which is the single most useful metric in the project. It's directly "how close is this to being killed," and unlike raw bytes it's comparable across workloads with different limits.

### 1.7 The two APIs

- `/api/v1/query` — value right now
- `/api/v1/query_range` — values across a time window, at a step interval

You will use `query_range` almost exclusively, for a reason worth understanding in §3.2.

---

## Part 2 — Setup

### 2.1 Install Prometheus

Wait until Shravan's cluster exists (or make your own — it's five minutes with the same config file, and then you're not waiting on anyone).

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.enabled=false \
  --set alertmanager.enabled=false \
  --set prometheus.prometheusSpec.retention=7d \
  --set prometheus.prometheusSpec.scrapeInterval=15s
```

One chart gives you Prometheus, cAdvisor scraping, and kube-state-metrics. Grafana and Alertmanager are disabled because we build our own view and don't need alerting.

**Retention of 7 days matters.** It's what lets you export data after the fact instead of capturing it live.

### 2.2 Reach it

Prometheus runs inside the cluster. Your browser is outside. `port-forward` bridges them:

```bash
kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090
```

Open `http://localhost:9090`, paste a query into the box, press Execute.

> **This connection dies silently** — a network blip, a laptop sleeping, a pod restart. Your overnight collection then quietly stops. Wrap it:
>
> ```bash
> while true; do kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090; sleep 2; done
> ```

### 2.3 Timebox this hard

**Give Prometheus 90 minutes. If it isn't working, fall back.**

```bash
docker stats --no-stream --format '{{json .}}'
```

That gives CPU and memory per container, which covers two of three fault types. Parse it into CSV in a loop and move on.

A working simple pipeline beats a broken sophisticated one, and the team needs data more than it needs Prometheus specifically.

---

## Part 3 — The eight metrics

### 3.1 Verify each one by hand

Paste each into the Prometheus UI. Confirm it returns three series (one per workload) before writing any code.

```
# 1 cpu_cores
rate(container_cpu_usage_seconds_total{namespace="prodrome",container!=""}[1m])

# 2 mem_bytes
container_memory_working_set_bytes{namespace="prodrome",container!=""}

# 3 mem_pct ← the most useful one
container_memory_working_set_bytes{namespace="prodrome",container!=""}
  / container_spec_memory_limit_bytes{namespace="prodrome",container!=""}

# 4 net_rx ← no container!= filter, see §1.5
rate(container_network_receive_bytes_total{namespace="prodrome"}[1m])

# 5 net_tx
rate(container_network_transmit_bytes_total{namespace="prodrome"}[1m])

# 6 fs_reads
rate(container_fs_reads_bytes_total{namespace="prodrome",container!=""}[1m])

# 7 fs_writes
rate(container_fs_writes_bytes_total{namespace="prodrome",container!=""}[1m])

# 8 restarts
kube_pod_container_status_restarts_total{namespace="prodrome"}
```

**Write this list into the README and treat it as frozen.** Changing it later means every model retrains.

### 3.2 The scraper

**Do not write a script that polls every 15 seconds and appends to a file.** If it dies at 3am you lose the night, and you won't find out until morning.

Prometheus is already storing everything with 7-day retention. **Let it be the database.** Your scraper exports a time range afterwards using `query_range`.

The logic:

1. For each of the eight queries, request a time range at a 15-second step
2. The response is nested JSON: a list of series, each with labels and a list of `[timestamp, value]` pairs
3. Flatten to long format: one row per (timestamp, workload, metric, value)
4. Pivot to wide format: one row per (timestamp, workload) with eight metric columns
5. Save as Parquet

**Parquet** is a columnar file format. Use it instead of CSV — it's smaller, keeps data types (so timestamps stay timestamps), and loads much faster. `pandas.to_parquet()` and `pandas.read_parquet()`; install `pyarrow`.

### 3.3 Derive stable workload names

Prometheus gives you pod names like `redis-7d9f8b-x2k1`. The suffix changes on every restart.

**This matters more than it sounds.** Sagar keys his detectors by workload name. If the name changes every restart, he silently gets a brand-new untrained detector each time, which behaves randomly, and neither of you will find it for hours.

Use the `container` label where available (it's just `redis`), or strip the two hash segments from the pod name. **Verify the output contains exactly `redis`, `nginx`, `postgres` before handing anything over.**

### 3.4 Sanity-check before you trust it

Run the scraper over the last ten minutes and check four things:

- [ ] Roughly 40 timestamps × 3 workloads ≈ 120 rows
- [ ] All eight metric columns present
- [ ] Missing values under a few percent, except possibly network columns
- [ ] Workload names are clean

```python
print(df.shape)
print(df.workload.unique())
print(df.isna().mean())
print(df.head())
```

Ten minutes here saves hours later.

---

## Part 4 — Load generation

### 4.1 Why this is not optional

You might think you could collect baseline data with no traffic. You cannot, and the reason is subtle.

Sagar's detector learns what normal looks like. **If load is constant, "normal" means "flat."** Then the first time real traffic rises — which it will, constantly — the detector sees deviation from flat and fires. Your false positive rate is catastrophic and nobody knows why.

**The variation is the training signal.** A detector trained on varying load learns that variation is normal, and only fires on the *shape* of degradation.

### 4.2 Compressed days

Real traffic has a daily rhythm: quiet overnight, ramp in the morning, dip midday, evening peak. You need many of those cycles, and you don't have many days.

**Compress a simulated day into ten minutes.** Twelve hours overnight gives you ~70 simulated days.

With **k6** (a scriptable load tool), a stage-based profile:

```javascript
export const options = {
  stages: [
    { duration: '2m', target: 5 },   // night
    { duration: '2m', target: 40 },  // morning ramp
    { duration: '2m', target: 25 },  // midday dip
    { duration: '2m', target: 60 },  // evening peak
    { duration: '2m', target: 5 },   // night again
  ],
};
```

`target` is concurrent virtual users; k6 ramps smoothly between stages.

```bash
kubectl port-forward -n prodrome svc/nginx 8080:80 &
k6 run --duration 14h collect/load/k6-diurnal.js &
```

For redis and postgres, shell loops with their own benchmark tools (`redis-benchmark`, `pgbench`) varying concurrency on the same rhythm.

**Run identical load against the control namespace.** Both arms need identical conditions or the comparison is meaningless.

---

## Part 5 — Chaos injection

### 5.1 Why this solves the hardest problem in ML

Every machine learning project dies on the same question: **where do the labels come from?**

Usually the answer is "a human spent three weeks annotating."

Ours doesn't need that. **The injector knows what it injected and when.** Start a memory leak on redis at 14:02:15, end it at 14:07:15 — every metric window in between is labeled `MEMORY_LEAK` by construction. Real degradation, in a real system, labeled for free.

You own that mechanism.

### 5.2 Use `kubectl exec`, not a chaos framework

There are dedicated chaos engineering tools. **Don't use them.** They mean a Helm install, several new custom resource types, and a new mental model — for exactly the same result you get from running a command inside a container.

```bash
kubectl exec -n prodrome deploy/redis -- \
  stress-ng --cpu 2 --timeout 300s
```

That's the whole mechanism. Wrap it in Python with `subprocess.run`.

### 5.3 The three faults

| Label | Command | What it does |
|---|---|---|
| CPU_HOG | `stress-ng --cpu N --timeout Xs` | N threads spinning |
| MEMORY_LEAK | `stress-ng --vm 1 --vm-bytes NM --vm-hang 0 --timeout Xs` | Allocates N megabytes |
| DISK_STRESS | `stress-ng --hdd 1 --hdd-bytes NM --timeout Xs` | Sustained disk writes |

Plus `POD_KILL` — just `kubectl delete pod`. Deliberately unpredictable; see §5.5.

### 5.4 Activation patterns — the part that makes lead time meaningful

**Run every fault twice: constant, and ramping.**

A **constant** fault jumps to full intensity instantly. That's a step function — trivially detectable, and a plain threshold would catch it. If all your faults are steps, your lead-time numbers measure "how fast can you notice a discontinuity," which is not interesting and not what real degradation looks like.

A **ramp** steps intensity up over the run — 80MB, then 160, then 240, then 320, then 420. That's a slide toward failure, which is what real leaks and real saturation look like, and it's the only kind of fault where "predicting" means anything.

`stress-ng` can't ramp on its own, so run successive invocations of increasing size.

**This is your call to make and nobody else will catch it if you skip it.**

### 5.5 What to run

```
3 workloads × 3 faults × 2 patterns = 18 runs, 5 minutes each with a recovery gap
+ 4 clean runs (no fault at all)
+ 2 pod kills
```

Roughly three hours. Start early.

**The clean runs are how you measure false positives, and they're the first thing teams skip.** Without them you cannot report precision. A detector that fires constantly will look perfect on recall and be completely useless — and you'd never know.

**The pod kills are the honest failure case.** They should produce roughly zero lead time. You report that on purpose. Showing where the method stops working is what separates a project from a sales pitch, and it's the first thing anyone technical will ask about.

### 5.6 The labels file

One row per run: start time, end time, workload, fault type, activation pattern, and a run identifier.

> **The run identifier is load-bearing.** Sadhil must split train and test *by run*. Consecutive metric windows overlap by 19 of 20 ticks, so a random row-level split puts near-identical rows on both sides and produces meaningless accuracy — he'd report 0.99 and it would mean nothing. **He physically cannot split correctly without this column.**

Make it something like `MEMORY_LEAK_ramp_redis_1730files`.

---

## Part 6 — Labeling

### 6.1 The mistake that breaks two people's work at once

**Label the entire fault trajectory, not just the dramatic part.**

Every 20-tick window from injection start to failure gets the fault label — **including the early boring ones where memory is at 62% and nothing looks wrong yet.**

Here's what happens if you don't. You label only the obvious windows, where memory is at 95% and everything is on fire. Sadhil's classifier trains on those and learns "a memory leak looks like memory at 95%." Fine.

But Sagar's detector fires at *minute one*, when memory is at 62%. The classifier receives a window it has never seen anything like, and returns garbage.

**The pipeline breaks at exactly the point that makes the project interesting**, and it breaks between two people's components, so neither of them can see it from their own side. **You are the only person positioned to prevent this.**

### 6.2 Also record seconds-to-failure

For each labeled window, record how many seconds remained until the failure occurred.

This is what lets Sadhil produce the accuracy-versus-lead-time curve — the headline result of the whole project. **Don't drop this column.**

---

## Part 7 — Evaluation

### 7.1 The concepts

**Lead time** — seconds between the detector first firing and the actual failure. The whole point of the product.

**Recall** — of the faults you injected, what fraction were detected before failure? Detected ÷ injected.

**Precision** — of the times it fired, what fraction were real faults? Real ÷ total firings. **You need the clean runs to compute this.**

They trade off. Fire constantly and recall is perfect and precision is terrible. Fire never and the reverse. This is why you report both.

**False positives per hour** — precision translated into something operational. "This tool would take three unnecessary actions per hour" is a sentence an engineer immediately understands.

**Time to recovery** — failure to healthy again, ours versus control.

### 7.2 Calibrate expectations now

The published work on this exact problem reports roughly 0.65 precision with 0.92 recall, and lead times from fifteen minutes to over two hours depending on fault type — on a cleaner testbed than yours.

**A third of predictions being false alarms is the state of the art.** If your team hits 0.7 precision, you've reproduced published results. Nobody should panic. And if someone reports 0.98, that's a signal to check for a methodological error, not to celebrate.

### 7.3 Report per fault type, always

| fault | runs | recall | median lead | recovery (ours / control) |
|---|---|---|---|---|
| CPU_HOG | 6 | ? | ? s | ? s / ? s |
| MEMORY_LEAK | 6 | ? | ? s | ? s / ? s |
| DISK_STRESS | 6 | ? | ? s | ? s / ? s |
| POD_KILL | 2 | ? | ~0 s | ? s / ? s |

**Never one aggregate number.** An average lead time hides that one fault type gets twenty minutes of warning and another gets none — and the cases where the method doesn't work are the most informative part of the result.

### 7.4 The dashboard — terminal, not web

Use `rich` (a Python terminal formatting library) for a live-updating table: workload, detector score, whether it fired, predicted class, confidence, action taken.

**Do not start a React app.** It will not finish and it will consume you for a full day. A clean terminal table demos perfectly well. Web UI is a later phase.

---

## Part 8 — Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Query returns nothing | Remove label filters one at a time until data appears, then add back |
| Values huge and always increasing | It's a counter — wrap in `rate(...[1m])` |
| Network columns all empty | Remove `container!=""` — network metrics live on the pod |
| Workload names are pod hashes | Fix name derivation before anyone trains on it |
| `stress-ng: not found` | Image lacks it, or wasn't `kind` loaded after rebuild |
| `kubectl exec` hangs | Add a timeout to `subprocess.run`; `stress-ng` sometimes outlives its own `--timeout` |
| Memory leak never kills anything | `--vm-bytes` is below the container limit. Raise it above the ceiling |
| Port-forward stopped overnight | It dies silently. Use the retry loop from §2.2 |
| Lots of NaN in the parquet | Some series didn't exist for part of the range. Forward-fill, and tell Sagar which columns |
| Prometheus disk full | Lower retention or `kind delete cluster` and start clean |

---

## Part 9 — Reading list

**Do read:** Prometheus "Querying basics" (20 min) — selectors and `rate()` only; the k6 "Test types" page (10 min); the pandas pivot/reshape guide (15 min).

**Don't read yet:** recording rules, alerting rules, Alertmanager, Thanos, federation, exporters you're not using.

---

## Definition of done

- [ ] All eight metrics verified by hand in the Prometheus UI
- [ ] Scraper produces valid Parquet with clean workload names
- [ ] Load generator running with genuine variation, on both namespaces
- [ ] Canonical healthy dataset collected overnight
- [ ] Chaos runner: 3 faults × 2 patterns, plus clean runs, plus pod kills
- [ ] Labels file with run identifiers and seconds-to-failure
- [ ] Full trajectories labeled, not just peaks
- [ ] One command prints the per-fault results table
- [ ] False positives per hour computed from the clean runs
- [ ] Live terminal dashboard for the demo
