# infra/

Cluster config, workload manifests, Dockerfiles, and the monitoring stack. Owned by the platform
pair (Shravan/Shaurya) — see [SETUP.md](../SETUP.md) §6.

| File | What it is |
|---|---|
| `kind-config.yaml` | The cluster: one control-plane, two workers. |
| `workloads.yaml` | The Prodrome arm's workloads (`redis`, `nginx`, `postgres`). |
| `workloads-control.yaml` | The control arm. **Byte-identical to `workloads.yaml` apart from the namespace** — that parity is what makes the two-arm comparison valid. A change to one is a change to both. |
| `images/` | The stress-tool images. `kind load docker-image` after every rebuild. |
| `monitoring/` | The Prometheus stack. See below. |

```bash
kind create cluster --config infra/kind-config.yaml
kubectl apply -f infra/workloads.yaml
kubectl apply -f infra/workloads-control.yaml
bash infra/monitoring/install.sh
```

---

## Monitoring: which mechanism is current

**Prometheus is the current collection mechanism, and `collect/scrape.py` is the current
collector.** `infra/monitoring/` installs it, and a fresh clone plus this directory is meant to
reproduce the environment the data was collected in.

**`collect/baseline.py`'s cAdvisor path is the supported fallback, not a workaround.** It polls
`/api/v1/nodes/<node>/proxy/metrics/cadvisor` through the API-server proxy and reconstructs by hand
the three things Prometheus does for us — `rate(...[1m])`, the `mem_pct` limit join, and
`changes(restarts[1m])`. It needs no component in the cluster at all, which is exactly why it is
kept: it is the only thing that still works when the monitoring stack is absent, and on a cluster
this repo calls disposable, "absent" is a state that will happen again.

### Why this was a question at all

On 2026-10-12 Prometheus was **gone**: the `monitoring` namespace was empty, `helm list -A` returned
nothing, kube-state-metrics was absent. `collect/scrape.py` had nothing to query, the healthy
baseline had to be polled live by `collect/baseline.py`, and there was no retention window to
recover the lost range from.

The root cause is not that someone deleted a namespace. It is that the stack existed **only as a
helm command in `docs/guides/shaurya.md` §2.1** and never in `infra/`. `infra/CLAUDE.md` already
said why that is not allowed: *nothing that only exists as a `kubectl` command someone ran once is
reproducible from a fresh clone, and if it is not reproducible it is not a result.* The workloads
were committed; the thing that measured them was not. That asymmetry is now fixed regardless of
which collector anyone prefers.

### Why restore rather than declare it gone

The deciding fact is **which dataset would have to be re-collected**.

`data/chaos/` — 160 fault runs, and every recall, lead-time and precision number in the project —
was collected **with** Prometheus, resampled onto an exact 15.00s grid by `query_range`. The healthy
baseline was collected **without** it: a live poll at 14.32s median, snapped to a grid afterwards by
`collect/baseline.py:align_to_grid`. Two frames, two instruments. The detector is fit on one and
evaluated on the other, so this is a provenance issue for every number computed across both.

Neither ruling fixes that on its own. Only re-collection does, and the two options are not the same
size:

- **Restore** → re-collect the healthy baseline under Prometheus. One run, a few hours, one
  workload set, no fault injection.
- **Declare it gone** → re-collect the chaos campaign under cAdvisor. 160 fault runs.

Declaring Prometheus gone would permanently strand the project's most expensive dataset on a
decommissioned instrument, with no committed way to reproduce the environment it came from. Restore
is the only ruling under which the two frames can ever be put on the same instrument at a price
anyone will actually pay.

Two things that did *not* decide it, for the record. Prometheus is not needed by the controller:
`control/controller.py` consumes a live metrics callable and never queries anything. And the
cAdvisor path is genuinely good — it is byte-compatible with `scrape.py`'s schema and imports
`coerce()` from it rather than copying, so a consumer cannot tell which collector wrote a file. That
last property is a virtue for the data and a hazard for provenance, which is the next section.

### The provenance caveat, until the baseline is re-collected

A consumer **cannot tell from the file** which collector produced a frame. Until
`data/healthy/metrics.parquet` is re-collected under Prometheus, any figure computed across the
healthy and chaos frames is computed across two instruments. Known differences:

- **Grid.** `query_range` evaluates at exact step boundaries. `baseline.py` polls (14.32s median,
  12.3–16.3s spread) and snaps backward afterwards, which repeats samples rather than inventing
  them — but a repeated sample and an evaluated one are not the same observation.
- **`mem_pct`.** Prometheus divides by `kube_pod_container_resource_limits` from
  kube-state-metrics; `baseline.py` reads the limit from the pod spec. Same number from its source
  of truth rather than from a re-export — but only while the two agree.
- **Counter resets.** `scrape.py`'s `mem_pct` query uses `max by (namespace,pod,container)` on both
  sides because cAdvisor briefly reports two series per container during a restart. The
  reconstructed rates difference a ring buffer instead.

Re-collecting the baseline is `collect/`'s call to make and its work to do; this directory's job was
to make the instrument reproducible, and to stop the repo from being silent about which one is
current. Both halves are now written down.

### Pinning

`infra/monitoring/chart-version.txt` currently reads `UNPINNED`, and `install.sh` **refuses to
install** while it does. That is deliberate and it is the honest state: this was committed from a
seat with no cluster and no network, so no chart version could be verified. An unpinned chart does
not reproduce an environment, it reproduces a date — the same class of gap as a helm command in a
guide. Resolve it once on a real cluster and commit the result:

```bash
bash infra/monitoring/install.sh --resolve   # writes chart-version.txt
bash infra/monitoring/install.sh             # then installs that exact version
bash infra/monitoring/install.sh --verify    # checks an existing install
```

`--verify` is the check that answers *is this the cluster the data came from*: it asserts
kube-state-metrics is present (the only source of the `mem_pct` limit join) and prints the live
scrape interval and retention. A scrape interval other than 15s is not a resolution preference — 
every frozen query is a `rate(...[1m])`, and that is a different estimator over a different scrape.

**None of `infra/monitoring/` has been run against a live cluster.** It was written offline. Read it
before you run it.
