# eval/

Evaluation harness, plots. Owned by Shaurya — see [SETUP.md](../SETUP.md) §6.

## `harness.py`

Fits Sagar's detector on `data/healthy/`, replays it against `data/chaos/`,
and reports lead time, recall, precision, and false-positives/hour per fault
type (guide Part 7.1/7.3 — never one aggregate number).

```bash
python -m eval.harness
```

Writes `eval/results.csv`.

**Not computed here:** recovery time, ours vs control. That is `recovery.py`
plus `collect/twoarm.py` — see below. `harness.py` stays single-arm and
detector-only: it answers "did we see it coming", the other half answers "and
did acting on it beat stock Kubernetes".

## `recovery.py` + `collect/twoarm.py` — the control arm

Ground rule 4: *"we recovered the service" means nothing without "and stock
Kubernetes didn't."* Two halves, split so the arguable part needs no cluster:

| file | what it is | needs a cluster |
|---|---|---|
| `collect/twoarm.py` | injects the SAME fault into `prodrome` and `control` simultaneously, samples both arms' serving health from one loop | yes |
| `eval/recovery.py` | turns those two files into recovery time per arm, per fault type, and refuses to publish a run that could not have produced a comparison | no |

```bash
python collect/twoarm.py --preflight                   # checks most of it with no cluster
python collect/twoarm.py --campaign --load-attested    # the real run
python -m eval.recovery --indir data/twoarm
```

Tests: `pytest eval/test_recovery.py collect/test_twoarm.py -q` — 60 tests, no
cluster, no Docker, no files under `data/`.

### Recovery time is not one number

`data/chaos/runs.csv`'s `failure_ts` is non-null only for MEMORY_LEAK (30/30)
and POD_KILL (10/10); it is **null for all 90 DISK_STRESS and all 30 CPU_HOG
runs**, because those faults never actually fail — `end_ts` is just when
`stress-ng` stopped. So two quantities, never pooled:

- **`recovery_seconds`** (POD_KILL, MEMORY_LEAK) — first not-serving sample to
  the first sample of a sustained-serving run. A real outage.
- **`degradation_seconds`** (CPU_HOG, DISK_STRESS) — how long readiness-probe
  latency sat outside *that arm's own* pre-injection band. These have no
  recovery time to report, and the code cannot produce one for them.

`NO_OUTAGE` is **counted, never scored as 0s**. A MEMORY_LEAK that Prodrome
restarts before the OOMKill has no recovery time; scoring it as zero would hand
the treatment arm a median of 0s against the control arm's real tens of seconds
— a spectacular result made entirely of a missing measurement.

### Two of four fault types cannot show a difference, by design

From `control/policy.py`: CPU_HOG → `scale_out`, MEMORY_LEAK →
`rolling_restart`, DISK_STRESS → `alert_only`, and POD_KILL has **no policy row
at all** (falls through to `UNKNOWN` → `nothing`). For DISK_STRESS and POD_KILL
the treatment arm is *specified* to behave like the control arm, so a tie is the
design working. The table carries a `lever` and a `comparable` column beside
every row, because a reader averaging four rows would otherwise conclude
Prodrome does nothing.

### It refuses to publish a run that proves nothing

`attest()` has to pass before any number is a comparison:

- **manifest-parity** — `infra/workloads.yaml` vs `infra/workloads-control.yaml`
  normalised for namespace and comments. Currently **identical across 172
  significant lines**, verified. The memory limit *is* the OOMKill threshold, so
  a one-line drift here silently becomes the result.
- **arm-isolation** — `control/controller.py`'s `NAMESPACE` is read, not assumed.
- **treatment-arm-live** — the expensive failure mode. With `DRY_RUN = True`,
  `execute_action()` returns `"dry-run"` before touching the API, so the
  treatment arm *is* a second control arm and every row is a tautology. **This
  attestation fails today.**
- **pair-skew** — arms injected more than 5s apart saw different cluster states.
- **load-parity** — an idle control arm recovers faster for reasons unrelated to
  Prodrome.

### Measured limitations (not hypothetical)

- **Resolution.** Each `kubectl` call costs 600–900ms of startup and API
  round-trip, so the health sampler achieved **6.4s** against a requested 5.0s.
  Differences at or below that are reported as `tied_pairs`, not wins —
  `sampling_resolution()` self-calibrates the band from the data. The first live
  pair came back prodrome 15.2097s vs control 15.2036s and the table reported
  "control faster" on a **6ms** difference; that is now a tie.
- **Probe latency is a weak SLI.** It is a `redis-cli ping` / `pg_isready` /
  `GET /` whose wall time is dominated by `kubectl exec` startup, not by the
  workload. A `NO_DEGRADATION` row means the probe did not notice, not that
  users would not have. The fix is a real request-latency column from
  `collect/load/`; it needs no change to any definition above.

### Two findings worth knowing before reading the numbers

1. **Don't use `ml.replay.replay()` for a long multi-fault file.** It calls
   `Detector.score()` → `WorkloadDetector.update()` on every tick regardless
   of fault status, so each fault it walks through keeps training the
   "healthy" reference — recall silently collapses the deeper into the file
   you go. `harness.py`'s `replay_no_leakage()` instead scores fault ticks
   with `score_only()` (mirrors `ml.replay.train_and_replay()`'s branching)
   and calls `on_restart()` after any run that actually restarted the pod.

2. **A restart's dead zone can outlast the campaign's recovery gap.**
   `on_restart()` wipes the error history (needs `WARMUP_TICKS` more ticks to
   refill) and suppresses firing for `POST_RESTART_SUPPRESS_TICKS` (~8 min) —
   but `collect/chaos.py`'s recovery gap between runs is only 120s. Any fault
   scheduled soon after a restart-prone one (here: DISK_STRESS always follows
   MEMORY_LEAK per workload) lands inside that dead zone and is structurally
   undetectable, independent of how good the detector actually is at that
   fault type. `harness.py` flags this per run (`contaminated` column) and
   reports recall both with and without those runs — the DISK_STRESS row
   comes back 100% contaminated in the current chaos dataset, which means
   this campaign genuinely cannot answer whether the detector catches
   DISK_STRESS. Fixing it means re-running chaos with either a longer
   recovery gap or a fault order that doesn't put DISK_STRESS right after
   MEMORY_LEAK every time.
