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

**Not computed here:** recovery time, ours vs control. That needs Shravan's
live controller and identical faults injected into both namespaces at once
(Phase 4). This harness covers what's measurable now — Phase 2/3, single-arm,
detector-only.

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
