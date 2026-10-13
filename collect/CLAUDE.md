# collect/ module rules

This directory belongs to the **collect** seat (Shaurya): the metric scraper, the chaos runner and
k6 load generation. It produces the data every other seat consumes, so its output schemas are the
most-depended-on thing in the repo.

- The two output schemas are contracts (`SETUP.md` §7), not local choices. Metrics table, Parquet,
  one row per workload per 15-second tick: `ts, workload, cpu_cores, mem_bytes, mem_pct, net_rx,
  net_tx, fs_reads, fs_writes, restarts`. Labels table, CSV, one row per injected fault:
  `start_ts, end_ts, workload, fault_type, pattern, run_id`. Changing a column means appending to
  `collect/interfaces` and announcing it labeled `signal` and `diagnosis`.
- `workload` is a stable name — `redis`, not `redis-7d9f8b-x2k1`. Pod-instance names do not join
  across runs and silently break every split.
- `run_id` is required on every label row. Splits are by run, never by row, and a missing `run_id`
  makes a run unusable rather than merely awkward.
- Faults come in both `constant` and `ramp` patterns. If every fault jumps straight to full
  intensity, the classifier never sees what a slide toward failure looks like, which is the case the
  whole project exists for.
- Healthy data must not be flat. Real baselines have a diurnal cycle and noise; a flat baseline
  teaches the detector that normal means flat, and it then fires on the first genuine traffic change.
- Label the full fault trajectory including the faint early windows, not just the peak. Labelling
  only the obvious part produces a classifier that works only once the failure is already
  catastrophic — exactly where it is no longer useful.
- `data/` is gitignored except `data/samples/`. Keep the committed fixture small (a few MB) and keep
  it valid against the current schema: it is the fallback that lets every other seat work when a
  collection run is not ready. Never commit a full run or a model artifact — git keeps every version
  forever.
- Chaos runs are recorded as they are injected, not reconstructed afterwards from memory. A run whose
  labels were written after the fact is not a run.
- Coordination happens on the CCP forum (MCP `ccp-forum`, session `prodrome`): set status, post to
  `collect/updates`, and append contract changes to `collect/interfaces/metrics-table-schema` and
  `collect/interfaces/labels-table-schema`. See `forum/HANDOFF.md`.
