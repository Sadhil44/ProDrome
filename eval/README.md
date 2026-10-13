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

Tests: `pytest eval/test_recovery.py collect/test_twoarm.py -q` — no cluster,
no Docker, no files under `data/`.

### The 2026-10-05 campaign — regenerate it with exactly this

```bash
python collect/twoarm.py --campaign --null-run \
  --order rotate --duration 180 --settle-gate --gap 0 --rounds 3 \
  --outdir data/twoarm
python -m eval.recovery --indir data/twoarm
```

Run both from **inside WSL**. kubectl only works there — the kubeconfig is
`~/.kube/config` inside WSL and there is no Windows one, so from PowerShell or
Git Bash every call fails with `dial tcp [::1]:8080`, which means *no
kubeconfig*, not *dead cluster*.

Three flags are new, and each one is a claim about how the arms were kept
comparable:

- **`--settle-gate`** polls **both** arms before every pair until each workload
  is Ready *and* its restart counter has not moved for 3 consecutive polls.
  This replaces the implicit half of the blind ~9-minute gap — "whatever the
  last pair broke has probably finished unbreaking" — with a checked
  precondition, because a pod still in CrashLoopBackOff would otherwise be
  sampled as the *next* pair's quiet baseline and poison its latency band. It
  waits for both arms, so it cannot hand one arm a cleaner starting state;
  `settled_ok`/`settled_s` land in `pairs.csv` per pair; a timeout is recorded,
  not fatal.
- **`--gap 0`** is only defensible here. The gap's *explicit* purpose — keeping
  a pair out of the detector's post-restart dead zone — still stands, and that
  bias runs **one way** (it makes Prodrome look worse). It is safe to drop in
  this campaign solely because `DRY_RUN = True` and no controller process is
  running, so there is no dead zone. **Restore `--gap` the moment the
  controller is live.**
- **`--order rotate`** emits one pair per fault type in turn, workload cycling
  fastest, so every *prefix* of an interrupted overnight campaign is still
  balanced across fault types and workloads. `pair_id` deliberately does not
  encode the order — it is the resume key, so changing the order continues a
  campaign rather than silently re-running it.

`pairs.csv` gained three columns, appended at the end (old files still load):
`settled_ok`, `settled_s`, and **`trail_s`, the censoring horizon**. A row
reading `NOT_RECOVERED` means "still down after `trail_s` seconds", and that
sentence cannot be read without the number.

### What a `DRY_RUN = True` campaign can and cannot measure

With shadow mode on, `execute_action()` returns `"dry-run"` before touching the
API. The treatment arm is therefore a **second control arm**, and a naive
two-arm table compares stock Kubernetes against stock Kubernetes plus a logger.
`attest_treatment_arm_live()` fails and nothing is publishable as a comparison.

Two things remain true, and `recovery.py` computes both:

- **`stock_baseline_table(measured, shadow_mode=True)`** — recovery per fault
  type with both arms **pooled**. Pooling is valid *because* the treatment arm
  is inert: the arm label carries no information, so the two columns are two
  draws from one distribution and `n` doubles. The same fact that permits this
  is the fact that forbids the comparison, so the function **raises** on
  `shadow_mode=False` — once Prodrome acts, pooling averages the treatment into
  its own baseline. This table is the bar a future live run has to clear.
- **`shadow_noise_floor(measured, tie_band)`** — the true per-pair delta here is
  **zero by construction**, so every difference reported is the harness
  measuring one system twice. `min_detectable_effect_s` (p90 of |delta|, floored
  at the sampling resolution) is the smallest advantage a live run could claim
  and still be distinguishable from this null run. A lopsided sign split would
  be evidence of an **asymmetry in this harness**, not of an effect — there is
  no treatment to have caused one.

**There is no counterfactual.** Shadow mode is supposed to still yield "Prodrome
*would have* restarted at t+22s" from `data/decisions/log.csv`, but that needs
the controller to have been *running* and logging; it was not, and the file does
not exist. `attest_counterfactual_available()` prints that in the report rather
than letting the absence pass as an omission — the gap between "decided not to
act" and "was never asked" is the whole difference. None was manufactured.

**POD_KILL caveat, narrowed.** "POD_KILL is structurally unmeasurable" is true
of **lead time** in `data/chaos/runs.csv` (`failure_ts == start_ts`, zero-width,
zero ticks). It is *not* true of recovery time here: `outage_outcome()` derives
the failure instant from the symmetric health series rather than from
`failure_ts`, so POD_KILL recovery is measurable in this harness. It remains a
designed tie (`POLICY` → `nothing`), so it is a sanity floor, not a target.

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

---

## `sensitivity.py` + `collect/poscontrol.py` — the positive control

The 2026-10-06 campaign came back null: median delta +0.00s, 7 pairs favouring
prodrome against 9 favouring control, sign test p=0.80. That was the **correct**
answer — `DRY_RUN=True` and `load_attested=False`, so both arms genuinely were
the same system, and a harness that found a difference there would have been
broken in the way that looks exactly like a discovery.

But a null run only rules out a harness that **invents** differences. It says
nothing about one that is **blind** to them. Until somebody shows this pipeline
recovering an effect of known size, every future "Prodrome did not beat stock
Kubernetes" is uninterpretable. These two files close that.

| File | Injects | Tests | Needs a cluster |
|---|---|---|---|
| `eval/sensitivity.py` | known delay rewritten into a finished run's `health.csv` | the **analysis** half, end to end | no |
| `collect/poscontrol.py` | asymmetric scale-to-zero hold, live | **collection + analysis** | yes |

Neither is evidence that Prodrome works. The effect is manufactured in both, and
`recovery.publishable()` is unchanged and still refuses the campaign — pinned by
`test_a_manufactured_effect_never_makes_the_campaign_publishable`.

```bash
# analysis half: the dense curve, on the real 63-pair campaign
python -m eval.sensitivity --indir data/twoarm --window trail --tag trail
python -m eval.sensitivity --indir data/twoarm --window trail --arm prodrome --tag signcheck
# collection half: live, ~4 min/pair
python -m collect.poscontrol --preflight
python -m collect.poscontrol --campaign --sizes 0,10,30 --repeats 4
python -m eval.recovery --indir data/twoarm-poscontrol --out eval/poscontrol-results.csv
```

### The number every future comparison has to clear

Replay sweep over the 63-pair campaign, measurement window bounded at each
pair's own `trail_s`. "Reliably recovers" = ≥90% of pairs on the correct side of
the tie band, **and** the median delta within one achieved sampling interval
(6.82s) of the injected size, **and** ≤10% of pairs lost to censoring.

| Fault type | Pairs | Noise floor p90 \|Δ\| | Min **detectable** effect | Min **accurately sized** | Same, honest tie band¹ |
|---|---|---|---|---|---|
| `POD_KILL` | 9 | 6.8s | **7s** | 7s | 7s |
| `MEMORY_LEAK` | 18 | 21.9s | **25s** | 25s | **30s** |
| `DISK_STRESS` | 18 | 78.9s | **40s** | 60s | 40s |
| `CPU_HOG` | 18 | 25.9s | **blind** — no size from 2s to 240s passes | — | blind |

¹ `--tie outage` (7.58s, the cadence achieved while *not* serving) instead of
the 6.82s all-sample median the campaign used. See finding 4. Only
`MEMORY_LEAK` moves, so **quote the conservative column**: 7s / 30s / 40s /
blind.

Read it as: *an improvement smaller than this could be real and this harness
would still report a tie.* It is a floor on claims, not a target.

### The live half — collection *and* analysis

12 pairs, 4 each at 0s / 10s / 30s, `nginx`, both arms taken down together and
one released later (`collect/poscontrol.py`, 2026-10-13, 47 min wall clock).
`inject_ok` 24/24, skew 0.000s on every pair, both arms restored to 1/1.
Tie band 9.17s (`--tie outage`; 5.71s under `--tie auto` — the conclusions are
the same either way).

| Manufactured effect | Pairs | Median recovered Δ | Correct direction | Tied | Verdict |
|---|---|---|---|---|---|
| **0s** (live null) | 4 | **0.0s** | 0 | 4 | correctly tied — invents nothing |
| **10s** | 4 | −5.7s | 1 | 3 | **not detected** (power 0.25) |
| **30s** | 4 | −19.2s | 4 | 0 | **detected** (power 1.00) |

So the live pipeline's minimum detectable effect is **>10s and ≤30s**, and it
**understates by 10.8s** — a true 30s gap is reported as 19.2s. That is the
opposite sign of the replay half's +5s bias and finding 4 explains both: the
replay's manufactured delay starts on a sample instant, whereas a live
recovery instant is quantised independently on each arm by a cadence that
collapses to 9–20s during the outage.

Two honest limits on this run. Four pairs per size is powered for direction but
not for significance — the exact sign test cannot go below p=0.125 at n=4, so
the 30s row's p=0.125 is the floor, not a weak result. And a scale-to-zero is
not a fault: it says nothing about behaviour under stress, only about whether
the instrument can measure a difference between two outages.

### Four findings the positive control produced

1. **The measurement window was open at the top, and that cost the harness
   DISK_STRESS entirely.** `measure()` searched for the recovery instant from
   the injection start to the *end of the campaign*. Five of eighteen CPU_HOG
   control-arm pairs had a "recovery" instant landing after the **next** pair on
   that workload was injected, and six more were reporting degradation that was
   simply the next fault. Bounding each pair at `end_ts + trail_s` moves
   DISK_STRESS from blind to a 40s MDE and leaves `POD_KILL`/`MEMORY_LEAK`
   unchanged. `recovery.pair_window_hi` implements it; the default is still
   open, so the published campaign numbers reproduce, and
   `measure(..., window_seconds="trail")` is the honest setting. **Any re-run of
   `eval.recovery` on degradation faults should pass it.**
2. **The harness systematically overstates.** A manufactured N comes back as
   N + 4.4s to N + 6.4s, consistently, across every fault type and every size.
   That is the quantisation: a recovery is only ever observed at the first
   sample *after* it happens, and the achieved cadence is 6.82s against a
   requested 5.0s. It biases toward whichever arm is being helped, so a reported
   improvement should be read as roughly one sample smaller than it says.
3. **`CPU_HOG` is not measurable through probe latency at this pair count.** One
   pair (`CPU_HOG_constant_nginx_001`) carries a 179.3s excursion in the arm
   nothing was done to, which no injected effect below that size can outvote on
   a per-pair basis; above ~150s the delay runs past the 300s trailing window
   and pairs start censoring instead. There is no effect size in between where
   the instrument works. Fixing it means a real request-latency SLI from
   `collect/load/`, not more pairs.

4. **The sampler slows down by 2.6× exactly while the outage it is measuring is
   happening — and the tie band is calibrated on the wrong population.** Only
   the live half could find this. `collect/twoarm.kubectl_probe` runs
   `kubectl exec deploy/<w> -- true` unconditionally; when the deployment has
   no pod there is nothing to exec into, so the call blocks for its full 15s
   timeout. Measured:

   | run | cadence while serving | cadence while **not** serving |
   |---|---|---|
   | `data/twoarm-poscontrol` (scale-to-zero, no pod at all) | 5.63s | **10.4–14.8s** |
   | `data/twoarm` (MEMORY_LEAK leaves a crashlooping pod to exec into) | 6.80s | **7.58s** |

   Both endpoints of a recovery time sit *inside* an outage, so the
   not-serving cadence is the resolution that actually bounds the measurement
   — but `sampling_resolution()` takes the median over all samples, which are
   overwhelmingly healthy ones. `TIE_BAND_SECONDS` therefore understates the
   real quantisation. `sensitivity.cadence_by_state()` reports the split and
   `sensitivity.outage_resolution()` returns the honest number;
   `--tie outage` uses it.

   **Two cheap fixes, in order of value.** (a) Skip the `exec` when
   `ready_replicas == 0` — there is no pod, the call can only time out, and
   that 15s block is the whole of the degradation. (b) Derive the tie band
   from the not-serving cadence. Neither is done here: (a) changes the
   collection contract mid-campaign and (b) would restate published numbers,
   so both are proposed rather than applied.

The sign is taken from the data, not from the sampler's polling order:
`--arm prodrome` produces **+N** where `--arm control` produces **−N**, which is
the check that matters given that the POD_KILL millisecond trap was a
measurement-order artifact.
