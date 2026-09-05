# Sagar - The Detector

## Prodrome: a complete beginner's guide

You decide whether a workload is behaving abnormally and heading for failure. This guide assumes no background in anomaly detection and builds the method from first principles.

**You never need a cluster.** Your input is a file of numbers. You can do all of this on a laptop with Python.

> **Where this fits** (see `PRD.md` §10): everything below is Phase 0–3 work, and unlike Shaurya's guide there's no ordering gotcha — you're never blocked on a cluster, so you can start immediately once Shaurya's Phase 0 synthetic generator (or `data/samples/`) exists. Part 8's later-phases items are Phase 5–6.
>
> **Repo status right now:** nothing in `ml/detector.py` or a shared feature file exists yet. Sadhil has already written a *provisional* windowing function (`ml/dataset.py`'s `windows()`) and a 40-feature summarizer (`ml/features.py`), explicitly flagged as standing in for your canonical shared feature file until it exists — worth reading before you start, so you're reconciling into something rather than duplicating it from scratch. No real or synthetic data exists yet either (Shaurya hasn't built the Phase 0 generator), so nothing you build can be tuned against real numbers yet.
>
> **Your next 3 steps:** (1) look at `ml/dataset.py` and `ml/features.py`, agree the canonical metric order with Sadhil, then move windowing into your shared file per Part 4.1. (2) build the detector itself, Part 4.3 — testable against any fabricated fixture in the meantime, doesn't need Shaurya's generator to exist. (3) build the replay harness, Part 4.2.

---

## Part 1 — What you're actually building

### 1.1 The job in one sentence

Every 15 seconds you receive eight numbers per workload — CPU, memory, memory as a fraction of its limit, network in and out, disk reads and writes, restart count. You answer one question:

> **Is this workload behaving abnormally in a way that precedes failure?**

You output a score and a yes/no. If you say no, nothing downstream happens. **You are the gate.**

### 1.2 Why this is harder than it sounds

The naive version is a threshold: `if memory > 90%: alert`. Three problems.

**The right number differs per workload.** Redis at 85% might be fine forever; postgres at 85% might be minutes from death. You'd hand-tune a threshold per metric per workload and re-tune whenever anything changes.

**Level doesn't distinguish healthy from sick.** A service under legitimate heavy load looks identical to a service leaking memory, at any single instant. The difference is in the *shape over time*.

**Some failures never cross a static threshold** until the moment they fail.

You need something that learns what normal looks like for each workload and notices deviation from *that*, rather than deviation from a number a human guessed.

### 1.3 The rule that defines your entire job

> **Your detector trains only on healthy data. It never sees a single injected fault.**

This is not a stylistic preference. It's the decision that makes the project's results defensible, and it comes from published work where the detector was fit on a week of failure-free execution with zero fault samples.

**Reason one — real failures can't be listed in advance.** You cannot enumerate every way a system can break. A detector that only recognizes faults it has already seen is useless against the one that actually takes you down. Learning "normal" and flagging deviation covers failure modes nobody anticipated.

**Reason two — and this is the one that matters for your credibility.** Shaurya's fault injector produces *step functions*. `stress-ng --cpu 4` takes CPU from 20% to 95% in a single tick. Real degradation is a *slide* over minutes.

If you train on injected faults, your model learns "detect sudden jumps." It will score beautifully on injected faults and be useless on real ones. And a `>` operator already detects sudden jumps for free, so you'd have built a neural network that reimplements an if-statement.

**Train on normal. Detect deviation. Chaos is the test set, permanently.**

If at any point you find yourself thinking *"it would probably work better if I mixed in a few faults"* — **that is the exact moment the project stops being defensible.** Stop and talk to the team instead.

### 1.4 Why we're not using the paper's model

The reference paper uses Hierarchical Temporal Memory, a biologically-inspired sequence model. We're not, for one practical reason worth stating in the README: the library is a fragile native build that can consume an entire day.

A companion paper found a **plain statistical detector** scored 96% precision against HTM's 100%, with recall of 86% versus 89%. Near-identical performance, zero installation cost. That's a defensible engineering call, not a shortcut.

---

## Part 2 — The method, built up from nothing

### 2.1 Step one — predict the next value

The simplest useful model of a time series: **the next value will be roughly like the recent values.**

An **exponentially weighted moving average** (EWMA) implements this:

```
prediction = α × current_value + (1 − α) × previous_prediction
```

α (alpha) between 0 and 1 controls memory. **α = 0.9** means "mostly just the latest value" — reacts fast, noisy. **α = 0.1** means "mostly history" — smooth, slow to react. We use around 0.3.

It's called *exponentially weighted* because expanding the recursion shows older values contribute with exponentially decaying weight. You don't store history; the single number carries it.

### 2.2 Step two — measure how wrong you were

Each tick, compute the **prediction error**: `|actual − predicted|`.

On a stable workload, error is small. When something changes suddenly, error spikes.

Naive approach: threshold the error. But how large is "large"? It depends on the metric, its units, and how noisy that workload naturally is. You're back to hand-tuning per metric.

### 2.3 Step three — the insight that makes it self-calibrating

Instead of asking "is this error large," ask:

> **Am I more wrong than I usually am?**

Keep a rolling window of your last 100 errors. Compute their mean and standard deviation. Then compare your *recent* errors (last 5) to that distribution:

```
score = (mean of last 5 errors − mean of all 100 errors) / (std dev of all 100 errors)
```

This is a **z-score**: how many standard deviations above typical is the recent behaviour.

**Why this is the key idea.** The score is in units of "standard deviations of my own error," so it means the same thing for memory in bytes, CPU in cores, and disk in bytes-per-second. A metric that's naturally noisy has a large error standard deviation, so it takes a bigger deviation to trigger. A metric that's normally rock-steady triggers on a small one.

**No per-metric tuning. It calibrates itself from the data.**

Threshold the score at roughly **3.0**. That's "three standard deviations above typical error," which under a rough normal assumption is uncommon.

### 2.4 Step four — don't fire on one metric

Run one of these detectors per metric: 8 metrics × 3 workloads = 24 independent detectors.

**Healthy systems throw individual anomalies constantly.** A garbage collection pause. A slow query. A network blip. If you alert every time any single metric looks odd, you'll alert every few minutes forever.

So confirm at the workload level with two parameters:

- **k** — how many of the 8 metrics must be anomalous simultaneously (try 2 or 3)
- **n** — how many consecutive ticks that must hold (try 1 or 2)

Fire only when both are satisfied. This kills the vast majority of false positives at the cost of a few seconds of lead time.

### 2.5 The full picture

```
8 metrics → 8 EWMA predictors → 8 error histories → 8 z-scores → 8 booleans
                                                                      ↓
                                                        count ≥ k for n ticks?
                                                                      ↓
                                                                    FIRE
```

That's the entire detector. Roughly sixty lines of numpy. **There is no training loop, no gradient descent, no GPU.** Feeding it the healthy data stream once *is* the training — which is also precisely why fault data must never enter that stream.

---

## Part 3 — Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyarrow matplotlib
```

That's it. No cluster, no Docker, no Kubernetes.

**Parquet** is a columnar file format. `pandas.read_parquet()` reads it; it's faster than CSV and preserves data types.

Your input is a table with columns: `ts`, `workload`, and the eight metrics. One row per workload per 15-second tick.

---

## Part 4 — Building it

### 4.1 Shared feature code — you own this file

Everyone imports it, so get it right early.

Define the **canonical ordered list** of the eight metric names. **Order matters** — Sadhil and Shravan both build arrays indexed by position. If someone reorders it, everything silently breaks.

Write a windowing function: given the metrics table and a workload name, produce sliding windows of 20 consecutive ticks (five minutes of history).

Decide missing-value handling **once**, here: forward-fill, then fill remaining with zero. Everyone gets identical behaviour.

Coordinate with Sadhil — he extends this same file with feature summarization. One file, two owners.

### 4.2 The replay harness — build this second

A function that feeds a stored dataset through your detector tick by tick, recording every firing with its timestamp.

**This is your most valuable tool.** Every experiment is a replay over a file, which takes seconds. You will run hundreds of them. Building this before the detector itself is the right order.

### 4.3 The detector

**Per-metric detector.** Holds a running prediction and a bounded history of the last 100 errors. On each update: compute error, update prediction, append to history, compute the z-score, return score and boolean.

Three guards you must include:

**Floor the standard deviation.** If a metric is perfectly constant when healthy, its error standard deviation is zero, and dividing by zero makes every deviation look infinite — that metric then fires constantly. Add a tiny constant to the denominator. Also, separately, **check your healthy data for near-zero-variance metrics and drop them entirely.** The restart counter is the usual offender — it's zero forever until it isn't.

**Warm-up suppression.** The first ~30 ticks have no meaningful error history. Return "not anomalous" until enough has accumulated.

**Non-finite values.** NaN and infinity will appear. Skip them rather than letting them poison the history.

**Workload-level detector.** Holds 8 per-metric detectors per workload. Counts anomalies, tracks the consecutive streak, fires when the streak reaches n.

### 4.4 The post-restart reset — do not skip this

After a pod restarts, its metrics look nothing like steady state: memory near zero, CPU spiking during startup, error rate settling.

**To your detector this looks exactly like a severe anomaly.** So it fires. So Shravan's controller restarts the pod. So it fires again.

**That's an infinite loop, and it will happen the first time the controller goes live.**

Expose a `on_restart(workload)` function that resets that workload's detectors and suppresses output for about 8 minutes. Tell Shravan to call it after every restart action. Test it together.

### 4.5 Fit and ship

Feed the entire healthy dataset through, in timestamp order, once. Save the warmed-up object with `pickle`.

> **Ship the file to Shravan early, even if it's terrible.** He is blocked on the *interface*, not the accuracy. A useless detector he can import beats a great one he can't. Agree the exact function signature and write it in the README:
>
> ```python
> score, fired = detector.score(workload_name, array_of_8_values)
> ```
>
> Don't change that signature afterwards without telling him.

---

## Part 5 — Tuning

### 5.1 The sweep

Four parameters:

| Parameter | Try | Controls |
|---|---|---|
| α | 0.1, 0.3 | Prediction responsiveness. Lower = slower to accept change = more sensitive |
| threshold | 2.5, 3.0, 4.0 | Per-metric sensitivity in standard deviations |
| k | 2, 3 | How many metrics must agree |
| n | 1, 2 | Consecutive ticks to confirm |

24 combinations. Each evaluation is a replay over a stored file — seconds. **Write it as a loop, never tune by hand.**

For each configuration, measure recall and median lead time per fault type on the chaos runs, and false positives per hour on the clean runs.

### 5.2 The tradeoff you cannot escape

Lower thresholds → more firings → higher recall, lower precision. Higher → the reverse.

**There is no configuration that's best at both.** Pick deliberately and report both the balanced choice and the precision-optimal one. The published paper found exactly the same tradeoff; naming it is a sign you understand the problem, not a weakness.

### 5.2.1 Chosen configuration

**Superseded once (below), then confirmed against real data.** First pass was a 24-combo sweep (`ml/tune.py`) against the bundled `data/samples/metrics.parquet` fixture, which picked `alpha=0.1, threshold=2.5, k=3, n=1` (0.00 fp/hr, 0.83/0.83/0.50 recall). That config was never validated against anything real -- and once `data/healthy/metrics.parquet` + `data/chaos/{metrics.parquet,labels.csv}` landed, re-running the same sweep for real (`Detector.fit_healthy` on real healthy data, `ml.replay.replay` against real chaos data -- safe here since the two files don't overlap in time) showed it collapsing:

| | alpha | threshold | k | n | fp/hr | CPU_HOG | DISK_STRESS | MEMORY_LEAK | POD_KILL |
|---|---|---|---|---|---|---|---|---|---|
| Synthetic-fixture pick, on real data | 0.1 | 2.5 | 3 | 1 | 0.27 | 0.42 recall, 260s lead | **0.00 recall** | **0.00 recall** | 0.0 |
| First real-data pick | 0.1 | 2.5 | 2 | 1 | 0.71 | 0.42 recall, 260s lead | 0.17 recall, 262s lead | 0.42 recall, 11s lead | 0.0 |
| **Balanced (chosen default)** | 0.1 | **2.0** | 2 | 1 | 1.79 | 0.50 recall, 269s lead | 0.33 recall, 236s lead | 0.58 recall, **27s lead** | 0.0 |
| **Sensitivity-leaning alternate** | 0.1 | **1.5** | **1** | 1 | 5.00 | 0.58 recall, 276s lead | 0.58 recall, 272s lead | 0.75 recall, **42s lead** | 0.25 recall, 5s |

**What changed and why, in two steps:** `k=3` measured well on the synthetic fixture because the generator stylizes each fault to move several metrics together cleanly; real `DISK_STRESS`/`MEMORY_LEAK` don't correlate across metrics that cleanly, so `k=3` never fired on either (0.0 recall) -- dropping to `k=2` fixed that. Separately, a finer real-data sweep (lower thresholds, `k=1`) showed threshold is the more powerful lever than k once k=2 is already in place: `threshold=2.0` meaningfully improves every fault type's recall and more than doubles `MEMORY_LEAK` lead time (11s -> 27s), for a real but modest fp cost (0.71 -> 1.79/hr, roughly one false alarm per 34 minutes). Pushing further to `threshold=1.5, k=1` buys still more recall and lead time but at 5.00 fp/hr -- probably too disruptive to ship as the default (that's a false controller action roughly every 12 minutes), so it's recorded as the explicit alternate Part 5.2 asks for rather than adopted.

**What real data revealed, not just retuned:**
- **A genuine precision/recall tradeoff finally shows up** (fp/hr from 0.71 to 5.00 across the rows above) -- the synthetic fixture was too small to ever show this (Part 5.2's original caveat).
- **Recall is lower across the board on real data** than the synthetic fixture ever suggested -- report per fault type, not as one aggregate (ground rule #6).
- **Real `MEMORY_LEAK` lead time tops out around 27-42s, not the ~165-225s the synthetic fixture implied** -- a real leak develops far faster (or breaches sooner) than the generator modeled. This is the number that actually matters for "can the controller act in time," and it's a much thinner margin than the synthetic result suggested.
- **`POD_KILL` recall is 0.0 at the chosen default, exactly as Part 5.4 predicts** ("~0s ← expected, reported on purpose") -- it's instantaneous, there's no window to catch. The 0.25 recall at the aggressive alternate is almost certainly catching the aftermath within one tick, not real warning, and shouldn't be read as "the detector predicted a pod kill."

The balanced pick is now `ml/detector.py`'s module defaults (`ALPHA`, `Z_THRESHOLD`, `K_OF_N_METRICS`, `N_CONSECUTIVE`).

### 5.3 Calibrate expectations before you start

Published results on this exact problem, on a cleaner testbed than yours:

- **Precision 0.65**, recall 0.92
- **Median lead times from 15 minutes to over 2 hours**, depending on fault type
- The workload-related fault was worst — 15 minutes, 0.52 recall, because congestion only looks like congestion once you're already congested

**A third of predictions being false alarms is the state of the art.** If you hit 0.7 precision you have reproduced published work. Do not chase 0.95 — it doesn't exist for this problem, and if you get it, **check whether fault data leaked into your healthy set.** Look at the timestamps.

### 5.4 Your deliverable

| fault | runs | recall | median lead time |
|---|---|---|---|
| CPU_HOG | 6 | ? | ? s |
| MEMORY_LEAK | 6 | ? | ? s |
| DISK_STRESS | 6 | ? | ? s |
| POD_KILL | 2 | ? | ~0 s ← expected, reported on purpose |

Plus false positives per hour from the clean runs.

**Per fault type, never one aggregate.** An average hides that one class gets twenty minutes and another gets none, and the failures are the informative part.

### 5.5 The plot that sells the idea

For one ramping memory leak, plot over time: memory usage, your detector score, the moment you fired, and the moment the failure occurred.

Anyone looking at that image understands the product in three seconds. It's worth more than any table.

---

## Part 6 — What will go wrong

| Symptom | Cause and fix |
|---|---|
| One metric fires constantly | Near-zero healthy variance. Drop that metric from the set |
| Fires right after every restart | Cold start. Increase warm-up, wire `on_restart` with Shravan |
| Fires during the clean runs | Threshold or k too low — you're detecting noise |
| Never fires at all | k too high, or the ramp is gentler than natural healthy variation |
| Great recall, terrible precision | Expected. Report both, choose deliberately |
| Recall above 0.95 everywhere | **Suspect fault data in your healthy set.** Check timestamps against the labels file |
| Works on generated data, not real | Real metrics are noisier and correlated. Raise thresholds. This is normal |
| Detector object won't unpickle | Class definition changed since pickling. Refit rather than debugging pickle |

---

## Part 7 — Working with Sadhil

You're paired. Two concrete interfaces between you:

**The shared feature file.** You own windowing; he adds summarization. Talk before editing.

**Your firing timestamps.** He needs them for the accuracy-versus-lead-time curve — the headline result. He buckets classification accuracy by how far from failure each window was; **your half is which of those windows you had fired on.** Give him the replay output as a file.

The curve shows accuracy rising as failure approaches. It quantifies the tradeoff nobody writes down: **the earlier you act, the less you know about what you're acting on.** That's the finding of the project, and half of it is yours.

---

## Part 8 — Later phases

**Automatic baselining.** Right now a human runs a script to fit you. In the shipped product, Prodrome installs, collects healthy data on its own for some hours, fits, and starts predicting. This one feature is what makes the tool installable by a stranger.

**A third tier.** The paper has one we skipped: a per-workload model over the *pattern* of which anomalies co-occur, rather than a raw count. Healthy systems have a characteristic anomaly signature; deviation from that signature is a better signal than counting. It's about ten lines with scikit-learn's `OneClassSVM`. Add it once the simple version is measured.

**Countdown regression.** Your output is binary today. Predicting *seconds until breach* unlocks the urgency logic — the difference between "restart now" and "restart at the next quiet minute." Labels come free from Shaurya's runs.

**The forecaster for Phase 6.** When the project reaches the learned-policy phase, you build the load forecaster feeding the agent's state vector, and the simulated training environment. The reference architecture uses a small two-layer LSTM over a 20-interval lookback — the same window size you're already using.

---

## Definition of done

- [x] Shared feature code, with the canonical metric order frozen (`ml/features.py`)
- [x] Replay harness producing recall and lead time from a stored file (`ml/replay.py`, `ml/tune.py`)
- [x] Detector fit only on healthy data (`Detector.fit_healthy` / `train_and_replay`'s healthy-only `update()` calls)
- [x] Zero-variance metrics identified and dropped (`MIN_HEALTHY_VARIANCE` guard in `Detector._init_workloads`)
- [x] Warm-up suppression working (`WARMUP_TICKS`)
- [x] Post-restart suppression (`on_restart()`) verified (`ml/test_restart_suppression.py`): fires without it, fully suppressed with it for the full window
- [x] Parameter sweep run (24 combos, `ml/tune.py`); configuration chosen deliberately -- see Part 5.2.1
- [x] Per-fault recall and lead-time table -- from the sample fixture initially, now superseded by a real run against `data/healthy/` + `data/chaos/` (Part 5.2.1): CPU_HOG 0.42/260s, DISK_STRESS 0.17/262s, MEMORY_LEAK 0.42/11s, POD_KILL 0.0 (expected)
- [x] False positives per hour -- 0.71/hr on real data (clean stretches within `data/chaos/`), plus the earlier 0-fire result on a held-out split of real healthy data alone
- [x] The score-over-time plot for one ramping leak (`ml/plot_leak.py` -> `ml/memory_leak_example.png`)
- [ ] Model file shipped (`ml/detector.pkl` via `ml/fit_detector.py`, frozen `score()`/`on_restart()` interface documented in `ml/README.md`) -- not yet wired into a live loop, because no such loop exists in `control/controller.py` yet (that's Shravan's piece to build)


---

## Appendix — the full phase plan, all four roles

This is the complete phase-by-phase breakdown from `PRD.md` §10–§11, reproduced here in full so nobody has to cross-reference the PRD to see how the four tracks connect. Identical copy in all four guides.

Seven phases. Each has a stated **exit criterion** — the thing that has to be true before anyone moves on, not just "time to move on."

### Phase 0 — Scaffolding
**Exit criterion:** *any one* component builds and tests with no cluster and no teammate. The whole point is nobody needs anybody else yet.

| Person | Deliverable |
|---|---|
| Shravan | Chaos-runner stub; cluster configuration committed |
| Shaurya | The synthetic data generator — "the artifact that unblocks everyone else" — plus committed samples |
| Sagar | Shared feature and windowing code; detector stub |
| Sadhil | Policy table; feature summarization design; classifier stub |

### Phase 1 — Skeleton
**Exit criterion:** a hand-injected fault produces a logged decision. The loop runs end-to-end for the first time — badly, on stubs, but completely.

| Person | Deliverable |
|---|---|
| Shravan | Cluster with limits and probes; controller loop running on stubs, in shadow mode |
| Shaurya | Metric collection with no workload instrumentation; scraper verified against real Prometheus |
| Sagar | The real detector, built and tuned against generated (synthetic) data |
| Sadhil | The real classifier, trained on generated faults |

Dependency to notice: Shravan's controller loop can run immediately against the ML pair's stubs — he doesn't need real models. But Sagar's and Sadhil's *real* models both need Shaurya's *full* Phase 0 generator (faults + labels, not just healthy data) to train against. That's the actual chokepoint in this phase.

### Phase 2 — Grounded
**Exit criterion:** the pipeline runs on real data, and the detector fires on real faults. Synthetic data gets replaced by the real thing.

| Person | Deliverable |
|---|---|
| Shravan | Control-arm namespace running identical workloads with no Prodrome |
| Shaurya | Canonical healthy baseline; fault matrix with two activation patterns (constant + ramp); clean runs; labeled dataset — all real, off the actual cluster |
| Sagar | Refit on the canonical healthy baseline; thresholds adjusted for real noise |
| Sadhil | Retrained on canonical labels; comparison against Phase 1 generated-data results |

This is where the biggest risks in `PRD.md` §13 concentrate: "detector trained on fault data by accident," "control arm omitted or added late" — both must be resolved by end of this phase, not later.

### Phase 3 — Learned
**Exit criterion:** a published comparison table, with the better model actually in the loop. The "prove it's not just a rule dressed up as ML" phase.

| Person | Deliverable |
|---|---|
| Shravan | The simple baseline classifier — a shallow, readable decision tree (cross-cutting responsibility, separate from his main track) |
| Shaurya | Evaluation harness producing the per-fault results table |
| Sagar | Per-fault recall and lead-time table; false positives per hour |
| Sadhil | Model comparison table (tree vs. forest vs. CNN); confusion matrix; feature attribution |

Standing rule governing this whole phase: a simple baseline is built and reported before any sophisticated model is credited. If the forest ties the tree, the tree ships — that's a stated legitimate outcome, not a failure.

### Phase 4 — Acting
**Exit criterion:** a defensible comparison against stock Kubernetes. `PRD.md` calls this out directly: "Phase 4 is where the project first has a result rather than a demo."

| Person | Deliverable |
|---|---|
| Shravan | Controller live (dry-run off), safety rails verified, head-to-head experiment executed |
| Shaurya | Head-to-head experiment execution; the actual results |
| Sagar | Detector live in the loop; restart-suppression verified (the infinite-loop guard) |
| Sadhil | Abstention measurement (both numbers); the accuracy-versus-lead-time curve — the headline result |

Sagar's live-firing timestamps and Sadhil's headline curve are directly coupled here — his replay output is the x-axis of that plot.

### Phase 5 — Packaged
**Exit criterion:** an external operator installs it from scratch and reports back.

| Person | Deliverable |
|---|---|
| Shravan | Declarative per-workload configuration resource, scoped ServiceAccount permissions, one-command install |
| Shaurya | Automatic baselining on install (no manual training step); the live terminal dashboard |
| Sagar | Automatic baselining for the detector; drift detection (refit trigger) |
| Sadhil | Pluggable failure taxonomy; open-set/novelty recognition (upgrading the abstention floor) |

### Phase 6 — Learned policy
**Exit criterion:** the learned policy matches or beats the hand-written table, in simulation *and* in shadow on the real cluster. The biggest phase by far — `PRD.md` §10.1 gives it its own detailed sub-plan.

Two implementations of one shared interface:

| Implementation | Built by | Purpose |
|---|---|---|
| Simulated environment | Sagar + Sadhil | Fast fake model of workload/pod/failure dynamics — thousands of episodes/hour |
| Real environment | Shaurya + Shravan | The actual cluster and fault harness, behind the identical interface |

| Person | Deliverable |
|---|---|
| Shravan | Real-cluster training environment behind the shared interface; shadow-mode deployment of the learned policy |
| Shaurya | Episode replay from the decision log; the cost-versus-latency curve comparing learned policy vs. hand-written table vs. HPA vs. KEDA |
| Sagar | The load forecaster feeding the agent's state vector; the simulated training environment itself |
| Sadhil | The RL agent, its reward function, and the offline training pipeline seeded from logged episodes |

**Safety ordering matters more than architecture here:** (1) bootstrap from the Phase 4 policy table so the agent starts competent, (2) train offline on logged decisions with zero exploration, (3) simulate — thousands of fast episodes, (4) then go online but only in the chaos environment, never a real workload, (5) ship in shadow, disagreements logged and read before anything gets authority. The reward needs an explicit cost term and action-count penalty, or the project reproduces NimbusGuard's failure mode — 5x faster, 80% more replicas — which the PRD treats as a cautionary tale, not a template.

**The honest exit scenario, stated explicitly in the PRD:** if the learned policy just converges to the hand-written table, that's not a null result — it's validation that the table was already correct.

### The thread that runs through all seven phases

`PRD.md` §12.1: **Baseline → Fit → Perturb → Learn → Decide → Act → Measure → Iterate.** Every new fault type, model, or action re-enters this cycle at the relevant stage, and nothing skips **Measure** — which is really just the standing review question restated: *what's the baseline, and did you beat it?*
