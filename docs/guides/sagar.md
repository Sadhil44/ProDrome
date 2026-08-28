# Sagar - The Detector

## Prodrome: a complete beginner's guide

You decide whether a workload is behaving abnormally and heading for failure. This guide assumes no background in anomaly detection and builds the method from first principles.

**You never need a cluster.** Your input is a file of numbers. You can do all of this on a laptop with Python.

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

- [ ] Shared feature code, with the canonical metric order frozen
- [ ] Replay harness producing recall and lead time from a stored file
- [ ] Detector fit only on healthy data
- [ ] Zero-variance metrics identified and dropped
- [ ] Warm-up and post-restart suppression both working
- [ ] Parameter sweep run, configuration chosen deliberately
- [ ] Per-fault recall and lead-time table
- [ ] False positives per hour from the clean runs
- [ ] The score-over-time plot for one ramping leak
- [ ] Model file running in Shravan's live loop
