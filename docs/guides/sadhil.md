# Sadhil - The Classifier and Policy

## Prodrome: a complete beginner's guide

You name the fault once Sagar's detector fires, and you own the table that turns a name into an action. This guide assumes no background in supervised learning and builds it from first principles.

**You never need a cluster.** Your input is a file of numbers and labels.

> **Where this fits** (see `PRD.md` §10): same as Sagar — no cluster dependency, no ordering gotcha. Parts 1–7 are Phase 0–4 work. Part 9's later-phases items are Phase 5–6. You're currently blocked on Phase 1 (a real classifier trained on generated faults) until Shaurya's Phase 0 synthetic generator exists — see Part 0 of `docs/guides/shaurya.md`.
>
> **Repo status right now:** your Phase 0 is done — `control/policy.py` (policy table), `ml/classifier.py` (stub classifier), `ml/features.py` (40-feature summarization), and `ml/dataset.py` (dataset assembly, with a provisional windowing function pending Sagar's canonical one) are all written, tested, and committed.
>
> **Your next 2 steps:** (1) formalize the ad-hoc smoke tests into a real test suite for `features.py`/`dataset.py`/`policy.py` — genuinely doable now, nothing to wait on. (2) once Sagar starts his shared feature file, reconcile your provisional `windows()` in `ml/dataset.py` into it rather than maintaining two versions.

---

## Part 1 — What you're building and why

### 1.1 The job in one sentence

Sagar's detector says *something is wrong with redis*. You answer:

> **What kind of wrong, and how confident am I?**

Then your policy table turns that into an action.

### 1.2 Why this half of the project exists

Kubernetes' only response to a sick pod is "restart it," because a health probe knows nothing except alive or dead. But the right fix depends entirely on the cause:

| Cause | Restart it? | Correct action | If you restart instead |
|---|---|---|---|
| Memory leak | Yes | Restart — fresh heap | Correct by luck |
| CPU saturation under load | No | Scale out | **You removed capacity mid-saturation.** Remaining pods get more load and start failing too |
| Disk pressure | No | Alert a human | Nothing changes. The disk is still full |
| Connection pool exhausted | No | Shed load | Crashloop against a database that's still drowning |

Same external symptom — pod unhealthy, latency climbing. **Opposite cures, and two of them make things worse.**

Choosing the lever is the entire job, and it's the argument for why Prodrome is more than a faster health check.

### 1.3 Read this before you start: your task is easy, and that's the problem

**Fault classification is nearly trivial**, for a slightly deflating reason: `stress-ng --cpu` spikes the CPU metric, `--vm` spikes memory, `--hdd` spikes disk I/O. The giveaway *is* the corresponding metric.

Published systems report ~0.95 accuracy. Shravan's four-line decision tree will get most of it right.

**So high accuracy proves almost nothing.** Your job isn't accuracy — it's the three things that make the result mean something:

**One: beat the simple baseline, or say clearly that you didn't.** If a readable rule ties your model, ship the rule and write that in the README. That's a stronger result than a marginal black-box win, and it makes every other number in your report more believable.

**Two: compound faults.** A memory leak *during* a CPU spike. Single-metric rules can't untangle overlapping signatures; a learned model might. This is where you actually earn your place.

**Three: abstention.** Knowing when you don't know. **Almost nobody measures this**, and it's what decides whether the system is safe to run unattended.

---

## Part 2 — Supervised learning from zero

### 2.1 The setup

Supervised learning means: you have examples where you know the right answer, and you want a program that gets the right answer on new examples.

- **Features (X)** — the inputs. Numbers describing one situation
- **Label (y)** — the correct answer for that situation
- **Model** — a function from features to label, whose internals are fitted from examples

You have thousands of examples because Shaurya's injector labels them for free.

### 2.2 Train and test

You cannot judge a model on data it learned from — it can memorize. So you split: fit on the training set, evaluate on the test set it has never seen.

**This is where your project's most dangerous bug lives.** See §4.3.

### 2.3 Features versus raw data

Your raw input is a 20×8 window — twenty timesteps, eight metrics. Flattened, that's 160 numbers.

That's a bad representation. A tree looking at "memory at timestep 7" learns nothing useful, because which timestep matters shifts constantly.

Instead, summarize each metric across the window into five numbers:

| Feature | What it captures |
|---|---|
| mean | The level — how high is it |
| slope | The trend — is it rising or flat |
| std dev | The volatility — is it steady or erratic |
| max | The peak reached |
| last | The current state |

8 metrics × 5 = **40 features**. Fewer numbers, far more signal.

> **Slope is the most important feature in the whole set.** It's what distinguishes a memory *leak* (climbing steadily) from steady high memory usage (flat and fine). Level alone cannot tell those apart — and level alone is exactly what a naive threshold sees. **If your model beats a threshold anywhere, slope is why.**
>
> Compute slope by fitting a straight line to the twenty points and taking its gradient (`numpy.polyfit` with degree 1).

### 2.4 Decision trees, in one page

A decision tree is a flowchart of yes/no questions:

```
is mem_pct_slope > 0.01?
├── yes → MEMORY_LEAK
└── no → is cpu_cores_mean > 0.45?
          ├── yes → CPU_HOG
          └── no → NORMAL
```

Training means choosing which questions, in which order, to best separate the labels. **Depth** is how many questions deep it can go — deeper fits training data better and generalizes worse.

A tree's superpower is that a human can read it. That matters enormously here: an on-call engineer can audit a flowchart.

### 2.5 Random forests

One tree overfits — it memorizes quirks of the training data. A **random forest** trains hundreds of trees, each on a random subset of the data and features, then has them vote.

Averaging many slightly-wrong models cancels out their individual quirks. Forests are more accurate than single trees, much harder to read, and still train in seconds.

Use `sklearn.ensemble.RandomForestClassifier` with `n_estimators=200` and `class_weight="balanced"` (which stops it from ignoring rare classes).

### 2.6 Confidence

Your model must output not just a label but how sure it is.

A forest gives you `predict_proba` — the fraction of trees voting for each class. If 180 of 200 trees say `MEMORY_LEAK`, confidence is 0.90.

**Confidence is what your policy table thresholds against**, so it has to mean something. A model that always reports 0.99 is useless even if it's usually right.

### 2.7 Reading a confusion matrix

A grid of actual versus predicted:

```
          predicted
          LEAK  CPU  DISK
actual LEAK  45    2     1
       CPU    3   41     4
       DISK   1    6    39
```

The diagonal is correct. Everything off-diagonal is a mistake, **and the pattern tells you what the model finds confusable.** Here CPU and DISK get mixed up in both directions — worth investigating; leak is cleanly separated.

**Accuracy** is diagonal ÷ total. **Macro F1** averages per-class performance, which matters when classes are unbalanced — accuracy can look great while one class is never predicted at all.

---

## Part 3 — Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyarrow scikit-learn matplotlib torch
```

No cluster, no Docker.

Inputs: a metrics Parquet file, and a labels CSV with start time, end time, workload, fault type, activation pattern, and run identifier.

---

## Part 4 — Building it

### 4.1 The policy table — do this first

The entire recovery brain, deliberately a lookup table:

| Failure class | Action | Confidence required |
|---|---|---|
| CPU_HOG | scale out | 0.50 |
| MEMORY_LEAK | rolling restart | 0.80 |
| DISK_STRESS | alert only | 0.90 |
| NORMAL | nothing | unreachable |
| UNKNOWN | nothing | unreachable |

Below 0.50 confidence, relabel anything as `UNKNOWN` and do nothing.

**Why not learn this?** Because an operator must be able to answer "why did this restart my pod at 3am." A table answers that; a policy network doesn't. Interpretability here is a product requirement.

**Why different thresholds per action?** Because being wrong costs different amounts:

- Adding a pod wrongly costs pennies → act on weak evidence
- Restarting wrongly drops capacity → wait until fairly sure
- Shedding load wrongly turns away users → require near-certainty

> This design has a consequence worth understanding, and it connects directly to your headline result. Early in a failure, confidence is low, so only the cheap hedging action clears its bar. Later, as the diagnosis firms up, the expensive action unlocks. **The same failure gets handled differently over its lifetime, automatically** — no special logic, it falls out of the thresholds.

### 4.2 Building the dataset

For each 20-tick window: summarize into 40 features, look up which fault (if any) was active at that timestamp, and record the label, the run identifier, and seconds until failure.

Before training anything, **check the label distribution**. If it's 99% NORMAL, something is wrong upstream — go talk to Shaurya.

### 4.3 The mistake that will silently ruin everything

**Do not use `train_test_split`.**

Your windows slide one tick at a time, so **consecutive windows share 19 of their 20 timesteps.** They are near-duplicates.

A random row-level split puts near-identical rows on **both** sides. The model sees essentially the test set during training. You will report 0.99 accuracy and it will mean **absolutely nothing.**

**Split by run.** Train on the `constant`-pattern runs, test on the `ramp`-pattern runs. That's what the run identifier in Shaurya's labels file exists for.

**If your accuracy comes out above ~0.97, assume you made this mistake and check before celebrating.** This is the single most common way student ML projects produce meaningless numbers, and it produces numbers that look like success.

This class of bug is called **data leakage**: information from the test set reaching the model during training. Two other forms to avoid:

**Normalization leakage.** If you scale features using statistics computed over the whole dataset, the training process has seen the test set's distribution. Compute mean and standard deviation on healthy data only, then apply them everywhere. Fault windows have different statistics precisely *because* they're faults — that's the answer, and it must not be in the input.

**Label leakage.** Don't include features derived from the label. Obvious once stated, easy to do accidentally.

### 4.4 Train and evaluate

Fit the forest on the training runs, predict on the held-out runs, print a classification report and a confusion matrix.

Ship the model file to Shravan as soon as it beats his stub. Agree the signature and write it in the README:

```python
label, confidence = classifier.predict(window)
```

---

## Part 5 — The neural network (optional)

**Only after the forest works.** If it doesn't, skip this entirely — it's the most expendable item in the project and nothing downstream depends on it.

### 5.1 Why a 1D convolution

The forest sees your 40 summary features and loses the actual sequence. A **1D convolutional network** operates on the raw 20×8 window, treating metrics as channels and time as the sequence — so it can learn temporal shapes you didn't hand-engineer.

Roughly: two convolution layers with ReLU activations, average-pool across time, then a linear layer to the number of classes. `torch.softmax` on the output gives class probabilities, and the maximum is your confidence.

Normalize with healthy-data statistics only (§4.3).

### 5.2 The comparison table — this is a deliverable

| Model | Accuracy | Macro F1 | Trains in | Readable? |
|---|---|---|---|---|
| Decision tree (Shravan) | | | 1s | Yes |
| Random forest (you) | | | 5s | Partly |
| 1D CNN (you) | | | 3 min | No |

**If the tree wins or ties, ship the tree and write that in the README.** Nobody will think less of you — the opposite. "We tried the sophisticated thing and the simple thing was just as good" is what separates engineering from résumé-driven development.

---

## Part 6 — The two experiments that matter

### 6.1 Abstention

Every published classifier trains on N classes and is graded on the same N. **In production, failures arrive that were never injected.** A model that confidently mislabels a novel fault triggers the wrong remediation — automatically, fast, and worse than doing nothing.

The experiment:

1. Train on `CPU_HOG` and `MEMORY_LEAK` only
2. Test on `DISK_STRESS`, held out entirely — the model has never seen it
3. Measure two numbers:
   - **Correctly abstained** — confidence below 0.50, so `UNKNOWN`
   - **Confidently wrong** — confidence above 0.80 on a class it has never seen

**The second number is the dangerous one.** Those are exactly the cases where Prodrome would take a real, irreversible action based on a fault it doesn't understand. Report it prominently.

Almost no paper does this. It's cheap, and it's the thing an on-call engineer would care about most.

### 6.2 Accuracy versus lead time — the headline result

Bucket your test windows by how many seconds remained until failure, and measure accuracy within each bucket:

```
seconds before failure: 300  180   60   10
classification accuracy: ?    ?    ?    ?    ← should rise
```

Overlay Sagar's detector firing rate on the same axis.

This is **the finding of the whole project.** It quantifies the tradeoff nobody writes down:

> **The earlier you act, the less you know about what you're acting on.**

And it's the *empirical justification* for your per-action confidence thresholds. Early, confidence is genuinely low — so only the cheap action should be available. Later, confidence is high — so the decisive action unlocks. **The plot proves the design was principled rather than arbitrary.**

Make it the headline figure in the README.

### 6.3 Feature importances — 20 minutes, high value

`rf.feature_importances_` ranks which features the forest relies on.

**Sanity check:** if `mem_pct_slope` is the top feature for `MEMORY_LEAK`, the model learned the right thing. If some unrelated network feature dominates, **something is wrong with your labels** and you want to know now.

**Explainability:** ship the top three features in every decision, so the log says *why*:

```
MEMORY_LEAK (0.88) — driven by: mem_pct_slope, mem_bytes_max, cpu_cores_mean
```

That's what makes the log auditable, and it's exactly what closed commercial products can't do — researchers evaluating one complained specifically that its scaling decisions were impossible to interpret.

---

## Part 7 — What will go wrong

| Symptom | Cause and fix |
|---|---|
| Accuracy above 0.97 | **Almost certainly a row-level split.** Split by run |
| One class never predicted | Class imbalance, or those labels never landed. Print the distribution first |
| Confidence always ~0.99 | Overfitting, or normalization stats leaked from the fault set |
| Everything predicted NORMAL | Most windows *are* normal. Filter to windows where Sagar's detector fired — that's the only distribution you'll see in production |
| NaN in your features | `polyfit` on a constant column. Fill upstream, skip zero-variance metrics |
| Your model loses to the tree | Report it. That's a result, not a failure |
| Great on generated data, bad on real | Real faults overlap more. Expected — note the drop in the results |
| Test accuracy varies wildly between runs | Too few test runs. Use more held-out runs, or cross-validate by run |

---

## Part 8 — Working with Sagar

You're paired. Two concrete interfaces:

**The shared feature file.** He owns the metric order and windowing; you add summarization. Talk before editing — if the metric order changes, every array silently misaligns.

**His firing timestamps.** You need them for the accuracy-versus-lead-time curve. He replays the detector over stored data and gives you a file of firings; you bucket your accuracy against them.

Also worth coordinating: **his detector filters the distribution you see.** In production you only classify windows where he fired. If you train on all windows including quiet ones, you're training on a distribution that never occurs at inference. Consider filtering your training set the same way and reporting both.

---

## Part 9 — Later phases

**Compound faults.** A leak during a spike. This is where classification stops being trivial and a learned model finally earns its keep. Report single and compound accuracy separately.

**Network fault discrimination.** Delay versus packet loss is the genuinely hard pair — both raise latency, and only the error-rate pattern separates them. The one place a simple rule should clearly fail.

**A countdown head.** A second output on the same network — class *and* predicted seconds-to-breach, trained jointly. Labels come free from Shaurya's `seconds_to_failure` column. This turns your lead-time curve from an observation into a feature.

**Open-set recognition.** Your abstention is a confidence floor, which is a crude proxy. Distance-based and energy-based novelty scores do meaningfully better at spotting genuinely unfamiliar inputs.

**Phase 6 — the learned policy.** Your hand-written table becomes both the benchmark an RL agent must beat and the policy it bootstraps from. You'd own the agent and the reward function.

The reward function is the critical artifact, and there's a cautionary published result to learn from: an open-source RL autoscaler achieved five-times-faster scaling while running roughly 80% more replicas, because nothing in its reward made resources expensive. Your reward needs an explicit cost term and an action-count penalty, or the agent learns to twitch and to over-provision.

Honest expectation: with a handful of classes and actions, the optimal policy may simply *be* a lookup table — in which case the agent learns what you already wrote, and that's a legitimate finding.

---

## Definition of done

- [ ] Policy table with justified per-action thresholds
- [ ] Feature summarization producing 40 features, slope included
- [ ] Dataset built with labels, run IDs, and seconds-to-failure
- [ ] **Split by run**, verified
- [ ] Random forest trained and evaluated on held-out runs
- [ ] Comparison against Shravan's decision tree, reported honestly
- [ ] Confusion matrix
- [ ] Abstention experiment — both numbers
- [ ] Accuracy-versus-lead-time plot
- [ ] Feature importances checked for sanity and shipped in decisions
- [ ] Model file running in Shravan's live loop
