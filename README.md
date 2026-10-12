# Prodrome

**Predict, diagnose, and heal Kubernetes workloads before they fail.**

Kubernetes already self-heals, but only reactively and only one way: when a health probe fails, it restarts the pod. It can't tell a memory leak from a CPU spike from a saturated connection pool, so it applies the same blunt fix to all three — and it waits until users are already affected.

Prodrome adds the two things Kubernetes lacks: **lead time** and **diagnosis**. It watches workloads, forecasts that one is heading for failure, classifies what kind of failure it is, and applies the matching remediation through the Kubernetes API — measured against stock Kubernetes on identical faults.

## Why not just autoscale?

Production predictive scaling (AWS, Google) and RL-based autoscalers all share one property: a single lever, scale up or down. That's why none of them need diagnosis. Prodrome's action space has several levers where the wrong choice is actively harmful:

| Cause | Kubernetes' response | Correct response |
|---|---|---|
| Memory leak | Restart | Restart — correct by accident |
| CPU saturation under load | Restart | Scale out. Restarting removes capacity mid-saturation |
| Connection pool exhaustion | Restart | Shed load. Scaling adds connections and worsens it |
| Disk pressure | Restart | Alert. Restarting changes nothing |

The moment scaling stops being the only option, "how much" becomes "which," and a system with no diagnosis has no basis for answering it.

## How it works

Four stages, evaluated per workload on each scrape interval:

| Stage | Question | Nature | Training data |
|---|---|---|---|
| Detector | Is this abnormal and heading for failure? | Unsupervised | Healthy data only |
| Classifier | What kind of failure? | Supervised | Injected faults |
| Policy | What do we do, and are we sure enough? | Lookup table | None |
| Controller | Execute | API client | None |

The detector never sees a fault during training — real failures can't be enumerated in advance, and injected faults are step functions while real degradation is a slide, so training on the former just teaches the model to recognize the injector. The policy layer is a hand-written table, not learned, because an operator must be able to answer "why did this restart my pod at 3am." Full rationale is in [PRD.md](PRD.md) §7.2.

Every claim is measured against a control arm running identical workloads and identical faults with no Prodrome — "we recovered the service" means nothing without "and stock Kubernetes didn't."

## Does it work?

**We don't know yet, and that is the honest answer.** The control-arm comparison — the only thing that could answer it — has been built but not yet run under load. Everything below is what has actually been measured.

### What is measured

The pipeline runs end to end on real data: 11,946 metric samples across 160 labeled fault runs on `nginx`, `postgres` and `redis`, producing 3,276 logged decisions through detector → classifier → policy → controller.

**Detector.** On known-clean windows the false-positive rate is **5.0%**, against a **61.5%** firing rate inside fault windows — a 9.1× separation, or 1.86 false positives per workload-hour. About 40% of all firings are a *post-fault tail* lasting ~300s, not false alarms on healthy traffic. Lead time, for the one class with a true failure instant, is a median of **35.8s** before failure.

**Classifier.** 0.92 accuracy and 0.82 macro F1 on held-out runs. Read the macro number, not accuracy: the test set is 65% `DISK_STRESS`, which classifies at 0.98 F1 and drags the aggregate up. `MEMORY_LEAK` sits at **0.61** and is the weak class.

### What is measured and is bad news

These are the results that matter most, because they bound what the system can currently claim.

**The model never abstains.** Train on two fault types, hold the third out, and ask what it does with a fault it has never seen. On real data it is confidently wrong — above the 0.80 threshold that triggers a real action — on **71.5%** of unseen-fault windows, and correctly abstains on **0%** of them. A confidence floor is not novelty detection. This is the single strongest reason the controller stays in dry-run.

**`CPU_HOG` is diagnosed from the wrong evidence.** Per-prediction attribution puts only **6.0%** of the decision on CPU metrics and ~80% on memory and filesystem. Deleting *every* CPU feature makes `CPU_HOG` classification **better** (F1 0.883 → 0.936, consistent across 7 random seeds). The model has learned the fault injector's memory and I/O footprint, not CPU pressure — so the `CPU_HOG` result is not evidence that Prodrome detects CPU exhaustion, and would not be expected to transfer to real application load.

**A fixed analysis window cannot represent half the fault classes.** The window is 20 samples at 15s = 5.00 minutes. `MEMORY_LEAK` runs median 87.5s and `POD_KILL` 8.7s, so **zero** runs of either are long enough to contain one full window. The two classes with high F1 are exactly the two injected at ~300s — almost precisely the window length. Reported accuracy therefore substantially measures how well each fault's duration happens to match the window, not diagnostic skill.

### Safety

The controller runs with `DRY_RUN = True` and decides without acting. Four defects were found in its safety rails by testing them against an adversarial caller, every one invisible while dry-run was on — including `scale_out` *removing* capacity when a deployment was already above its replica cap, and a failed API call leaving no audit row and no cooldown, so the loop would retry every tick forever with no trace. The rails now run in CI on every push, as part of a 187-test contract suite that runs from a fresh clone with no cluster and no Docker.

## Getting started

Reproducing the analysis needs no cluster and no Docker — just the Python dependencies in [requirements.txt](requirements.txt):

```bash
python -m ml.train        # classifier, split by run: report, confusion matrix, importances
python -m ml.abstention   # the held-out-fault safety numbers
python -m ml.ablation     # is a fault class detected from its own metric?
python -m ml.classifier   # refit and re-pickle the shipped model
python -m dashboard.terminal   # live view over the decision log
```

Running the full loop against a cluster, and the two-arm comparison, needs a kind cluster and the workload manifests in [`infra/`](infra/). [SETUP.md](SETUP.md) has the setup path and the data contracts; [PRD.md](PRD.md) is the full spec.

### Repository layout

| Path | What's in it |
|---|---|
| [`ml/`](ml/) | Detector, classifier, feature extraction, and the analysis tools behind the numbers above |
| [`control/`](control/) | Policy table and the Kubernetes controller, with its safety rails |
| [`collect/`](collect/) | Metric scraping, fault injection, and the two-arm campaign harness |
| [`eval/`](eval/) | Recovery-time measurement and the evaluation harness |
| [`infra/`](infra/) | Cluster config and workload manifests for both the treatment and control arms |
| [`dashboard/`](dashboard/) | Terminal view over the decision log |
| [`tests/`](tests/) | Contract and integration tests; runs from a fresh clone with no cluster |

## Methodology notes

A few conventions that the numbers above depend on, and that are easy to get wrong:

**Split by run, never by row.** Sliding windows share 19 of their 20 samples, so a random split trains the model on near-duplicates of its own test set. Training uses `constant`-pattern runs and tests on `ramp`-pattern runs. Accuracy above ~0.97 on this data should be assumed to be this bug until checked.

**Report per fault type, never one aggregate.** The aggregate hides the class that hasn't improved — which, here, is consistently the one whose remediation is the destructive one.

**`end_ts` is not a failure instant.** For `CPU_HOG` and `DISK_STRESS` the injector stops and the workload recovers; nothing fails. Lead-time and recovery numbers for those classes measured from end-of-injection are meaningless, and are not reported.

**Results must be reproducible from a fresh clone.** Model artifacts are gitignored and refit from the module that owns them; a result that depends on a pickle nobody else can regenerate is not a result.

## What it doesn't do

No cross-service root cause analysis. No log or trace analysis — metrics only. Not a general replacement for the horizontal autoscaler. No hosted service. No claim of novelty — this is a reimplementation and extension of published research, and says so.

Open source, Apache 2.0.
