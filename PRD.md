# Prodrome

## Product Requirements Document

**Predict, diagnose, and heal Kubernetes workloads before they fail.**

Status: Draft · Team of four · Open source, Apache 2.0

---

## 1. Summary

Prodrome is a Kubernetes controller that watches workloads, forecasts that one is heading for failure, classifies what kind of failure it is, and applies the matching remediation through the Kubernetes API.

Kubernetes already self-heals, but only reactively and only one way: when a health probe fails, it restarts the pod. It cannot distinguish a memory leak from a CPU spike from a saturated connection pool, so it applies the same blunt fix to all three — and it waits until users are already affected.

Prodrome adds two things Kubernetes lacks: **lead time** and **diagnosis**.

---

## 2. Problem

### 2.1 Reactive healing is late

Detection, decision, and pod startup together take roughly a minute. A reactive system spends that minute *after* the failure, so users absorb the damage. A predictive system spends it before, and capacity or a fresh process is ready when the failure would have landed.

The resource being spent is not speed — it is lead time. That distinction is the core insight of the product.

### 2.2 One fix does not fit all failures

| Cause | Kubernetes' response | Correct response |
|---|---|---|
| Memory leak | Restart | Restart — correct by accident |
| CPU saturation under load | Restart | Scale out. Restarting removes capacity mid-saturation |
| Connection pool exhaustion | Restart | Shed load. Scaling adds connections and worsens it |
| Disk pressure | Restart | Alert. Restarting changes nothing |

A liveness probe knows only alive or dead. Three of the four rows above are cases where the default response is useless or actively harmful.

### 2.3 Existing solutions do not fill the gap

Three lines of work each solve part of this, and the shape of what they leave out is what defines Prodrome.

**Production predictive scaling** ships at AWS and Google. It forecasts load and provisions ahead of it, but it has exactly one available action — add capacity. It needs no diagnosis because it has no choice of lever.

**Reinforcement-learning autoscaling** closes the loop that pure forecasting leaves open. Forecast-driven autoscalers using recurrent models improve on threshold-based scaling, but they are open-loop: they predict future demand without any mechanism to learn from the real-world impact of their scaling decisions. RL replaces the fixed mapping from forecast to replica count with a learned one.

Two representative results:

- Chinnam and Karanam integrate Q-learning with node autoscaling on a managed cloud cluster, reporting a 34% infrastructure cost reduction at 99.7% availability. The results come from a private production deployment with no public artifact, so the numbers are directional rather than reproducible.
- NimbusGuard is open source and reproducible: a Dueling DQN whose state vector is augmented by an LSTM workload forecaster, with three discrete actions — scale down, hold, scale up — benchmarked directly against HPA and KEDA.

NimbusGuard's headline result is the instructive one. It scales roughly five times faster than the reactive baselines and runs roughly 80% more replicas to do it. The authors name the trade-off directly: proactive, performance-focused scaling versus the reactive, cost-efficient stability of traditional autoscalers.

**It did not beat the autoscaler. It bought latency with money** — and it did so because the reward function under-weighted resource cost relative to responsiveness.

**Academic failure prediction** stops at the alarm. It predicts, reports lead time per fault type, and does not act.

### 2.4 What every one of them shares

**A single lever.** Scale up, or scale down. That is why none of them needs diagnosis, and it is why their learned policies optimize a one-dimensional trade-off between latency and cost.

Prodrome's action space contains several levers where the wrong choice is actively harmful, per the table in §2.2. The moment scaling stops being the only option, "how much" becomes "which," and a system with no diagnosis has no basis for answering it.

**Prodrome's contribution is the closed loop over a widened action space:** predict, diagnose, act, and measure whether acting beat the default. Predictive autoscaling is the degenerate case of this system where the action space has one element.

---

## 3. Goals and non-goals

### 3.1 Goals

- **G1** — Detect that a workload is heading for failure, with measurable lead time before the failure occurs
- **G2** — Classify the failure type accurately enough to select the correct remediation
- **G3** — Apply remediation automatically and safely, with per-action confidence thresholds
- **G4** — Abstain rather than guess when the situation is unrecognized
- **G5** — Demonstrate measurable improvement over stock Kubernetes healing under identical conditions
- **G6** — Be installable and trustworthy enough that an operator will run it on a real cluster

### 3.2 Non-goals

- **NG1** — Not a general replacement for the horizontal autoscaler. Workloads that only need CPU-based scaling should use the built-in one
- **NG2** — No cross-service root cause analysis. If A degrades because B is slow, Prodrome flags A
- **NG3** — No log parsing or trace analysis. Metrics only
- **NG4** — No hosted service. Self-hosted, no phone-home
- **NG5** — No claim of novelty. This is a reimplementation and extension of published research, and says so

---

## 4. Users and scenarios

### 4.1 Primary user

A platform or SRE engineer running services on Kubernetes, who is currently paged for failures that had visible precursors in metrics nobody was watching.

### 4.2 Scenarios

**S1 — Gradual resource exhaustion.** A service leaks memory over hours. Prodrome detects the trend well before the limit, classifies it, and schedules a rolling restart during low traffic. The operator sees an entry in the decision log and no incident.

**S2 — Wrong-fix prevention.** A service saturates its database connection pool. Prodrome classifies it and *withholds* the scale-out action, which would add connections to an already-drowning database, alerting instead.

**S3 — Unrecognized failure.** A novel failure mode with no matching training class. Prodrome's detector fires, the classifier's confidence falls below the abstention floor, no action is taken, and a human is paged with the anomaly context.

**S4 — Instant failure.** A pod is killed outright. Prodrome provides no warning, exactly like Kubernetes, and the default reactive healing handles it. **This is expected and documented.**

**S5 — Evaluation.** An operator runs Prodrome in shadow mode for two weeks, reads the log of actions it would have taken, and then grants it authority one action at a time.

---

## 5. Functional requirements

### 5.1 Detection

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Collect per-container resource metrics without requiring instrumentation of the workload | Must |
| FR-2 | Learn a per-workload model of normal behaviour from failure-free data only | Must |
| FR-3 | Emit a firing signal when a workload deviates from normal in a sustained way | Must |
| FR-4 | Suppress detection during and shortly after workload restarts | Must |
| FR-5 | Estimate seconds remaining until failure | Should |
| FR-6 | Detect drift in its own error distribution and trigger a refit | Could |

### 5.2 Diagnosis

| ID | Requirement | Priority |
|---|---|---|
| FR-7 | Classify a firing into one of a defined set of failure types with a confidence score | Must |
| FR-8 | Emit UNKNOWN when confidence falls below the abstention floor | Must |
| FR-9 | Report the features that most influenced each classification | Must |
| FR-10 | Support a user-defined failure taxonomy | Could |

### 5.3 Remediation

| ID | Requirement | Priority |
|---|---|---|
| FR-11 | Map each failure type to exactly one action via declarative configuration | Must |
| FR-12 | Enforce a distinct confidence threshold per action | Must |
| FR-13 | Support scale-out, rolling restart, and alert-only | Must |
| FR-14 | Enforce a cooldown between actions on the same workload | Must |
| FR-15 | Enforce a replica ceiling and floor | Must |
| FR-16 | Provide a kill switch that halts all action immediately | Must |
| FR-17 | Operate in shadow, advise, or act mode, defaulting to shadow | Must |
| FR-18 | Roll back an action automatically if metrics worsen after it | Should |
| FR-19 | Support user-defined action plugins | Could |

### 5.4 Observability and configuration

| ID | Requirement | Priority |
|---|---|---|
| FR-20 | Log every evaluation with score, firing state, class, confidence, driving features, chosen action, and outcome | Must |
| FR-21 | Expose per-workload configuration declaratively as a Kubernetes resource | Must |
| FR-22 | Collect its own baseline automatically on install, with no manual training step | Should |
| FR-23 | Provide a dashboard showing live state and decision history | Should |

---

## 6. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | Evaluation cycle completes within the metric scrape interval |
| NFR-2 | Controller permissions scoped to reading and patching deployments and their scale — nothing else |
| NFR-3 | Adding no instrumentation to observed workloads |
| NFR-4 | Every reported result reproducible from a fresh clone |
| NFR-5 | Failure of Prodrome degrades the cluster to stock Kubernetes behaviour, never worse |
| NFR-6 | Installation to first prediction achievable in an afternoon |
| NFR-7 | Apache 2.0, no proprietary dependencies |

**NFR-5 is the load-bearing one.** Prodrome is an addition to a self-healing system. If it crashes, misbehaves, or is uninstalled, the cluster must fall back to exactly what it did before.

---

## 7. System design

### 7.1 Pipeline

Four stages, evaluated per workload on each scrape interval:

| Stage | Question | Nature | Training data |
|---|---|---|---|
| Detector | Is this abnormal and heading for failure? | Unsupervised | Healthy data only |
| Classifier | What kind of failure? | Supervised | Injected faults |
| Policy | What do we do, and are we sure enough? | Lookup table | None |
| Controller | Execute | API client | None |

### 7.2 Design decisions and rationale

**The detector never sees a fault during training.** Two reasons. Real failures cannot be enumerated in advance, so a detector that recognizes only known faults is useless against the one that takes you down. And injected faults are step functions while real degradation is a slide — training on the former produces a model that detects your injector.

Fault injection is therefore the *test* set, permanently.

**Two models rather than one.** The detector alone says something is wrong without indicating which of several mutually-harmful fixes to apply. The classifier alone identifies the problem only once it is already breaking. Together they support "memory leak, roughly twenty minutes out, restart now while it is cheap."

**The policy layer is not learned.** An operator must be able to answer "why did this restart my pod at 3am." A table answers that; a policy network does not. Interpretability here is a product requirement, not an engineering preference.

**Per-action confidence thresholds.** Being wrong costs different amounts by action. Scaling wrongly costs money; restarting wrongly costs capacity; shedding load wrongly costs users. Cheap actions may act on weak evidence; expensive actions wait.

This has a useful emergent property: **the same failure is handled differently over its lifetime.** Early, when confidence is low, only the cheap hedging action is available. Later, as the diagnosis firms, the decisive action unlocks. No special logic is needed — it falls out of the thresholds.

**The size of the action space changes the nature of the problem.** With one lever, the only questions are how much and when — a forecasting problem, and the reason production predictive scaling ships without any classifier. With several levers where the wrong choice worsens the incident, the question becomes which — a diagnosis problem. This is why Prodrome carries two models where the autoscaling literature carries one, and it is also why Prodrome's eventual learned policy is a harder problem than the published RL autoscalers: the action space is larger and its errors are asymmetric.

**Cost must be explicit in any objective.** The clearest lesson available from the RL autoscaling work is what happens when it is not — NimbusGuard's agent achieved its responsiveness by running substantially more replicas, because nothing in its reward made that expensive. Prodrome encodes the same discipline by hand in Phase 4, as per-action confidence thresholds weighted by the cost of being wrong, and must carry it explicitly into the reward function if the policy is ever learned.

**Abstention is a first-class output.** Every published classifier trains on N classes and is graded on the same N. In production, unfamiliar failures arrive, and a confidently mislabeled novel fault triggers the wrong remediation automatically and quickly — worse than inaction.

---

## 8. Interfaces and data contracts

Two contracts define every internal boundary. They are agreed before implementation and changed only by explicit agreement; everything else is an implementation detail its owner may change freely.

**The metric window** — one row per workload per interval, carrying the agreed metric set, assembled into a fixed-length sliding window. Consumed by both models.

**The decision** — workload, firing state, detector score, predicted class, confidence, seconds to breach, driving features, chosen action, mode, and outcome. Produced by the pipeline, consumed by the controller, the log, and the evaluation harness.

**Workstreams exchange files, not network calls** — datasets, model artifacts, decision logs. If one workstream's output is unavailable, the others fall back to synthetic or committed sample data and continue. Service boundaries are introduced at packaging, once interfaces have stabilized.

---

## 9. Success metrics

### 9.1 Product metrics

| Metric | Definition | Target |
|---|---|---|
| Lead time | Seconds between first firing and failure, per fault type | Positive and reported per type |
| Recall | Fraction of injected faults detected before failure | Comparable to published work |
| Precision | Fraction of firings corresponding to real faults | Comparable to published work |
| False positives per hour | Firings during clean runs | Low enough that actions are not disruptive |
| Classification accuracy | Against held-out runs, versus a simple baseline | Must beat the baseline or the baseline ships |
| Abstention rate | Correct UNKNOWN on a held-out fault type | Reported, high |
| Time to recovery | Failure to healthy, versus the control arm | Lower than control |
| Resource cost | Pod-seconds consumed over an experiment, versus the control arm | Not materially higher |

The last row is deliberate. The published RL autoscaling result achieved responsiveness at roughly 80% more replicas than the reactive baseline, which is a trade rather than a win. Any Prodrome result that improves recovery time while consuming substantially more resources must report both numbers together and be described as a trade-off, not an improvement.

### 9.2 Reporting discipline

**Always per fault type, never one aggregate.** An average lead time hides that one class gets twenty minutes and another gets none, and the cases where the method does not work are the most informative part of the result.

**Always against the control arm.** An identical deployment with no Prodrome, taking the same faults under the same load. Kubernetes already self-heals, so "we recovered the service" means nothing without "and the control arm did not."

**Calibrate expectations.** Published work on this problem reports roughly 0.65 precision with 0.92 recall, and lead times from fifteen minutes to over two hours by fault type. A third of predictions being false alarms is the state of the art. Accuracy far above that on classification usually indicates a methodological error, not a breakthrough.

### 9.3 Adoption metrics

Time from install to first useful prediction. Number of operators running past shadow mode. Number who grant a second action.

---

## 10. Phases and release criteria

Phases advance on demonstrated conditions, not dates.

| Phase | Goal | Exit criterion |
|---|---|---|
| 0 — Scaffolding | Every component buildable in isolation | Any one component builds and tests with no cluster and no teammate |
| 1 — Skeleton | The loop runs end to end, badly | A hand-injected fault produces a logged decision |
| 2 — Grounded | Real data replaces synthetic | The pipeline runs on real data and the detector fires on real faults |
| 3 — Learned | Models beat their baselines, or baselines ship | A published comparison table, better model in the loop |
| 4 — Acting | The loop closes and the claim is measured | A defensible comparison against stock Kubernetes |
| 5 — Packaged | Installable by a stranger | An external operator installs it and reports back |
| 6 — Learned policy | The policy table is learned rather than written | The learned policy matches or beats the hand-written table in simulation and in shadow |

**Phase 0 determines whether the project is pleasant or miserable.** Its central artifact is a synthetic data generator emitting the real schema — realistic load variation plus injectable faults that ramp toward failure. With it, model work begins before any infrastructure exists, and later integration is a path change rather than a first meeting.

**Phase 4 is where the project first has a result rather than a demo.**

### 10.1 Phase 6 in detail — the learned policy

This phase exists in the plan from the beginning, because every earlier phase quietly produces its training data. It is written down here so that nothing is retrofitted later.

**Why it can be attempted at all.** Reinforcement learning has not shipped in production autoscaling for one reason: an agent learns by taking bad actions, and on live traffic a bad action is an outage. Every published attempt adds guardrails — bootstrapping from an existing controller, circuit breakers, canaries — and each guardrail removes the exploration that motivated RL in the first place.

**Prodrome removes that blocker structurally.** The fault-injection harness built in Phase 2 is a disposable environment we already break on purpose. Exploration is free there — real system dynamics, no users affected.

**Reference architecture.** NimbusGuard is the closest published system and the natural starting point: a Dueling DQN with an experience replay buffer, a state vector augmented by an LSTM forecast, and a discrete action space, running on a fixed decision interval against a Prometheus-fed observation. Prodrome's version differs in three ways, each following from earlier phases.

| | Published RL autoscalers | Prodrome |
|---|---|---|
| State | Resource metrics plus a load forecast | The same, plus predicted failure class, classification confidence, and seconds to breach |
| Actions | Scale down, hold, scale up | The same, plus rolling restart and shed load |
| Reward | Latency and utilization | The same, plus an explicit resource-cost term and an action-count penalty |

**The reward terms are not incidental.** The cost term exists because the published result shows what its absence produces — an agent that achieves responsiveness by running far more replicas than the baseline. The action-count penalty exists because without it the agent learns to twitch, and a flapping cluster is worse than a slow one.

**The safety ordering, which is the entire reason this is Phase 6 and not Phase 3:**

1. **Bootstrap** from the hand-written policy table so the agent begins competent rather than random. The table from Phase 4 is both the initialization and the benchmark
2. **Offline first.** Every decision logged since Phase 1 is a state, action, and reward tuple. Batch-train on those with no exploration at all
3. **Simulate.** A fast environment behind the same interface as the real cluster, so training runs in seconds rather than in wall-clock fault cycles. The gap between simulated and real performance is itself a reportable result
4. **Then online, chaos environment only.** Never against a workload anyone depends on
5. **Ship in shadow.** The learned policy proposes, the hand-written table decides, and the disagreements are logged and read

**Honest expectation.** With a handful of failure classes and five actions, the optimal policy may simply *be* a lookup table — in which case the agent learns what was already written, and the result is that the hand-written policy was correct. That is a legitimate finding and a better report than an overclaim.

**The prerequisite, and it starts in Phase 1:** the controller logs every evaluation as a state, action, and reward tuple. It costs almost nothing at the time and it means this phase opens with a dataset rather than a blank slate. This is the single most droppable-looking item in the early phases and it must not be dropped.

---

## 11. Team and per-person workflow

Four permanent workstreams, one owner each. They exist at every phase and never merge, because each has a distinct failure mode: Signal fails by crying wolf, Diagnosis by confidently mislabeling, Policy by acting wrongly, Platform by producing numbers nobody can reproduce. Separation keeps each failure independently diagnosable.

The team works as two pairs. Sagar and Sadhil pair on Signal and Diagnosis; Shaurya and Shravan pair on Platform. Pairs sit together and share context freely within their half; the two halves synchronize on the data contracts in §8 and, from Phase 6, on the environment interface described below.

**Nobody blocks anybody.** Phase 0 produces a synthetic data generator and a stub for every component. From then on each workstream can be built and tested standalone. The Signal and Diagnosis workstreams never require a cluster at all — their input is a file. Handoffs exist, but each has a fallback to a Phase 0 artifact, so a late handoff costs quality rather than progress.

### 11.1 Shravan — Platform: cluster and control

**Owns** the cluster, the observed workloads, resource limits, the recovery controller, safety rails, permissions, packaging, and the one-command developer experience.

**Requirements owned:** FR-11 through FR-19, FR-21, NFR-2, NFR-5, NFR-6.

**Position:** both ends of the pipeline. The cluster produces everything the models learn from, and the controller is the only component that changes the world.

**Recurring loop**

1. Declare desired workload state and resource limits — **the memory ceiling is what makes failures happen at all**, so without it there is nothing to predict
2. Build a controller action, test it in isolation before wiring it into anything
3. Wrap it in safety rails: cooldown, ceiling, kill switch
4. Run it in shadow mode and read the log before granting authority
5. Verify the control arm is running identical conditions
6. Package so the whole thing reproduces from a fresh clone

**Deliverables by phase**

| Phase | Deliverable |
|---|---|
| 0 | Chaos-runner stub; cluster configuration committed |
| 1 | Cluster with limits and probes; controller loop running on stubs in shadow |
| 2 | Control-arm namespace running identical workloads with no Prodrome |
| 3 | The simple baseline classifier — a shallow readable rule |
| 4 | Controller live, safety rails verified, head-to-head executed |
| 5 | Declarative configuration resource, scoped permissions, one-command install |
| 6 | Real-cluster training environment behind the shared interface; shadow-mode deployment of the learned policy |

**Definition of done:** one command from clone to running system; one command to demo; safety rails verified by test; a decision log a stranger can read.

**Cross-cutting responsibility:** Shravan owns the simple baseline for classification — a shallow, human-readable rule. It exists so the team can tell how much of the sophisticated model is doing real work. If the rule ties the model, the rule ships and the README says so.

### 11.2 Shaurya — Platform: observability, perturbation, evaluation

**Owns** metric collection, load generation, fault injection, labeling, the evaluation harness, and the reported results.

**Requirements owned:** FR-1, FR-20, FR-23, NFR-1, NFR-3, NFR-4.

**Position:** produces the data everyone consumes and the numbers that say whether any of it worked.

**Recurring loop**

1. Generate load with **realistic variation** — constant load teaches the detector that normal means flat, so it fires on the first genuine traffic change. The variation is the training signal
2. Collect the baseline over several full load cycles
3. Inject faults, recording type, intensity, activation pattern, and a run identifier
4. Include clean runs with no fault — **the only source of false-positive measurement, and the first thing teams skip**
5. Label the full trajectory, including faint early windows, with seconds remaining until failure
6. Evaluate: lead time, recall, precision, recovery time, per fault type, against the control arm

**Deliverables by phase**

| Phase | Deliverable |
|---|---|
| 0 | **The synthetic data generator** — the artifact that unblocks everyone else — plus committed samples |
| 1 | Metric collection with no workload instrumentation; scraper verified |
| 2 | Canonical healthy baseline; fault matrix with two activation patterns; clean runs; labeled dataset |
| 3 | Evaluation harness producing the per-fault table |
| 4 | Head-to-head experiment execution; results |
| 5 | Automatic baselining on install; dashboard |
| 6 | Episode replay from the decision log; the cost-versus-latency curve comparing learned policy, hand-written table, HPA, and KEDA |

**Definition of done:** one command prints the per-fault results table including false positives per hour; the canonical datasets are reproducible.

**Two rules that only Shaurya can enforce.** Every fault gets a **ramping** activation pattern as well as a constant one — a constant fault is a step change any threshold catches, and only a ramp makes lead time meaningful. And labels cover the **entire** trajectory: if only the dramatic windows are labeled, the classifier learns to recognize faults solely when they are already catastrophic, and it will fail precisely where the detector fires. That failure spans two people's work and neither can see it from their own side.

### 11.3 Sagar — Signal

**Owns** anomaly detection, the failure forecast, threshold selection, and lead-time measurement.

**Requirements owned:** FR-2 through FR-6.

**Position:** the gate. If Signal does not fire, nothing downstream runs.

**Recurring loop**

1. Fit per-metric models of normal behaviour on **failure-free data only**
2. Confirm at the workload level — several metrics anomalous for several consecutive intervals, because healthy systems throw isolated anomalies constantly
3. Replay against stored fault data to measure recall and lead time
4. Replay against clean runs to measure false positives
5. Sweep parameters, select deliberately, and report both the balanced and the precision-optimal configuration
6. Refit as workloads change

**Deliverables by phase**

| Phase | Deliverable |
|---|---|
| 0 | Shared feature and windowing code; detector stub |
| 1 | The real detector, built and tuned against generated data |
| 2 | Refit on the canonical healthy baseline; thresholds adjusted for real noise |
| 3 | Per-fault recall and lead-time table; false positives per hour |
| 4 | Detector live in the loop; restart suppression verified |
| 5 | Automatic baselining; drift detection |
| 6 | The load forecaster feeding the agent's state vector; the simulated training environment |

**Definition of done:** per-fault recall and median lead time; false positives per hour on clean runs; quiet on held-out healthy data; artifact running live.

**The one rule.** The detector trains only on healthy data — this is what keeps every result defensible. If it starts to feel like it would work better with a few faults mixed in, that is the moment to stop. It is also the rule most likely to be violated by accident: a mislabeled time range, a fault run left in the healthy set. Suspiciously good results are a reason to check timestamps, not to celebrate.

**Never blocked:** no cluster required. The Phase 0 generator supplies healthy signals and ramping faults, so the real detector is built and tuned before any infrastructure exists.

### 11.4 Sadhil — Diagnosis and policy

**Owns** failure classification, feature engineering, abstention, explainability, and the action policy table.

**Requirements owned:** FR-7 through FR-10, FR-12 (thresholds), and the policy content behind FR-11.

**Position:** the selector. Signal decides whether; Diagnosis decides what; the policy table decides what to do about it.

**Recurring loop**

1. Summarize each metric across the window into level, trend, volatility, peak, and current value — **trend is what separates a leak from steady high usage**, and level alone is exactly what a naive threshold sees
2. Train on labeled fault data, **split by run rather than by row**
3. Compare against the simple baseline every time
4. Set the abstention floor and validate it on a fault type held out entirely
5. Maintain the policy table: one action per class, one threshold per action
6. Report feature attribution with every decision

**Deliverables by phase**

| Phase | Deliverable |
|---|---|
| 0 | Policy table; feature summarization design; classifier stub |
| 1 | The real classifier, trained on generated faults |
| 2 | Retrained on canonical labels; comparison against generated-data results |
| 3 | Model comparison table; confusion matrix; feature attribution |
| 4 | Abstention measurement; the accuracy-versus-lead-time curve |
| 5 | Pluggable taxonomy; open-set recognition |
| 6 | The agent, the reward function, and the offline training pipeline seeded from logged episodes |

**Definition of done:** comparison against the simple baseline on held-out runs; confusion matrix; abstention rate on a held-out fault type; attribution in every decision; artifact running live.

**Expectation-setting.** Classification is the easy half. Each injected fault spikes its own metric, so a shallow rule gets most of it right and published systems report roughly 0.95 accuracy. High accuracy here proves very little. The work that matters is: beating the simple baseline or honestly reporting that it tied; handling **compound faults** where a leak and a spike overlap and single-metric rules cannot separate them; and **abstention**, which almost nobody measures and which decides whether the system is safe to run unattended.

**The headline result.** Classification accuracy bucketed by how far ahead you are looking. It quantifies the tradeoff nobody writes down — **the earlier you act, the less you know about what you are acting on** — and it is the empirical justification for per-action confidence thresholds.

**Never blocked:** no cluster required. The Phase 0 generator emits labeled faults, so a real classifier trains before any container has been stressed, and pipeline bugs surface early rather than under pressure.

### 11.5 The environment interface — the seam for Phase 6

Through Phase 5 the two pairs exchange files. Reinforcement learning needs something a file cannot provide: an environment that responds to actions. Phase 6 therefore introduces a second contract.

**One interface, two implementations.**

| Implementation | Built by | Purpose |
|---|---|---|
| Simulated environment | Sagar and Sadhil | A fast model of workload, pods, startup delay, and failure dynamics. Thousands of episodes per hour |
| Real environment | Shaurya and Shravan | The actual cluster and fault harness behind the identical interface |

The agent moves between them unchanged. Training happens in simulation because real fault cycles take minutes and RL needs thousands of them; validation happens on the real cluster. **The gap between simulated and real performance is itself a result worth reporting** — sim-to-real transfer is the standard failure mode of applied RL, and measuring it honestly is more valuable than hiding it.

This also preserves the no-blocking property into the final phase. The ML pair trains against their simulator on their own schedule; the platform pair hardens the real environment on theirs.

---

## 12. Shared workflow

### 12.1 The core cycle

Every capability the project gains — a fault type, an action, a model, a workload kind — moves through these stages in order. Finishing the last starts the first again.

**Baseline → Fit → Perturb → Learn → Decide → Act → Measure → Iterate**

A new fault type re-enters at Perturb. A new model at Fit or Learn. A new action at Decide. **Nothing skips Measure.**

### 12.2 Standing rules

1. The detector never trains on injected faults
2. Every claim ships with the control-arm comparison
3. A simple baseline is built and reported before any sophisticated model is credited
4. New models and actions land in shadow mode and are read before being trusted
5. Train/test splits are by run, never by row
6. Results are reported per fault type
7. What the system cannot do is stated explicitly
8. Any result not reproducible from a fresh clone is not a result

### 12.3 The standing review question

For every change, experiment, and merge:

> **What is the baseline, and did you beat it?**

If there is no baseline, that is the first piece of work. If the baseline won, that is a result — report it and move on.

### 12.4 Ways of working

**Contracts before code.** The two data contracts are agreed before implementation and changed only by explicit agreement.

**Files, not services.** Workstreams exchange artifacts. If one is unavailable, the others fall back to Phase 0 samples.

**Everyone runs their own stack.** Shared environments create queues. Cluster configuration is committed so environments match.

**Canonical versus local.** One environment produces the reported numbers so results trace to a consistent source. Everyone develops against their own data freely. Canonical is about result integrity, not permission.

**Integrate on a rhythm.** Run the full loop end to end regularly and treat a broken loop as top priority. Long gaps turn small interface drifts into multi-day merges.

---

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Detector trained on fault data by accident | Results indefensible | Strict separation; suspicious accuracy triggers a timestamp audit |
| Row-level train/test split | Meaningless accuracy | Split by run; accuracy above the expected range triggers review |
| Labels cover only late windows | Classifier fails where the detector fires | Label full trajectories; spans two owners so it is called out explicitly |
| Control arm omitted or added late | No defensible claim | Stood up in Phase 2, before any results exist |
| False positives make actions disruptive | Worse than no tool | Clean runs measure it; per-action thresholds; shadow default |
| Novel fault confidently mislabeled | Wrong remediation, automatically | Abstention, measured on a held-out fault type |
| Sophisticated model no better than a rule | Wasted effort | Baseline built first; ship the rule and say so |
| Chasing the easy half | Impressive numbers, little value | Success metrics weight lead time, false positives, and abstention |

---

## 14. Out of scope

Cross-service root cause analysis. Log and trace analysis. Replacing the horizontal autoscaler generally. A hosted service. Workload types beyond deployments, initially.

---

## 15. Open questions

1. What abstention floor balances safety against inaction, and should it vary by action?
2. Should rollback be automatic or require approval?
3. Does the simple baseline beat the learned model on single faults but lose on compound ones — and if so, is the right answer to ship both?
4. What is the minimum baseline collection period before predictions become useful, and can it be detected automatically rather than configured?
5. Does the detector's per-metric approach hold up on workloads with many more metrics, or does confirmation logic need to change?
6. What weighting between latency, resource cost, and action count produces a learned policy that is genuinely better rather than merely faster and more expensive?
7. How large is the sim-to-real gap for the learned policy, and is a simulator accurate enough to train in even achievable at this scale?
8. If the learned policy converges to the hand-written table, is that a null result or a validation of the table?
