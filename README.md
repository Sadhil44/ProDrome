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

## Status

Draft, team of four, open source (Apache 2.0). Currently Phase 0 — see [PRD.md](PRD.md) §10 for the phase breakdown and exit criteria.

## Getting started

New to the repo? Start with [SETUP.md](SETUP.md), then read your own guide in [docs/guides/](docs/guides/):

| Area | Owner | Guide |
|---|---|---|
| Cluster, workloads, controller | Shravan | [docs/guides/shravan.md](docs/guides/shravan.md) |
| Prometheus, chaos, evaluation | Shaurya | [docs/guides/shaurya.md](docs/guides/shaurya.md) |
| Detector (Signal) | Sagar | [docs/guides/sagar.md](docs/guides/sagar.md) |
| Classifier and policy (Diagnosis) | Sadhil | [docs/guides/sadhil.md](docs/guides/sadhil.md) |

Full spec: [PRD.md](PRD.md). Ground rules and conventions: [SETUP.md](SETUP.md) §10.

## What it doesn't do

No cross-service root cause analysis. No log or trace analysis — metrics only. Not a general replacement for the horizontal autoscaler. No hosted service. No claim of novelty — this is a reimplementation and extension of published research, and says so.
