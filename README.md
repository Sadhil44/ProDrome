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
| Testing and CI | Aahan | [docs/guides/aahan.md](docs/guides/aahan.md) |

Full spec: [PRD.md](PRD.md). Ground rules and conventions: [SETUP.md](SETUP.md) §10.

## The team forum — how to connect

We coordinate through a shared CCP forum that both people and coding agents read and write.
It matters here because §8 has the five areas exchange **files, not services**, so nothing
otherwise warns you when someone changes a column, a metric order or a threshold.

**The current URLs live in [`forum/CURRENT-URLS.md`](forum/CURRENT-URLS.md), with a timestamp
saying when they were last verified.** Read the timestamp before trusting it — see
*If it's dead* below. The session name is `prodrome` and never changes.

There are four ways in, depending on what you want:

### 1. Just look at it — browser, nothing to install

Open the **web view** URL from `forum/CURRENT-URLS.md`. It shows who is working on what right
now, the activity feed, and every entry with its full history. Refreshes every few seconds.
Start here if you only want to know what's happened.

### 2. Command line — if you already have `ccp-client`

```sh
ccp-client subscribe prodrome --server <FORUM-API-URL>
ccp-client brief-me prodrome              # what's on the board
ccp-client master-instructions prodrome   # the rules
```

### 3. New machine — the installer

```sh
sudo apt install -y python3-venv python3-pip     # do this FIRST, see below
export CCP_AGENT_NAME=<yourname>-<aspect>        # e.g. sagar-signal
curl -fsSL <WEB-VIEW-URL>/setup-client.sh | sh
```

Then **restart Claude Code / Codex**. This installs `ccp-client`, subscribes it, and registers
an MCP server named `ccp-forum`.

> **Install `python3-venv` and `python3-pip` before running the installer.** A stock Ubuntu WSL
> image has neither, and without them the MCP's virtualenv is built empty and the install
> **fails silently** — you end up with a working `ccp-client` next to a `ccp-forum` MCP that
> never loads, and nothing in the output says why. Linux/macOS/WSL only; the client needs
> glibc 2.38+ (Ubuntu 24.04+).

### 4. Point your agent at it

Once the MCP is registered, start an agent task with:

> Use the `ccp-forum` MCP, session `prodrome`. You are the **signal** agent
> (or `cluster` / `collect` / `diagnosis` / `testing`). Read `master_instructions` first
> and follow it.

You don't need to explain the forum to it — the rules are on the board, and the `CLAUDE.md`
in each directory carries the coordination line, so an agent working in `ml/` or `control/`
picks it up without being told.

### How the board is organised

One shelf per area — `cluster`, `collect`, `signal`, `diagnosis`, `testing` — each with
`updates` (progress) and `interfaces` (the contracts other areas consume). The shared
`crosstalk` shelf carries `announcements`, `blockers`, `decisions` and `questions`. The
condensed project plan is `prodrome-plan-v1` in `crosstalk/decisions`.

### If it's dead

Likely, and not your fault. The server runs on Sadhil's laptop behind Cloudflare **quick**
tunnels, which are ephemeral by design: every restart mints a new random hostname, and they
have rotated as often as every 30–60 minutes. A watchdog reopens the tunnel and republishes
the board automatically, so the forum recovers on its own — but the new hostname still has to
reach you out of band, because a URL you cannot resolve is a board you cannot read.

So: if `forum/CURRENT-URLS.md` is more than an hour old, or the URL just doesn't resolve,
**ping Sadhil** rather than assuming the forum is gone. It also only runs while his laptop is
awake. Full setup, hosting and troubleshooting: [`forum/HANDOFF.md`](forum/HANDOFF.md) and
[`forum/README.md`](forum/README.md).

## What it doesn't do

No cross-service root cause analysis. No log or trace analysis — metrics only. Not a general replacement for the horizontal autoscaler. No hosted service. No claim of novelty — this is a reimplementation and extension of published research, and says so.
