# Prodrome — Repo Setup

Read this first. It gets you from nothing to a working environment, and covers the conventions everyone follows.

**What we're building:** a Kubernetes controller that predicts workload failures, classifies what kind, and applies the matching fix — measured against stock Kubernetes on identical faults. See `README.md` for the design and `PRD.md` for the full spec.

**Time to complete:** about 45 minutes.

---

## 1. Prerequisites

| Tool | Why | Who needs it |
|---|---|---|
| **Git** | Obviously | Everyone |
| **Python 3.11+** | Everything is Python | Everyone |
| **Docker** | Runs containers; the cluster lives inside it | Platform pair |
| **kind** | Kubernetes in Docker — throwaway local cluster | Platform pair |
| **kubectl** | Talks to the cluster | Platform pair |
| **helm** | Installs Prometheus | Platform pair |
| **k6** | Load generation | Platform pair |

**The ML pair does not need Docker or a cluster.** Their input is a data file. If you're on Signal or Diagnosis, install Python and stop there — you can do everything else later if you're curious.

### macOS
```bash
brew install git python@3.11 kind kubectl helm k6
# Docker Desktop: https://www.docker.com/products/docker-desktop
```

### Linux
```bash
sudo apt update && sudo apt install -y git python3.11 python3.11-venv

curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64
chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind

curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Windows
Use **WSL2** and follow the Linux instructions inside it. Do not try this in PowerShell.

> **Give Docker at least 8GB of memory** in its settings. The default is often 2GB and the cluster will die in confusing ways.

---

## 2. Get running

```bash
git clone <repo-url> && cd prodrome

python3.11 -m venv .venv
source .venv/bin/activate          # Windows/WSL: same command
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas>=2.0
pyarrow>=14.0
numpy>=1.24
scikit-learn>=1.3
torch>=2.0
matplotlib>=3.7
requests>=2.31
kubernetes>=29.0
rich>=13.0
```

Verify:
```bash
python -c "import pandas, sklearn, torch, kubernetes; print('ok')"
```

---

## 3. Everyone runs their own cluster

**This is deliberate.** Sharing one cluster creates a queue, and a queue means people wait. Creating a cluster takes five minutes and four people can do it simultaneously.

Platform pair only:

```bash
kind create cluster --name prodrome --config infra/kind-config.yaml
kubectl get nodes                  # expect 3 nodes, all Ready
```

The config file is committed, so everyone's cluster is identical.

**If something breaks badly:**
```bash
kind delete cluster --name prodrome
```
Five minutes to rebuild. **It's disposable — that's the whole point.** Don't spend an hour debugging a cluster you can recreate.

---

## 4. Verify your setup

Run through this before you start real work. It takes ten minutes and saves hours.

**Everyone:**
- [ ] `python -c "import pandas, sklearn, torch; print('ok')"` succeeds
- [ ] You can load the committed sample data: `python -c "import pandas as pd; print(pd.read_parquet('data/samples/metrics.parquet').shape)"`
- [ ] `git log` shows the repo history

**Platform pair, additionally:**
- [ ] `docker run hello-world` works
- [ ] `kubectl get nodes` shows 3 Ready nodes
- [ ] `kubectl config current-context` is `kind-prodrome`
- [ ] `kubectl get pods -A` shows the cluster's own pods running

---

## 5. Repo layout

```
prodrome/
├── infra/              Cluster config, workload manifests, Dockerfiles
├── collect/            Metric scraper, chaos runner, load generation
├── ml/                 Detector, classifier, shared feature code
├── control/            Controller loop, policy table
├── dashboard/         Live view of decisions — terminal today, see dashboard/README.md for why not web yet
├── eval/               Evaluation harness, plots
├── data/               GITIGNORED except samples/
│   ├── samples/        Small committed fixtures — everyone can run everything
│   ├── healthy/        Baseline collection
│   ├── chaos/          Fault runs and labels
│   └── decisions/      Controller decision logs
├── docs/               Per-role guides
├── README.md           What this is and why
├── PRD.md              Full spec
├── PLAN.md             Workstreams, the core cycle, phase gates
└── SETUP.md            This file
```

`.gitignore` must contain:
```
.venv/
data/*
!data/samples/
!data/samples/**
*.pkl
__pycache__/
.DS_Store
```

> **Never commit large data or model files.** They bloat the repo permanently — git keeps every version forever. Only `data/samples/` is tracked, and it stays under a few MB.

---

## 6. Who owns what

| Area | Owner | Guide |
|---|---|---|
| Cluster, workloads, controller | **Shravan** | `docs/guides/shravan.md` |
| Prometheus, chaos, evaluation | **Shaurya** | `docs/guides/shaurya.md` |
| Detector (Signal) | **Sagar** | `docs/guides/sagar.md` |
| Classifier and policy (Diagnosis) | **Sadhil** | `docs/guides/sadhil.md` |

Sagar and Sadhil pair. Shaurya and Shravan pair. Pairs share context freely; the two halves sync on the contracts in §7 and nothing else.

---

## 7. The contracts

**These two schemas are the only things that need agreement across pairs.** Everything else is an implementation detail its owner can change freely.

### Metrics table (Parquet)
One row per workload per 15-second tick:

```
ts, workload, cpu_cores, mem_bytes, mem_pct,
net_rx, net_tx, fs_reads, fs_writes, restarts
```

`workload` is a stable name — `redis`, not `redis-7d9f8b-x2k1`.

### Labels table (CSV)
One row per injected fault:

```
start_ts, end_ts, workload, fault_type, pattern, run_id
```

`run_id` is required. Train/test splits are by run, never by row.

### Decision log (CSV)
One row per controller evaluation:

```
ts, workload, detector_score, fired, predicted_class,
confidence, top_features, action, result, mode
```

**Changing any of these requires telling everyone.** Don't do it quietly.

---

## 8. Files, not services

Workstreams exchange **files**, not HTTP calls:

| Artifact | Written by | Read by |
|---|---|---|
| `data/healthy/metrics.parquet` | Shaurya | Sagar |
| `data/chaos/metrics.parquet` + `labels.csv` | Shaurya | Sadhil |
| `ml/detector.pkl` | Sagar | Shravan |
| `ml/classifier.pkl` | Sadhil | Shravan |
| `data/decisions/log.csv` | Shravan | Shaurya |

No ports, no service discovery, no debugging someone else's process. **If someone's output isn't ready, fall back to `data/samples/` and keep working.** Nobody waits.

---

## 9. Git conventions

**Branches:** `<area>/<short-description>` — e.g. `signal/ewma-detector`, `platform/chaos-runner`.

**Commits:** present tense, one logical change. `add cooldown to controller`, not `fixes`.

**Pull requests:** open a draft early so others can see direction. Small and frequent beats one enormous merge.

**Never commit:** data files, model artifacts, credentials, `.venv/`.

---

## 10. Ground rules

These aren't style preferences — each one prevents a specific way of producing results that look good and mean nothing.

**1. The detector never trains on injected faults.** It trains on healthy data only. Injected faults are step functions; real degradation is a slide. A model trained on injected faults learns to recognize our injector.

**2. Split train/test by run, never by row.** Consecutive windows overlap by 19 of 20 ticks. A random split puts near-duplicates on both sides and produces meaningless accuracy. **If accuracy exceeds ~0.97, assume this bug and check.**

**3. Label the full fault trajectory, not just the peak.** Including the faint early windows. Otherwise the classifier only recognizes faults that are already catastrophic — and fails exactly where the detector fires.

**4. Every claim ships with the control-arm comparison.** Kubernetes already self-heals. "We recovered the service" means nothing without "and the control arm didn't."

**5. Build the simple baseline first.** A threshold, a shallow rule. It's how anyone knows the complicated thing earned its place. If the simple version ties, we ship it and say so.

**6. Report per fault type, never one aggregate.** An average hides the cases where the method doesn't work, and those are the most informative part.

**7. Say what it can't do.** Instant failures get no warning. No cross-service root cause. Closed label set. Stating these is what makes everything else credible.

**8. If it isn't reproducible from a fresh clone, it isn't a result.**

**The standing question for every change:** *what's the baseline, and did you beat it?*

---

## 11. First-day problems

| Symptom | Fix |
|---|---|
| `pip install` fails on torch | Use the CPU-only wheel from pytorch.org — the default pulls CUDA |
| `python3.11: command not found` | Try `python3`; check `python3 --version` is 3.11+ |
| `kubectl` talks to the wrong cluster | `kubectl config use-context kind-prodrome` |
| `kind create cluster` hangs | Docker isn't running, or has too little memory. Raise it to 8GB |
| Pods stuck `Pending` | Not enough CPU/memory on nodes. Lower resource requests |
| `ImagePullBackOff` on `prodrome/*` | Run `kind load docker-image` after every rebuild, then delete the pods |
| Cluster suddenly dead | `docker ps` — kind nodes are containers. Docker is probably out of memory |
| Everything broken | `kind delete cluster --name prodrome` and start over. Five minutes |

---

## 12. Where to go next

1. Read `README.md` — what this is and why the design is what it is (10 min)
2. Read your own guide in `docs/guides/` — written assuming no prior background
3. Skim `PLAN.md` §2, the core cycle — the eight stages every piece of work moves through
4. Start on your Phase 0 deliverable

Questions about someone else's area go to them, not into a guess. The whole point of the contracts is that you never need to understand the internals of another workstream.
