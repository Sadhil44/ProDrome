# Shravan - Cluster and Controller

## Prodrome: a complete beginner's guide

You own the machine everything runs on and the component that actually changes things. This guide assumes you have never touched Kubernetes and explains everything from zero.

> **Where this fits** (see `PRD.md` §10): Part 3 is Phase 0–1 (the cluster). Part 4 runs from Phase 1 (controller loop on stubs, in shadow) through Phase 4 (controller live, safety rails verified). Part 5.3 is Phase 6.
>
> **Repo status right now:** the `prodrome` kind cluster exists and is running (3 nodes Ready, context `kind-prodrome`) — Part 3.1 is done. Nothing past that yet: no stress-tool images, no `infra/workloads.yaml`, no controller code in `control/`. Two things you'll wire in already exist, ready to import: `ml/classifier.py` (stub, returns `("MEMORY_LEAK", 0.9)`) and `control/policy.py` (`decide(predicted_class, confidence) -> action`). Sagar's detector stub isn't written yet, so per Part 4.4 you'll need your own throwaway `fired = memory > 80%` detector until his real one exists.
>
> **Your next 3 steps:** (1) build the stress-tool images and load them into the cluster, Part 3.3. (2) deploy the workloads with resource limits and probes, Part 3.4, then prove a failure actually happens by hand, Part 3.5 — nothing else substitutes for watching an OOMKill happen once. (3) start the controller loop against the stubs above, in shadow mode.

---

## Part 1 — Understanding what you're building

### 1.1 Why containers exist

Before containers, deploying software meant: get a server, install the right version of Python, install the right libraries, copy your code over, hope the server's operating system matches the one you developed on. It usually didn't. "Works on my machine" was a genuine, constant problem.

A **container** solves this by packaging your program *together with* everything it needs — libraries, runtime, system tools, configuration — into one sealed unit. That unit runs identically anywhere a container runtime exists.

A container is **not** a virtual machine. A VM emulates an entire computer including its own operating system, which is heavy — gigabytes, and tens of seconds to boot. A container shares the host's operating system kernel and just isolates the process. It's megabytes, and it starts in under a second.

**Image versus container.** An image is the frozen template — a file on disk. A container is a running instance of that image. One image, many containers, exactly like one class and many objects.

### 1.2 Why Kubernetes exists

Now you have containers. You need twenty of them running across five machines. Something crashes at 3am. Traffic triples at lunchtime. A machine dies.

Without an orchestrator, a human does all of this. Kubernetes automates it.

**The single idea you need to understand — and everything else follows from it:**

> **You declare what you want. Kubernetes continuously makes reality match.**

This is called a **reconciliation loop**, and it runs a few times per second, forever:

```
observe actual state → compare to desired state → act to close the gap → repeat
```

You never say "start a container." You say "I want three copies of this running." Kubernetes counts, finds two, and starts one. If one dies, it counts again, finds two, and starts another. Nobody is paged.

This is different from every kind of programming you've done. You are not issuing commands; you are editing a target that a machine is continuously chasing.

### 1.3 The vocabulary

Learn these seven. Everything else is a variation.

**Node** — a machine that runs containers. In our project the "machines" are Docker containers pretending to be machines, which is a strange thing to think about for about ten minutes and then stops being strange.

**Pod** — the smallest thing Kubernetes runs. Usually one container. Pods are disposable — they get killed and recreated constantly, and they get a new IP address each time. **Never depend on a specific pod.**

**Deployment** — the declaration: "keep N pods of this image alive." When you scale or restart something, you're editing a Deployment. **This is the object your controller writes to.**

**Service** — a stable name and address that load-balances across whatever pods currently exist. Pods come and go; the Service name doesn't change. This is how anything finds anything.

**Namespace** — a folder for grouping objects. We use two: `prodrome` for the workloads Prodrome manages, and `control` for identical workloads it doesn't touch.

**Probe** — a health check Kubernetes runs on your container.

- *Readiness probe*: "can you serve traffic right now?" Fails → removed from the Service, but left running.
- *Liveness probe*: "are you alive at all?" Fails → **the pod is killed and restarted.**

**Resource requests and limits** — requests are what the scheduler reserves for you; limits are the hard ceiling. Exceed the CPU limit and you get throttled (slowed down). Exceed the **memory** limit and the container is killed instantly. That kill is called **OOMKilled** — Out Of Memory Killed.

### 1.4 The most important thing in this entire project

> **Resource limits are what make failures possible.**

Without a memory limit, a leaking container just consumes the host's memory until something arbitrary and messy happens. With a memory limit of 512Mi, Kubernetes kills it precisely when it crosses 512Mi.

**That kill is the failure event the entire project predicts.** No limits means no failures means nothing to detect means no project. If you get one thing right this weekend, it's this.

### 1.5 Where your controller fits

Kubernetes ships with a component called the **HorizontalPodAutoscaler** (HPA). Beginners assume it's some deep part of the system. It isn't.

The HPA is an ordinary program that:

1. Reads a metric (usually CPU)
2. Does arithmetic
3. Writes one number — a Deployment's replica count — through a public HTTP API

That's it. **You are writing a competitor to it.** Not a plugin, not a patch — a peer that reads better inputs and writes the same number. You have no special privileges. Everything you do, anyone could do with `curl`.

The single API call your entire controller is built around:

```
PATCH /apis/apps/v1/namespaces/prodrome/deployments/redis/scale
{"spec": {"replicas": 3}}
```

Understanding that this is *all* the HPA does is the moment the project stops feeling intimidating.

---

## Part 2 — Setup

### 2.1 Install

You need four tools.

**Docker** — runs containers. Install Docker Desktop (Mac/Windows) or Docker Engine (Linux). Verify:

```bash
docker run hello-world
```

**kind** — "Kubernetes IN Docker." Creates a throwaway cluster on your laptop in about a minute. No cloud account, no cost, and `kind delete cluster` erases every mistake you've ever made.

**kubectl** — the command-line client that talks to a cluster.

**helm** — a package manager for Kubernetes. You'll use it exactly once, to install Prometheus.

macOS:
```bash
brew install kind kubectl helm
```

Linux:
```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64
chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind

curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

**Give Docker at least 8GB of memory** in its settings. The default is often 2GB and the cluster will die mysteriously.

### 2.2 The five commands you will actually use

```bash
kubectl get pods -n prodrome                       # what's running, and its state
kubectl get pods -n prodrome -w                     # same, but live-updating
kubectl describe pod <name> -n prodrome              # why is it unhappy — read the Events at the bottom
kubectl logs <name> -n prodrome                      # the container's output
kubectl logs <name> -n prodrome --previous           # output from before it crashed ← the one people forget
```

Plus the best debugging command nobody teaches:

```bash
kubectl get events -n prodrome --sort-by=.lastTimestamp
```

OOMKills, failed scheduling, and image pull failures all appear here in plain English.

### 2.3 Reading pod status

| Status | Meaning |
|---|---|
| `Running` | Good |
| `Pending` | Not scheduled yet — usually no node has enough free CPU/memory |
| `ContainerCreating` | Pulling the image or mounting things. Normal for ~30s |
| `ImagePullBackOff` | Can't find the image. **Almost always the kind-load problem in §3.3** |
| `CrashLoopBackOff` | Starts, dies, restarts, repeats. Use `logs --previous` |
| `OOMKilled` | Exceeded its memory limit. **For us this is often a success, not a bug** |
| `Terminating` | Shutting down |

---

## Part 3 — Building the cluster

### 3.1 Create it

`infra/kind-config.yaml`:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
```

```bash
kind create cluster --name prodrome --config infra/kind-config.yaml
kubectl get nodes
```

You should see three nodes, all `Ready`. If you run `docker ps` you'll see three Docker containers — those are your "machines."

**Why two workers and not one?** The control-plane node is reserved for Kubernetes' own components. Worker nodes run your workloads. With only one worker, scaling to five pods puts all five on one machine and hides scheduling problems you'd hit in Part 5.

**Commit the config file.** Everyone on the team runs their own cluster, and identical configs mean identical environments.

### 3.2 Understanding YAML

Every Kubernetes object is described in YAML. Four fields appear in all of them:

```yaml
apiVersion: apps/v1     # which API this object belongs to
kind: Deployment         # what type of object
metadata:                # name, namespace, labels
  name: redis
  namespace: prodrome
spec:                     # the desired state — the actual content
  ...
```

Two things that will bite you:

**Indentation is two spaces and tabs are illegal.** A tab produces a parse error that doesn't tell you it's about a tab.

**Labels and selectors.** Labels are arbitrary key-value tags on objects. A selector is a query over them. This is how a Deployment finds "its" pods and how a Service decides where to send traffic — **not** by name, by label match. If your Service sends traffic nowhere, the selector doesn't match the pod labels. This is the single most common Kubernetes mistake.

### 3.3 Build images with a stress tool

You need to be able to deliberately overload a container. Stock images don't include a tool for that, so you build thin images that add one.

`infra/images/Dockerfile.redis`:

```dockerfile
FROM redis:7
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends stress-ng procps \
 && rm -rf /var/lib/apt/lists/*
```

Line by line: start from the official Redis image; switch to the root user so you can install things; install `stress-ng` (generates CPU, memory, and disk load on demand) and `procps` (gives you `ps` and `top` for debugging); delete the package cache so the image stays small.

Repeat for `nginx:1.27` and `postgres:16`.

```bash
cd infra/images
docker build -t prodrome/redis:1    -f Dockerfile.redis .
docker build -t prodrome/nginx:1    -f Dockerfile.nginx .
docker build -t prodrome/postgres:1 -f Dockerfile.postgres .

kind load docker-image prodrome/redis:1 prodrome/nginx:1 prodrome/postgres:1 --name prodrome
```

> **The gotcha that costs everyone an hour**
>
> **Your cluster cannot see your local Docker images.** The kind nodes are separate Docker containers with their own image storage. `kind load` copies an image into them.
>
> Three consequences:
>
> 1. Every rebuild needs another `kind load`
> 2. After loading, **delete the pods** so they pick up the new image
> 3. Your manifests must set `imagePullPolicy: IfNotPresent`, or Kubernetes will try to download from Docker Hub, fail, and sit in `ImagePullBackOff` while you stare at it
>
> If you see `ImagePullBackOff` on an image whose name starts with `prodrome/`, it is one of these three. Every time.

### 3.4 Deploy the workloads

`infra/workloads.yaml` — this is the pattern; repeat for nginx and postgres.

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: prodrome
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: prodrome
spec:
  replicas: 1
  selector:
    matchLabels: { app: redis }        # must match the pod labels below
  template:
    metadata:
      labels: { app: redis }            # ← these labels
    spec:
      containers:
        - name: redis
          image: prodrome/redis:1
          imagePullPolicy: IfNotPresent
          ports: [{ containerPort: 6379 }]
          resources:
            requests: { memory: "256Mi", cpu: "200m" }
            limits:   { memory: "512Mi", cpu: "500m" }
          readinessProbe:
            exec: { command: ["redis-cli", "ping"] }
            initialDelaySeconds: 5
            periodSeconds: 5
          livenessProbe:
            exec: { command: ["redis-cli", "ping"] }
            initialDelaySeconds: 15
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: prodrome
spec:
  selector: { app: redis }
  ports: [{ port: 6379, targetPort: 6379 }]
```

**Units:** `256Mi` is mebibytes. `200m` is millicores — 200/1000ths of one CPU core. `500m` is half a core.

**Choosing the memory limit.** Start the container with no limit, watch its idle memory, then set the limit to roughly double. Too tight and it dies at startup; too loose and a stress run never triggers a kill. You want stress to be able to breach it in a few minutes.

**Liveness delay must exceed readiness delay.** Otherwise Kubernetes kills the container while it's still booting, forever.

For nginx use `httpGet: { path: /, port: 80 }` for both probes. For postgres use `exec: ["pg_isready","-U","postgres"]` and add `env: [{ name: POSTGRES_PASSWORD, value: prodrome }]`.

```bash
kubectl apply -f infra/workloads.yaml
kubectl get pods -n prodrome -w
```

### 3.5 Prove a failure happens — do this before anything else

**This is the verification step that matters most, and it takes five minutes.**

```bash
kubectl exec -n prodrome deploy/redis -- \
  stress-ng --vm 1 --vm-bytes 600M --vm-hang 0 --timeout 120s

# in another terminal:
kubectl get pods -n prodrome -w
```

You should watch the pod get killed and restarted. Then confirm why:

```bash
kubectl describe pod <redis-pod> -n prodrome | grep -A3 "Last State"
```

You want to see `Reason: OOMKilled`.

**If this doesn't happen, stop and fix it.** Either `--vm-bytes` is below the limit (raise it), or the limit is too high (lower it). Everything the team does depends on this working.

### 3.6 The control namespace

```bash
sed 's/namespace: prodrome/namespace: control/; s/name: prodrome/name: control/' \
  infra/workloads.yaml > infra/workloads-control.yaml
kubectl apply -f infra/workloads-control.yaml
kubectl get pods -A | grep -E "prodrome|control"
```

Identical workloads. Identical faults. **No Prodrome.**

**Why this cannot be skipped or deferred.** Kubernetes already restarts crashed pods on its own. If you inject a fault and Prodrome recovers the service, that proves nothing — stock Kubernetes would have too. The only way to claim anything is to run both side by side under identical conditions and show a difference.

Teams always plan to add the control arm "later." It never happens, and then there is no result. Set it up on day one.

---

## Part 4 — The controller

### 4.1 What it is

A Python program in a loop. Every 15 seconds it asks two models what's happening and, if warranted, makes one API call.

```
read metrics → ask detector → ask classifier → look up action → call Kubernetes → log
```

Install the client:

```bash
pip install kubernetes
```

### 4.2 Build the actions individually first

**Do not write the loop yet.** A loop full of untested functions is unbearable to debug. Build four capabilities and test each by hand.

**Read the current replica count.** Uses the `read_namespaced_deployment_scale` call.

**Scale.** Patches the Deployment's scale subresource with a new replica count. Test it:

```bash
python -c "from control.controller import scale; scale('redis', 3)"
kubectl get pods -n prodrome
```

You should see three redis pods. Scale back to 1.

**Restart.** There's no "restart" verb in Kubernetes. The standard technique: write a timestamp into an annotation on the *pod template*. Kubernetes sees the template changed, so it rolls out new pods one at a time. Test it and watch — pods should cycle **one at a time**, not all at once.

> **The mistake here:** patching the Deployment's own metadata instead of `spec.template.metadata`. Nothing happens and you'll stare at it. The annotation must be on the pod template.

**Log a decision.** Append one row per evaluation to a CSV: timestamp, workload, detector score, whether it fired, predicted class, confidence, chosen action, and what actually happened.

### 4.3 Safety rails — build these now, not after an incident

**Cooldown.** No two actions on the same workload within two minutes. Without it, a twitchy model scales up, down, up, down and your demo looks broken.

**Replica ceiling.** Cap it at five. A bug that scales to 200 will take down your laptop.

**Kill switch.** If a file named `STOP` exists, take no action. Test it, then delete the file. **You will need this**, probably the first time you turn off dry-run.

**Dry-run mode, on by default.** Log the action, execute nothing. You run in this mode for the first day or two.

### 4.4 Stub the models so you're never blocked

Sagar and Sadhil won't have real models for a while. Write throwaway versions with the correct interface:

- A "detector" that returns `fired = True` when memory exceeds 80% of the limit
- A "classifier" that always returns `("MEMORY_LEAK", 0.9)`

**This is the whole point of the file contract.** Your loop runs end-to-end today. When their real model files appear, you change a file path and nothing else. Agree the exact function signatures with them before you start and write them in the README.

### 4.5 Going live

1. Set dry-run to false
2. **Watch one complete fault cycle. Sit there and watch it.** Do not walk away
3. Check the log — no back-to-back actions on one workload (cooldown works)
4. Test the kill switch, then remove it
5. Wire in the detector's post-restart reset

**Why step 5 matters:** after a restart, metrics look nothing like steady state — memory near zero, CPU spiking during startup. To Sagar's detector this looks exactly like an anomaly, so it fires, so you restart, so it fires again. **Infinite loop.** Ask him for a suppression function and call it after every restart action.

---

## Part 5 — Packaging and Phase 6

### 5.1 Make it reproducible

Write a `Makefile` with `cluster`, `demo`, and `clean` targets. Then **verify it by cloning the repo fresh into a new directory and running it**, not by reading it. Reading always works.

Document why the resource limits are the numbers they are. Future-you will not remember.

### 5.2 Permissions, when it moves in-cluster

Right now your controller runs on your laptop with your credentials, which are admin. When it becomes a pod inside the cluster, it needs its own identity — a **ServiceAccount** — with narrowly scoped permissions:

```yaml
rules:
  - apiGroups: ["apps"]
    resources: ["deployments/scale"]
    verbs: ["get", "patch"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch", "patch"]
```

It can resize and restart deployments. It cannot read secrets, delete anything, or touch other namespaces.

**This is worth understanding rather than copying.** "Why did you scope it that narrowly?" is a question you'll get asked, and the answer — limiting blast radius if the controller has a bug — is the difference between someone who ships features and someone who ships systems.

### 5.3 Phase 6 — your part

When the project reaches the learned-policy phase, you build the **real training environment**: the cluster and fault harness wrapped behind the same interface Sagar and Sadhil's simulator uses. Same functions, same inputs and outputs, so their agent runs against either without modification.

They train in the simulator because it's thousands of times faster. You provide the real thing for validation. The gap between them is a result worth reporting.

---

## Part 6 — Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImagePullBackOff` on `prodrome/*` | Not kind loaded, or `imagePullPolicy` isn't `IfNotPresent`. Load, then delete the pods |
| `CrashLoopBackOff` | `kubectl logs <pod> --previous` — the current container has no output yet |
| `Pending` forever | `kubectl describe pod` → Events. Usually not enough CPU/memory on nodes. Lower requests or your replica ceiling |
| `OOMKilled` at startup | Memory limit is below idle usage. Raise it |
| Service reaches nothing | Selector labels don't match pod labels. `kubectl get endpoints <svc> -n prodrome` — empty means no match |
| `stress-ng: not found` | Image wasn't rebuilt with it, or wasn't reloaded |
| Restart does nothing | Annotation patched on the Deployment, not `spec.template.metadata` |
| `kubectl` talks to the wrong cluster | `kubectl config current-context` → should be `kind-prodrome` |
| Cluster suddenly dead | `docker ps` — kind nodes are containers. Docker is likely out of memory |
| Everything is broken | `kind delete cluster --name prodrome && make cluster`. Five minutes. **It's disposable — that's the entire point** |

---

## Part 7 — Reading list

**Do read:** the Kubernetes concepts pages for Pod, Deployment, and Service (30 min); the kind quick start (10 min); "Configure Quality of Service for Pods" for requests and limits (15 min).

**Don't read yet:** anything about Ingress, StatefulSets, operators, service meshes, or GitOps. None of it applies here and all of it will make Kubernetes seem harder than it is.

---

## Definition of done

- [ ] One command from fresh clone to running cluster
- [ ] One command runs the demo
- [ ] A deliberately stressed container gets OOMKilled and restarted, verified by hand
- [ ] Control namespace running identical workloads with no controller
- [ ] All four controller actions tested individually
- [ ] Cooldown, ceiling, and kill switch each verified
- [ ] Decision log readable by someone who didn't write it
- [ ] Simple decision-tree baseline trained and printed for the team


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
