# infra/ module rules

This directory belongs to the **cluster** seat (Shravan): the kind cluster config, the workload
manifests and the stress-tool images.

- Everyone runs their own cluster, and that is deliberate — sharing one creates a queue and a queue
  means people wait. The configs here are committed so that everyone's cluster is identical.
- The cluster is disposable. `kind delete cluster --name prodrome` and rebuild takes five minutes;
  do not spend an hour debugging state you can recreate.
- Every workload carries resource limits and probes. Without a memory limit nothing ever OOMKills,
  without probes nothing ever restarts, and then there is no failure for Prodrome to predict — the
  manifests are what make the experiment possible.
- Workload names are stable and match what the metrics table records (`redis`, `nginx`, `postgres`).
  A rename ripples into every collected run and every trained model, so announce it.
- `kind load docker-image` after every image rebuild, then delete the pods. An `ImagePullBackOff` on
  a `prodrome/*` image is almost always this and not a registry problem.
- Two workload sets exist and stay in sync: the Prodrome arm and the control arm. The comparison is
  only meaningful if they run identical workloads under identical faults, so a change to one is a
  change to both.
- Keep manifests declarative and committed. Nothing that only exists as a `kubectl` command someone
  ran once is reproducible from a fresh clone, and if it is not reproducible it is not a result.
- Coordination happens on the CCP forum (MCP `ccp-forum`, session `prodrome`): set status, post to
  `cluster/updates`, and append changes that affect other seats — workload names, resource limits,
  the control-arm setup — to `cluster/interfaces`, announced labeled `collect`. See
  `forum/HANDOFF.md`.
