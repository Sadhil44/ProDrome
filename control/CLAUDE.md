# control/ module rules

Two seats share this directory. `controller.py` belongs to **cluster** (Shravan); `policy.py`
belongs to **diagnosis** (Sadhil). Work on the other seat's file by posting to its owner.

- `policy.py` is a hand-written lookup table and stays one. It is deliberately not learned, because
  an operator has to be able to answer "why did this restart my pod at 3am" — replacing it with a
  model is a project-level decision that belongs in `crosstalk/decisions`, not a refactor. Rationale:
  `PRD.md` §7.2.
- `policy.py` is pure: a class and a confidence in, an action out. No Kubernetes client, no I/O, no
  clock. Everything that touches the cluster lives in `controller.py`.
- Confidence thresholds differ per action on purpose, because being wrong costs different amounts.
  Below the floor everything becomes `UNKNOWN` and nothing happens. The consequence is intended:
  early in a failure only the cheap hedging action is available, and the expensive one unlocks as the
  diagnosis firms up. That falls out of the thresholds — do not add special-case logic to reproduce it.
- The controller writes one decision-log row per evaluation, and its column set is a contract
  (`ts, workload, detector_score, fired, predicted_class, confidence, top_features, action, result,
  mode`). The dashboard and the evaluation harness both read it by column name. Adding or renaming a
  column means appending to `cluster/interfaces/decision-log-schema` and announcing it.
- `mode` is `shadow` or `live`. Shadow decides and logs without touching the cluster, and is the
  default until the safety rails are verified. A change that can act on a cluster ships behind shadow
  first.
- Safety rails (cooldowns, max actions per window, what the controller refuses to do) are part of the
  contract, not an implementation detail. Post them; other seats reason about the system assuming they
  exist.
- The controller consumes `detector` and `classifier` through their published interfaces and nothing
  else. If it needs a model's internals, the interface is wrong — raise it on the forum rather than
  reaching into `ml/`.
- Every remediation claim ships with the control-arm comparison. "We recovered the service" means
  nothing without "and stock Kubernetes didn't."
- Coordination happens on the CCP forum (MCP `ccp-forum`, session `prodrome`): set status, post to
  `cluster/updates` or `diagnosis/updates`, and append contract changes to
  `cluster/interfaces/decision-log-schema` or `diagnosis/interfaces`. See `forum/HANDOFF.md`.
