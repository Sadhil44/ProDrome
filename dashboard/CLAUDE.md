# dashboard/ module rules

This directory belongs to the **diagnosis** seat (Sadhil): the live view into what the controller is
doing. It reads the decision log that the **cluster** seat writes.

- Read-only. The dashboard displays decisions; it does not make them, trigger them or adjust
  thresholds. The moment it needs to *do* something, that is a real design decision for
  `crosstalk/decisions`, not a default to reach for.
- It reads the decision log by column name, against the schema the controller publishes
  (`ts, workload, detector_score, fired, predicted_class, confidence, top_features, action, result,
  mode`). When that contract changes, this breaks — watch `cluster/interfaces/decision-log-schema`.
- Never crash when the controller has not run yet. A missing or empty log is the normal state early
  on; show a waiting state instead.
- No web app, and this is a deliberate call rather than an oversight: `PRD.md` FR-23 lists the
  dashboard as "Should," not "Must," and as a Phase 5 deliverable. A half-finished frontend costs a
  real day for something with no users and a terminal equivalent that already works. When Phase 5
  actually arrives, keep it read-only over the same CSV, served by a tiny local API and a single
  static page polling a JSON endpoint — not a new framework with a build step. Do not quietly
  override this; if the call should change, change it on the forum first.
- Coordination happens on the CCP forum (MCP `ccp-forum`, session `prodrome`): set status, post to
  `diagnosis/updates`, and read `cluster/interfaces/decision-log-schema` before assuming a column
  exists. See `forum/HANDOFF.md`.
