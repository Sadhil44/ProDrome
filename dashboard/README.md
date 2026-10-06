# dashboard/

Live view into what Prodrome is doing. Two pieces, deliberately unequal in how finished they are.

## `terminal.py` — built, works today

```bash
python -m dashboard.terminal
```

A live-updating table over the decision log at `data/decisions/log.csv` (`SETUP.md` §8; cluster's controller writes it): workload, detector score, fired, predicted class, confidence, the features that drove the call, action, result, and whether that action was real or shadow. Uses [`rich`](https://github.com/Textualize/rich), refreshes every 2 seconds. If the controller hasn't logged anything yet, it shows a waiting state instead of crashing.

This is `PRD.md` FR-23 and `docs/guides/shaurya.md` Part 7.4's dashboard.

## Why there's no web app here

This isn't an oversight — it's the guide's explicit call, and worth repeating rather than quietly overriding:

> "Do not start a React app. It will not finish and it will consume you for a full day. A clean terminal table demos perfectly well. Web UI is a later phase." — `docs/guides/shaurya.md` §7.4

`PRD.md` FR-23 lists a dashboard as **"Should," not "Must,"** and it's explicitly a Phase 5 deliverable (`PRD.md` §11.2) — the project isn't there yet (currently Phase 1–2). Building a half-finished web frontend now would cost a real day of someone's time for something with no users yet and a terminal equivalent that already works.

**When it's actually time for a web version** (Phase 5, after the terminal one has been demoed and the team wants something shareable beyond a laptop terminal):

- Keep it read-only at first: same `data/decisions/log.csv` source, served over a tiny local API (FastAPI is the lowest-ceremony option already compatible with the rest of the Python stack) rather than a new framework to learn.
- Don't reach for a full SPA framework for a table with a refresh button. A single static HTML page polling a `/decisions` JSON endpoint covers the FR-23 requirement ("showing live state and decision history") without a build step.
- The moment it needs to *do* something (trigger a manual action, adjust a threshold) instead of just display, that's a real design decision — not a default to reach for early.

## Definition of done (terminal, current scope)

- [x] Reads the decision log and matches what `control/controller.py` actually writes —
  now structurally rather than by inspection: `terminal.py` imports `LOG_FILE` and
  `DECISION_LOG_COLUMNS` from the controller instead of keeping its own copies.

  Both copies had drifted, and the second one mattered. `COLUMNS` said `timestamp`
  with no `top_features` and no `mode`; that was caught and fixed. The *path* was
  not — it still pointed at `control/decisions.csv` long after the controller moved
  to `data/decisions/log.csv`, so the dashboard would have sat showing its "waiting
  for decisions" state forever while the controller wrote perfectly good rows
  somewhere else. An operator watching a blank dashboard during an incident would
  reasonably conclude nothing was happening.

  Importing from the single source makes that divergence impossible rather than
  merely detectable. It is only safe because `controller.py` imports `kubernetes`
  lazily inside `connect_to_kubernetes()`, so it loads fine with no cluster.
- [x] Handles the "controller hasn't run yet" case without crashing
- [x] Renders correctly against fabricated decisions matching the real schema (both
  states checked: no file yet, and a populated log with mixed `fired`/`NORMAL` rows,
  newest-first)
- [ ] Verified against a live controller loop once one exists (currently `control/controller.py` has the individual action-building blocks but not the wired-up loop yet)
