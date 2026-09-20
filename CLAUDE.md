# Prodrome: rules for agents

Read `README.md` for the design, `SETUP.md` for conventions, `PRD.md` for the spec, and the
`CLAUDE.md` in whichever directory you are working in for that module's rules.

## You are one seat of four

Work here is split into four aspects, one per person (`SETUP.md` §6). Each is a shelf on the forum:

| Aspect | Owner | Owns |
|---|---|---|
| `cluster` | Shravan | `infra/`, `control/controller.py` — cluster, workloads, controller, safety rails |
| `collect` | Shaurya | `collect/`, `eval/`, everything written into `data/` — scraping, chaos, load, evaluation |
| `signal` | Sagar | `ml/detector.py`, `ml/features.py`, `ml/replay.py` — the detector and the shared feature code |
| `diagnosis` | Sadhil | `ml/classifier.py`, `ml/train.py`, `ml/abstention.py`, `control/policy.py`, `dashboard/` |

Work on your own seat's files. To change another seat's file, post to its owner instead of editing it.

## Coordination: the CCP forum

The seats exchange **files, not services** (`SETUP.md` §8), which keeps everyone unblocked but means
nothing tells you when someone changed a column, a metric order or a confidence threshold. The forum
is what tells you.

Use the `ccp-forum` MCP, session `prodrome`. Read `master_instructions` first — it is the rules, and
it is more specific than this file. Then: `set_status`, read `crosstalk/announcements` and
`crosstalk/blockers`, read your aspect's `interfaces` and `updates`, post progress to
`<aspect>/updates`, publish contracts in `<aspect>/interfaces`, and announce anything that affects
another seat in `crosstalk/announcements` labeled with that seat. Setup and hosting:
`forum/HANDOFF.md`.

If the forum is unreachable, keep working and post the backlog when it returns — say in your update
that it was written offline.

## The ground rules (`SETUP.md` §10)

Each one prevents a specific way of producing results that look good and mean nothing.

1. The detector never trains on injected faults — healthy data only.
2. Split train/test by run, never by row. Above ~0.97 accuracy, assume this bug and check.
3. Label the full fault trajectory, not just the peak.
4. Every claim ships with the control-arm comparison.
5. Build the simple baseline first; if the simple version ties, ship it and say so.
6. Report per fault type, never one aggregate.
7. Say what it can't do.
8. If it isn't reproducible from a fresh clone, it isn't a result.

The standing question for every change: **what's the baseline, and did you beat it?**

## Conventions

- Branches `<area>/<short-description>`; commits present tense, one logical change.
- Never commit data files, model artifacts, credentials or `.venv/`. `data/` is gitignored except
  `data/samples/`, which stays small and schema-valid.
- The three cross-pair schemas in `SETUP.md` §7 (metrics table, labels table, decision log) are the
  only things needing agreement across seats. Changing one is never quiet: append to its
  `interfaces` entry and announce it.
