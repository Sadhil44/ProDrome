# tests/

Correctness tests. Distinct from `eval/harness.py`, which **measures** how well the
detector performs — this asks whether the code does what it claims, and it answers in
about a minute.

```bash
python -m pytest tests -q
```

## The rule these follow

Everything here runs from a **fresh clone** with no cluster, no Docker, and no collection
run. That means they read `data/samples/` only, because it is the sole committed data
(`SETUP.md` §5). Ground rule 8 says a result that isn't reproducible from a fresh clone
isn't a result; a test that can't run from a fresh clone isn't a test.

This is why `ml/test_restart_suppression.py` is not part of this suite — it reads
`data/healthy/metrics.parquet`, which is gitignored, so it cannot run for anyone who
hasn't received a collection handoff.

## What is covered, and why each one exists

**`test_data_contracts.py`** — the three schemas from `SETUP.md` §7. The forum rule is
that changing one is never quiet. A test is the cheapest way to make that true: rename a
column and this goes red immediately instead of surfacing days later as somebody else's
confusing `KeyError`.

**`test_code_contracts.py`** — the published interfaces. `METRICS` order and
`WINDOW_SIZE` from signal's window-and-feature contract, the 40 summary features,
`policy.decide` across the whole table. Both halves of the pipeline must summarize a
window identically or the controller feeds two models two different things.

**`test_invariants.py`** — `SETUP.md` §10's ground rules made executable, plus regression
tests for bugs that actually happened here. No speculative coverage; every test maps to a
rule or a real defect.

## Bugs this suite already caught

- **The reported metrics and the shipped model came from different datasets.**
  `ml.train` defaulted to `data/samples/` through argparse while
  `classifier.fit_and_save()` preferred `data/chaos/`. That is how a 0.99 synthetic
  accuracy and a 0.82 real accuracy were both true at once.
- **`ml/abstention.py` could not be pointed at real data.** It called
  `load_labeled_windows()` with no arguments and had no CLI override, so the published
  "confidently wrong" figure could only ever describe the fixture. On real data it is
  roughly 2.7x worse.
- **The classifier artifact could only be loaded from one entrypoint.** `save()` pickled
  the wrapper instance, so `python -m ml.classifier` recorded its class as
  `__main__.RandomForestClassifier`. Any other importer — including the controller doing
  `from ml.classifier import classifier`, which `README.md` documents as *the* interface —
  got `AttributeError`. Found by writing a test that imports it from pytest, which is by
  definition not that `__main__`.

## What this suite deliberately does not do

It does not tell you whether Prodrome **works**. That question is the control-arm
comparison — identical workloads and identical faults, with and without Prodrome — and it
needs a live cluster (`PRD.md` Phase 4). These tests protect the contracts so that the
experiment, when it runs, is measuring what it thinks it is measuring.

Nor does it assert accuracy thresholds. Pinning a number to a dataset that is still
changing would produce a test that fails for the right reason at the wrong time.
