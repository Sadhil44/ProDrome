# tests/

Correctness tests. Distinct from `eval/harness.py`, which **measures** how well the
detector performs — this asks whether the code does what it claims, and it answers in
about a minute.

```bash
python -m pytest tests -q     # or just `pytest tests -q`
```

Either form works: `tests/conftest.py` puts the repo root on `sys.path`. Only the
`python -m` form worked before, because `python -m` happens to prepend the working
directory and bare `pytest` does not — so `from ml.features import ...` failed at
collection.

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

**`test_units.py`** — the stages one at a time. `summarize`'s arithmetic (including that it
uses the population std, not pandas' sample std — the obvious reimplementation differs by
2.6% at `WINDOW_SIZE=20`); `windows`' edge cases (shorter than a window, exactly a window,
overlap by 19 of 20, workload isolation, sort order, gap-fill *values*, and that it does not
mutate the caller's frame); `ml/dataset.py`'s labelling rule, which is that a window is
labelled by its **last** tick; the policy table's structural invariants and threshold
boundaries; and the detector's warmup, `score_only`-vs-`update` split, non-finite handling,
constant-metric guard and post-restart suppression. That last one is the fresh-clone
counterpart to `ml/test_restart_suppression.py`, which reads the gitignored `data/healthy/`
and so cannot run for anyone without a collection handoff.

**`test_integration.py`** — the stages wired together on `data/samples/`. Not "does it work"
but "do they fit": every class the shipped classifier can emit has a policy row, every action
policy emits is in the vocabulary a controller implements, the window `windows()` hands over
satisfies `predict()`'s documented precondition, and `Detector.score()` really is positional
in `METRICS` order. The controller stage is a fake, because `control/controller.py` imports
`kubernetes` at module scope; the real one is inspected as source with `ast`.

**`test_degradation.py`** — one stage broken at a time, asserting the next degrades safely.
An empty detector gate → nothing downstream runs, *and* the classifier records zero calls;
`POD_KILL` → `"nothing"` through the whole loop, never an action; sub-floor and
sub-threshold confidence → no action but the class still logged; a window missing a metric
or too short to fit a slope → a raise, not a prediction.

**`conftest.py`** — the sample-data fixtures, the replayed firing log (session-scoped; it is
the most expensive thing in the suite) and the four-stage `run_loop` both of the above drive.
The wiring lives in a fixture because what both files test is the seams.

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

## Open defects this suite records as strict xfails

Four findings sit in the suite as `@pytest.mark.xfail(strict=True)` rather than as red
builds. They are other seats' files, and testing asserts contracts rather than renegotiating
them — but a finding that only exists in a report gets lost. `strict=True` means that when
the owner fixes it the test **XPASSes and CI goes red**, which is the signal to delete the
marker. None are speculative; all were reproduced.

| Where | What |
|---|---|
| `control/controller.py` | `execute_action()` branches only on `"restart"`. Policy emits `scale_out`, `rolling_restart`, `alert_only`. The two vocabularies do not intersect, so wired up as written every decision logs as `unknown-action` and nothing is ever remediated — and `DRY_RUN=True` hides it, because the dry-run branch returns before the comparison. |
| `control/controller.py` | `log_decision()` writes `timestamp`, not `ts`, and has no `top_features` or `mode` column. `mode` is what separates a shadow observation from an executed action, so the control-arm comparison (ground rule 4) is unanswerable from the log. It also writes `control/decisions.csv` while `SETUP.md` §8 names `data/decisions/log.csv`. |
| `dashboard/terminal.py` | `COLUMNS` mirrors the controller's actual header rather than `SETUP.md` §7, and `row.get(col, "")` renders a missing column as blanks — an operator cannot tell "took no action" from "the column is gone". |
| `ml/classifier.py` | `predict()` never checks `len(window) == WINDOW_SIZE`. Any length from 2 up summarizes fine and returns a confident label computed over the wrong amount of history. `windows()` never produces one, so this only bites a caller that assembles a window itself — which is what the controller does on a short buffer at startup. |

And one finding that is **not** an xfail, because its mechanism is now pinned by a passing
test (`test_update_cannot_fire_on_an_arbitrarily_extreme_anomaly`):
`MetricDetector.update()` appends this tick's error to the reference distribution *before*
scoring against it, which makes the z-score scale-invariant. The single-tick ceiling is
`(1/RECENT_WINDOW) * sqrt(ERROR_HISTORY)`, about 2.0, and a sustained burst peaks near 3.9 —
both under `Z_THRESHOLD = 4.0`. So `Detector.score()`, the only public scoring API and the
one the pickled artifact hands to the controller, cannot fire on a step change of *any*
magnitude: 10x and 10^12x produce the identical score. `score_only()` on the same input
fires immediately. That is the whole of the "fires 3 times across 160 fault runs"
desensitization — not tuning, but training on the tick being scored.

## Handoff: the end-to-end demo

Not built here, on purpose — a runnable pseudo-application is Aahan's to write by hand. What
it should cover, given what the suite already establishes:

- **Drive `conftest.run_loop`**, or the same four stages in the same order. The wiring is
  already exercised and the fixtures already load the data; the demo's value is the
  narration, not new plumbing.
- **Print per stage**, per fired tick: the detector's score, the predicted class and its
  confidence, the policy row that matched and the threshold it cleared or missed, and the
  action with the reason it was or was not executed. The abstentions are the interesting
  rows, not the actions.
- **Use `data/samples/` and no cluster.** The sample fixture does fire — 12 discrete events
  across 3 workloads, 171 fired ticks, via the `score_only()` replay path — so there is
  something to show. Say plainly on screen that the *live* `Detector.score()` path would show
  almost nothing, and why (the desensitization finding above).
- **What it needs that does not exist yet**: a controller whose `execute_action()` handles the
  real action vocabulary, and a decision-log writer at the `SETUP.md` §7 columns. Until both
  land the execute stage has to be a stand-in, and the demo should say so rather than
  implying the loop is closed.

## What this suite deliberately does not do

It does not tell you whether Prodrome **works**. That question is the control-arm
comparison — identical workloads and identical faults, with and without Prodrome — and it
needs a live cluster (`PRD.md` Phase 4). These tests protect the contracts so that the
experiment, when it runs, is measuring what it thinks it is measuring.

Nor does it assert accuracy thresholds. Pinning a number to a dataset that is still
changing would produce a test that fails for the right reason at the wrong time.
