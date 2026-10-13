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

**CI runs more than this directory.** Three other seats keep tests next to their code, and
until 2026-10-06 none of them ran on a push. `.github/workflows/tests.yml` now runs:

```bash
python -m pytest -q tests \
  control/test_safety_rails.py control/test_policy_gates.py \
  ml/test_explain.py ml/test_scale_sensitivity.py ml/test_abstention.py \
  collect/test_twoarm.py eval/test_recovery.py
```

**313 passed, 2 xfailed, about 70s** as of 2026-10-12, verified from a copy of the tree whose only
data is `data/samples/` and with no `classifier.pkl` — i.e. what CI actually gets. `collect/` and
`eval/` joined that day: they were held out while those directories were being rewritten mid-campaign
(a moving target as a merge gate is one seat's work-in-progress as everyone's red build), the
campaign landed, and the 84 tests they bring need nothing beyond `pandas` and `pytest`. Neither
reads `data/` — collect's cluster calls are injected callables and eval's frames are built
in-process — which is what makes them enforceable from a fresh clone at all.

The files stay where their owners keep them — `tests/CLAUDE.md` says other seats' tests live
beside their code, so the invocation widened rather than the files moving. The paths are named
explicitly so the install list (`pandas pyarrow numpy scikit-learn pytest rich`, deliberately
not `requirements.txt`) stays honest about what the suite imports. **If you add a test file
anywhere in the repo, add its path to that workflow**; `tests/test_ci_enforcement.py` fails if
a `test_*.py` is neither run by CI nor listed in its `KNOWN_UNENFORCED` with a reason.

## The rule these follow

Everything here runs from a **fresh clone** with no cluster, no Docker, and no collection
run. That means they read `data/samples/` only, because it is the sole committed data
(`SETUP.md` §5). Ground rule 8 says a result that isn't reproducible from a fresh clone
isn't a result; a test that can't run from a fresh clone isn't a test.

This is why `ml/test_restart_suppression.py` is not part of this suite. It is also why naming it
in the workflow would not help: it defines no `test_` functions at all — it is a `main()` script —
so CI would collect zero tests from it and report green. Its `main()` reads
`data/healthy/metrics.parquet`, which is gitignored and so absent from the clone CI builds, even
though it exists on machines that have had a collection handoff. `test_units.py` carries the
fresh-clone counterpart of what it checks.

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
in `METRICS` order. The controller stage in the replay fixture is still a fake — it asserts that
an action outside the vocabulary is a *loud* failure, which the real controller deliberately is
not — but the controller's own behaviour is now checked by **calling it**: `execute_action` for
every action in `POLICY`, `log_decision` with the CSV read back, and `dashboard.terminal`
imported and rendered. These were source parses until 2026-10-06, because
`control/controller.py` imported `kubernetes` at module scope and the module could not be
loaded at all. That import is lazy now, and a test pins it: if it ever returns to module scope,
CI says so instead of failing at collection with what looks like a broken runner.

**`test_degradation.py`** — one stage broken at a time, asserting the next degrades safely.
An empty detector gate → nothing downstream runs, *and* the classifier records zero calls;
`POD_KILL` → `"nothing"` through the whole loop, never an action; sub-floor and
sub-threshold confidence → no action but the class still logged; a window missing a metric
or too short to fit a slope → a raise, not a prediction.

**`test_ci_enforcement.py`** — the only file here that asserts about CI rather than about the
pipeline, and it earns its place because the gap it guards was real for most of this project:
`pytest tests` ran 83 tests and silently skipped the 76 safety-rail tests, the policy gates and
the attribution identity. It fails if any `test_*.py` in the repo is neither run by the
workflow nor listed in `KNOWN_UNENFORCED` with a reason, if the three load-bearing files are
dropped from the invocation, or if CI starts installing `requirements.txt`.

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

## Defects this suite recorded as strict xfails — all four now fixed

`tests/` carries **no xfails today**. It carried four, as `@pytest.mark.xfail(strict=True)`
rather than red builds: they were other seats' files, and testing asserts contracts rather
than renegotiating them — but a finding that only exists in a report gets lost. `strict=True`
means that when the owner fixes it the test **XPASSes and CI goes red**, which is the signal to
delete the marker. That is exactly what happened to all four; the tests remain as regression
guards, now passing.

| Where | What it was | Now |
|---|---|---|
| `control/controller.py` | `execute_action()` branched only on `"restart"`. Policy emits `scale_out`, `rolling_restart`, `alert_only`. The vocabularies did not intersect, so every decision would log as `unknown-action` and nothing would ever be remediated — and `DRY_RUN=True` hid it, because the dry-run branch returns before the comparison. | Fixed. Now checked by *calling* `execute_action` for every action in `POLICY`, with a spy API, and requiring a mutating action to actually issue its write. |
| `control/controller.py` | `log_decision()` wrote `timestamp`, not `ts`, with no `top_features` and no `mode`. `mode` separates a shadow observation from an executed action, so the control-arm comparison (ground rule 4) was unanswerable from the log. It also wrote `control/decisions.csv` while `SETUP.md` §8 names `data/decisions/log.csv`. | Fixed. Now checked by calling `log_decision` and reading the CSV back — header, row width, `ts` parsing, `mode` derived from `DRY_RUN`. |
| `dashboard/terminal.py` | `COLUMNS` mirrored the controller's actual header rather than `SETUP.md` §7, and `row.get(col, "")` rendered a missing column as blanks — an operator could not tell "took no action" from "the column is gone". | Fixed, and then fixed harder: diagnosis replaced both `COLUMNS` and the path with imports from the controller, so the divergence is impossible rather than merely detectable. The path was separately wrong (`control/decisions.csv`), which no column check could have caught; both halves are pinned now. |
| `ml/classifier.py` | `predict()` never checked `len(window) == WINDOW_SIZE`. Any length from 2 up summarized fine and returned a confident label computed over the wrong amount of history — which is what the controller assembles on a short buffer at startup. | Fixed. |

The two xfails in the repo today are both in `control/test_policy_gates.py`, and they are
**imperative `pytest.xfail()` calls** (lines 89 and 124), not `@pytest.mark.xfail` markers. That
distinction matters and this file previously got it wrong. An imperative call aborts the test and
reports xfail unconditionally, so it can never XPASS: if somebody fixes these thresholds tomorrow,
nothing goes red and nothing tells them the finding is stale. These two are recorded, not pinned.

They are deliberately not strict markers: CPU_HOG's `min_confidence` equals the abstention floor
so its gate is dead code, and a mutating action unlocks cheaper than an alert. The fix is a
threshold value, which is a project decision on `crosstalk/decisions`, not something a test should
force. There is deliberately no `pytest.ini` — but note that adding `xfail_strict` to one would
**not** affect these two either way, since that setting governs only the marker. An earlier version
of this paragraph claimed it would turn them into a red build; that was wrong.

And one finding that is **not** an xfail, because the defect it records has since been fixed
and its absence is pinned by a passing test
(`test_update_fires_in_proportion_to_how_extreme_the_anomaly_is` in `tests/test_units.py`):
`MetricDetector.update()` *used to* append this tick's error to the reference distribution
before scoring against it, which made the z-score scale-invariant — 10x and 10^12x produced
the identical score, under a single-tick ceiling of `(1/RECENT_WINDOW) * sqrt(ERROR_HISTORY)`,
about 2.0, with a sustained burst peaking near 3.9, all below `Z_THRESHOLD = 4.0`. Since
`Detector.score()` is the only public scoring API and the one the pickled artifact hands to
the controller, the live gate never opened — the whole of the "fires 3 times across 160 fault
runs" desensitization, not tuning but training on the tick being scored.

What changed: `ml/detector.py:135-148` now scores first and folds the tick's error into the
reference only when it did **not** look anomalous, capped at `SELF_EXCLUDE_CAP` consecutive
exclusions so a genuine baseline shift is still learned. Magnitude therefore reaches the
z-score, and `score()` — which still routes through `update()` — fires on input that
`score_only()` fires on. The ceiling figures above describe the old ordering only; no
replacement ceiling is quoted here because none has been measured against this code.

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
- **Both blockers are gone.** This used to say the demo needed a controller whose
  `execute_action()` handles the real action vocabulary and a decision-log writer at the
  `SETUP.md` §7 columns, neither of which existed. Both landed, and `control.controller` now
  imports without a kubernetes client — so the demo can drive the **real** controller with
  `DRY_RUN` left at `True` rather than a stand-in, and say honestly that the loop is closed.

## What this suite deliberately does not do

It does not tell you whether Prodrome **works**. That question is the control-arm
comparison — identical workloads and identical faults, with and without Prodrome — and it
needs a live cluster (`PRD.md` Phase 4). These tests protect the contracts so that the
experiment, when it runs, is measuring what it thinks it is measuring.

Nor does it assert accuracy thresholds. Pinning a number to a dataset that is still
changing would produce a test that fails for the right reason at the wrong time.
