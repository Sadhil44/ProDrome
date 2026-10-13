# Aahan - Testing and CI

## What you own

`tests/` and `.github/workflows/`. Nothing else.

**You never need a cluster.** Everything you write runs from a fresh clone against
`data/samples/`, which is the only committed data in the repo.

> **Where this fits** (`PRD.md` §10): you are not gated on a phase. The contract suite exists
> now and CI runs it on every push. The deeper work — integration coverage, a test application —
> gets more valuable as the other four land real code, but none of it waits on them.

---

## Part 1 — The job in one sentence

Everyone else builds a stage. You answer:

> **Does the thing actually do what we say it does — and will we find out the moment it stops?**

That is two jobs. The first is coverage. The second is CI, which is what turns a passing test
into a promise rather than a thing someone ran once.

## Part 2 — Why this seat exists

Prodrome's credibility rests on `SETUP.md` §10's ground rules, and most of them are the kind of
rule that is easy to violate by accident and impossible to spot by reading a diff:

- The detector must never train on injected faults.
- Splits must be by run, never by row.
- Nobody may quietly change a schema column.

Written as prose these are reminders. Written as tests they are enforced. Three real bugs in this
repo were invisible to code review and obvious to a test — see "Bugs the suite has already caught"
below.

There is a second, less obvious reason. Four areas exchange **files, not services**
(`SETUP.md` §8), so nothing warns a consumer when a producer changes shape. The forum covers the
human half of that. Tests cover the mechanical half.

## Part 3 — The rule that defines your job

> **A test that cannot run from a fresh clone is not a test.**

This is ground rule 8 applied to yourself. It has a sharp consequence: you may depend on
`data/samples/` and nothing else. `data/chaos/` and `data/healthy/` are gitignored, are absent on
most machines, and are handed over out of band. A test that reads them passes for exactly one
person.

`ml/test_restart_suppression.py` is the cautionary example. It is a genuinely good test of the
post-restart guard, and it cannot run for anyone who has not received a data handoff, so it sits
outside the suite doing nothing. Giving that behaviour a synthetic fixture is real work worth
doing — but the file is signal's, so add your own under `tests/` rather than editing theirs.

## Part 4 — The boundary, which matters more than the coverage

**You assert contracts. You do not renegotiate them.**

These files belong to other people and you do not edit them:

| Files | Owner |
|---|---|
| `ml/detector.py`, `ml/features.py`, `ml/replay.py`, `ml/fit_detector.py` | Sagar (signal) |
| `collect/`, `eval/` | Shaurya (collect) |
| `control/controller.py`, `infra/` | Shravan (cluster) |
| `ml/classifier.py`, `ml/train.py`, `ml/abstention.py`, `control/policy.py`, `dashboard/` | Sadhil (diagnosis) |

Reading them is the whole job. Changing them is somebody else's call.

When a test and an implementation disagree, you have found something. Two ways to get it wrong:

1. **Editing their file** to make your test pass. You have now silently changed behaviour four
   people depend on.
2. **Relaxing your test** to make it pass. You have now deleted the finding.

Do neither. Post it to `crosstalk/questions` addressed to the owning area, and leave the test red
if you believe it is right. A red test that is correct is doing its job.

**The obligation is reciprocal**, and it is on the other four as much as you: if they change
something a test pins, they announce it. They should learn a contract moved from their own
announcement, not from a red build three days later.

## Part 5 — What to pin, and what not to

**Pin:** shapes, orders, column sets, function signatures, invariants, the behaviour of edge
cases. Things that are supposed to be true no matter how the data moves.

**Do not pin accuracy numbers.** Fixing `assert accuracy > 0.8` to a dataset that is still
changing gives you a test that fails for the right reason at the wrong time — and the moment CI
goes red for a reason nobody caused, everyone learns to ignore CI. That is worse than having no
test.

The place for a number is a report with its baseline beside it, not an assertion.

## Part 6 — What exists now

`python -m pytest tests -q`, about a minute, no cluster, no Docker.

- **`test_data_contracts.py`** — the three `SETUP.md` §7 schemas, plus the properties the prose
  asks for: stable workload names (`redis`, not `redis-7d9f8b-x2k1`), non-null `run_id`, both
  fault patterns present.
- **`test_code_contracts.py`** — `ml.features.METRICS` order and `WINDOW_SIZE` exactly as signal
  published them, the 40 summary features, `policy.decide` across the whole table including the
  threshold ordering and the unmapped-class fallthrough.
- **`test_invariants.py`** — §10's ground rules as executable checks, plus regressions for real
  defects.

CI (`.github/workflows/tests.yml`) installs `pandas pyarrow numpy scikit-learn pytest` rather than
`requirements.txt`, which would pull ~2 GB of torch for `ml/cnn.py` that no test imports. If you
add a dependency, add it to the workflow explicitly and say why.

## Part 7 — Bugs the suite has already caught

Worth reading, because each one shows the *shape* of bug tests find here.

**The reported metrics and the shipped model came from different datasets.** `ml.train` defaulted
to `data/samples/` through argparse while `classifier.fit_and_save()` preferred `data/chaos/`. A
0.99 synthetic accuracy and a 0.82 real accuracy were both true at once, and the README quoted the
wrong one.

**`ml/abstention.py` could not be pointed at real data.** It called `load_labeled_windows()` with
no arguments and had no CLI override, so its published safety figure could only ever describe the
fixture. On real data the number is roughly 2.7× worse.

**The classifier artifact loaded from exactly one entrypoint.** `save()` pickled the wrapper
*instance*, so `python -m ml.classifier` recorded its class as `__main__.RandomForestClassifier`.
Every other importer — including `from ml.classifier import classifier`, which `README.md`
documents as *the* interface the controller uses — got `AttributeError`. Found by writing a test
that imports it from pytest, which is by definition not that `__main__`.

None of the three was visible in a diff. All three were obvious to a test.

## Part 8 — Landmines

- **`ml/classifier.py` has an import-time side effect**: importing it loads or refits the pickle.
  Watch the cost, and never let a test depend on a stale `ml/classifier.pkl`.
- **`control/policy.py` has no `POD_KILL` row.** It must fall through to `"nothing"`. That is
  deliberate — instantaneous kills have no diagnosis lever — and it is exactly the kind of thing to
  lock in, because the safe default is easy to break by adding a well-meaning `else`.
- **The live detector barely fires.** Across 160 real fault runs the live path produced 3 firings
  (see `crosstalk/announcements/detector-score-desensitizes-in-the-live-controller-path`). Do not
  write a test asserting the detector fires on real faults — it does not. That is signal's open
  defect, not yours to fix, and not yours to encode as correct behaviour either.
- **The controller cannot currently be tested without a live cluster.** Asking cluster for a
  dry-run mode or an injectable API client is a legitimate request to put on the forum.

## Part 9 — Where to go next

The unit and integration layers are the foundation and come first. Beyond them, the interesting
work — and the reason this seat is yours rather than a script's:

**A test application.** A harness that stands the pipeline up and exercises it as a system rather
than as a collection of functions: feed it a synthetic workload, watch a fault develop, see which
stage fires and what the controller decided, and be able to re-run it identically. Done well it
becomes how Prodrome gets demonstrated, and how a regression gets reproduced. It needs real design
thought — what to fake, what to keep honest, how to make failure legible — which is why it is a
person's job and not something to auto-generate.

**Adversarial cases.** The abstention result says the classifier is confidently wrong on a quarter
of novel-fault windows. What else is it confident and wrong about? That is a testing question
before it is a modelling one.

**The control-arm seam.** `SETUP.md` §10 rule 4 says every claim ships with the comparison against
stock Kubernetes. Nothing currently makes that easy to run twice. Making it repeatable is the
difference between a demo and a result.

---

Coordination happens on the CCP forum (MCP `ccp-forum`, session `prodrome`), shelf `testing`. Post
progress to `testing/updates`; publish what the suite guarantees **and what it deliberately does
not** in `testing/interfaces`. When a test pins another area's contract, tell that area — they
should hear it from you, not from CI. Setup: `forum/HANDOFF.md`.
