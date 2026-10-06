# tests/ module rules

This directory belongs to the **testing** seat (Aahan), along with `.github/workflows/`. The job is
to prove the pipeline does what it claims — and to make a broken claim fail loudly and early.

- **Runs from a fresh clone, always.** No cluster, no Docker, no collection handoff. That means
  `data/samples/` only, because it is the sole committed data. Ground rule 8 says a result that is
  not reproducible from a fresh clone is not a result; the same applies to a test. `pytest tests -q`
  finishes in about a minute and CI runs it on every push.
- **Assert contracts, do not renegotiate them.** The schemas in `SETUP.md` §7 and the interfaces
  published on the forum are owned by other seats. When a test and an implementation disagree, ask
  the owner in `crosstalk/questions` — do not edit their files, and do not quietly relax the test to
  make it pass. A red test that is correct is doing its job.
- **Never edit another seat's implementation.** `ml/detector.py` and `ml/features.py` are signal's;
  `collect/` and `eval/` are collect's; `control/controller.py` and `infra/` are cluster's;
  `ml/classifier.py` and `control/policy.py` are diagnosis's. Reading them is the whole point.
  Changing them is somebody else's call.
- **Pin no accuracy thresholds.** Fixing a number to a dataset that is still changing produces a
  test that fails for the right reason at the wrong time, and trains everyone to ignore CI. Pin
  shapes, orders, column sets, signatures and invariants — the things that are supposed to be true
  regardless of how the data moves.
- **Every test earns its place with a real defect or a stated rule.** No speculative coverage. Each
  one should map to a ground rule in `SETUP.md` §10, a published contract, or a bug that actually
  happened — and say which in a docstring, so a future reader knows whether it still matters.
- **A test that cannot run is not a test.** `ml/test_restart_suppression.py` reads
  `data/healthy/metrics.parquet`, which is gitignored, so it is excluded from the suite and cannot
  run for anyone without a data handoff. Giving it a synthetic fallback is real work worth doing.
- **CI installs only what the suite imports**, not `requirements.txt` — that would pull ~2 GB of
  torch for `ml/cnn.py` which no test touches. If a new test needs another package, add it to the
  workflow explicitly and say why.
- The suite deliberately does **not** answer whether Prodrome works. That is the control-arm
  comparison and it needs a live cluster. What these tests buy is that when that experiment runs, it
  measures what it thinks it measures.
- Coordination happens on the CCP forum (MCP `ccp-forum`, session `prodrome`): set status, post to
  `testing/updates`, and publish what the suite guarantees — and what it deliberately does not — in
  `testing/interfaces`. When a test pins another seat's contract, tell that seat, so they learn it
  from you rather than from a red build. See `forum/HANDOFF.md`.
