# ml/ module rules

Two seats share this directory. `detector.py`, `features.py` and `replay.py` belong to **signal**
(Sagar); `classifier.py`, `dataset.py`, `train.py`, `abstention.py` and `cnn.py` belong to
**diagnosis** (Sadhil). Work on the other seat's file by posting to its owner, not by editing it.

- `features.py` is the canonical shared file and signal owns it. `METRICS` (the metric order) and
  `WINDOW_SIZE` are a contract, not a local choice: the detector and the classifier must summarize a
  window identically, or the controller feeds two different models two different things. Changing
  either means appending to `signal/interfaces` AND announcing it labeled `diagnosis` and `cluster`.
- Never name a module in here after a Python stdlib module. `features.py` was originally `signal.py`
  and shadowed the stdlib `signal` module; that is why it has its current name. The forum shelf is
  still called `signal` — the seat, not the file.
- The detector trains on **healthy data only**, never on injected faults. Injected faults are step
  functions and real degradation is a slide, so a detector trained on them learns to recognise our
  injector. This is not tunable; if a change needs fault labels to work, it belongs in the classifier.
- Split train/test **by run, never by row**. Sliding windows share 19 of their 20 timesteps, so a
  random split trains on near-duplicates of the test set. Splits key on `run_id`. If accuracy comes
  out above ~0.97, assume this bug and check before reporting the number.
- `classifier.predict(window) -> (label, confidence)` is the interface the controller codes against;
  `window` is `WINDOW_SIZE` rows by `METRICS`. The classifier summarizes internally so callers never
  touch features. Do not change this signature without telling cluster.
- Prefer real data over the fixture: read `data/chaos/` when it exists on disk and fall back to
  `data/samples/` otherwise, so nobody is blocked waiting for a collection run.
- Model artifacts (`*.pkl`) are gitignored and refit from the module that owns them. Never commit one,
  and never let a result depend on a pickle that is not reproducible from a fresh clone.
- Report per fault type, never one aggregate, and always next to the baseline it beat. Say plainly
  when a number came from the synthetic fixture: it describes a working pipeline, not a result about
  real failures.
- The known safety gap is live: on a held-out fault class the model abstains on none of the windows
  and is confidently wrong on a quarter of them. Anyone touching confidence thresholds, the label set
  or the classifier reads `abstention.py` first and re-runs it after.
- Coordination happens on the CCP forum (MCP `ccp-forum`, session `prodrome`): set status, post to
  `signal/updates` or `diagnosis/updates`, and append contract changes to
  `signal/interfaces` (window and feature contract, detector interface) or
  `diagnosis/interfaces` (classifier and policy contract). See `forum/HANDOFF.md`.
