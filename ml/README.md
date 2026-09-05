# ml/

Detector, classifier, shared feature code. Detector owned by Sagar (Signal); classifier and policy owned by Sadhil (Diagnosis) — see [SETUP.md](../SETUP.md) §6.

## Frozen interfaces

Both are pickled artifacts (`*.pkl`, gitignored) with a module-level ready instance and a `fit_and_save()` you can rerun. Neither signature changes without telling the owner first (docs/guides/sagar.md Part 4.5 / sadhil.md).

**Detector** (`ml/detector.py` + `ml/fit_detector.py`, written by Sagar, read by Shravan):
```python
from ml.fit_detector import detector  # loads ml/detector.pkl, fitting it first if missing

score, fired = detector.score(workload_name, array_of_8_values)  # values in ml.features.METRICS order
detector.on_restart(workload_name)  # call after every restart action -- see Part 4.4
```
Refit: `python -m ml.fit_detector`. Prefers `data/healthy/metrics.parquet`, falls back to the bundled sample.

**Classifier** (`ml/classifier.py`, written by Sadhil, read by Shravan):
```python
from ml.classifier import classifier

label, confidence = classifier.predict(window)  # window: WINDOW_SIZE rows x ml.features.METRICS columns
```
Refit: `python -m ml.classifier`. Prefers `data/chaos/`, falls back to the bundled sample.
