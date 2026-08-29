"""Feature summarization: a 20-tick window -> 40 features.

Extends Sagar's shared windowing file (he owns metric order and
windowing; this is the summarization half) - once that file exists,
this should move into it rather than staying standalone. Keyed by
metric name, not array position, so it doesn't depend on whatever
order his windowing code produces.

See docs/guides/sadhil.md Part 2.3.
"""

import numpy as np

METRICS = [
    "cpu_cores",
    "mem_bytes",
    "mem_pct",
    "net_rx",
    "net_tx",
    "fs_reads",
    "fs_writes",
    "restarts",
]


def summarize(window):
    """window: a table with one column per metric in METRICS, 20 rows.

    Returns a flat dict of 40 features: mean, slope, std, max, last
    for each metric.
    """
    features = {}
    ticks = np.arange(len(window))

    for metric in METRICS:
        values = np.asarray(window[metric], dtype=float)
        slope = np.polyfit(ticks, values, 1)[0]

        features[f"{metric}_mean"] = values.mean()
        features[f"{metric}_slope"] = slope
        features[f"{metric}_std"] = values.std()
        features[f"{metric}_max"] = values.max()
        features[f"{metric}_last"] = values[-1]

    return features
