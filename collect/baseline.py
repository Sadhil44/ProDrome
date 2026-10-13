"""Collect a HEALTHY baseline of the eight frozen metrics with no Prometheus.

Part 3.2's job (`collect/scrape.py`) done the hard way. `scrape.py` asks
Prometheus for a window after the fact; that is the right tool and this is not
a replacement for it. But as of 2026-10-12 **there is no Prometheus on this
cluster** -- the `monitoring` namespace is empty and `helm list -A` returns no
releases -- so `scrape.py` has nothing to query and `data/healthy/metrics.parquet`
cannot be produced the documented way.

cAdvisor is still there. It always was: Prometheus never had a private source,
it scraped the kubelet. So this module polls the same endpoint Prometheus
polled, through the API server proxy (no port-forward, no extra component):

    /api/v1/nodes/<node>/proxy/metrics/cadvisor

and does by hand the three things Prometheus did for us:

  1. `rate(...[1m])` -- cumulative counters are differenced against the sample
     RATE_WINDOW_SECONDS back in our own ring buffer. The first few ticks of a
     run therefore have no rate and are dropped, exactly as a `[1m]` range
     vector is undefined for the first minute of a series.
  2. the `mem_pct` join -- Prometheus divided by
     `kube_pod_container_resource_limits`, which comes from kube-state-metrics,
     which is also gone. The limit is read from the pod spec instead. Same
     number from its source of truth rather than from a re-export of it.
  3. `changes(restarts[1m])` -- `restartCount` from pod status, differenced
     over the same window, clamped to the transient 0/1 the contract wants.

OUTPUT IS THE FROZEN SCHEMA (SETUP.md 7), byte-compatible with `scrape.py`:

    ts, workload, cpu_cores, mem_bytes, mem_pct, net_rx, net_tx,
    fs_reads, fs_writes, restarts

same column order, same dtypes (`coerce()` is imported from `scrape.py` rather
than copied, so the two cannot drift), same 15s cadence, `workload` a stable
name. A consumer cannot tell which collector wrote the file, which is the point.

WHAT THIS DOES NOT DO: it does not inject anything. A healthy baseline is
metrics collected with no faults, and the only thing that should be running
beside it is the load generator (`collect/load/run-load.sh`), so the frame
shows a serving system rather than an idle one.

Usage:
    python -m collect.baseline --minutes 40 --out data/healthy/metrics.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import pandas as pd

from collect.scrape import METRIC_COLUMNS, VALUE_COLUMNS, coerce, workload_name

NAMESPACE = "prodrome"
STEP_SECONDS = 15

# Must match the `[1m]` in collect/README.md's frozen query list. A rate
# computed over a different window is a different metric with the same name.
RATE_WINDOW_SECONDS = 60

# kubectl through WSL on win32, for the same reason chaos.py does it: the
# kubeconfig lives inside WSL and a bare kubectl from Windows silently falls
# back to localhost:8080. Same env override as chaos.py so one variable moves
# every collector at once.
KUBECTL_CMD = (["wsl.exe", "-e", "kubectl"] if sys.platform == "win32" else ["kubectl"])

# cAdvisor counter -> which of the eight it feeds. Counters only; `mem_bytes`
# is a gauge and `restarts`/`mem_pct` come from elsewhere.
COUNTERS = {
    "container_cpu_usage_seconds_total": "cpu_cores",
    "container_fs_reads_bytes_total": "fs_reads",
    "container_fs_writes_bytes_total": "fs_writes",
    "container_network_receive_bytes_total": "net_rx",
    "container_network_transmit_bytes_total": "net_tx",
}
GAUGE = "container_memory_working_set_bytes"

# Metrics 4 and 5 are per-INTERFACE and carry no usable `container` label, so
# they are summed per pod -- collect/README.md's `sum by (namespace,pod)`.
POD_LEVEL = {"net_rx", "net_tx"}

_SAMPLE = re.compile(r'^([a-z_]+)\{([^}]*)\}\s+([0-9.eE+-]+)')


def kubectl(*args: str, timeout: int = 30) -> str:
    out = subprocess.run([*KUBECTL_CMD, *args], capture_output=True, text=True,
                         timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {out.stderr.strip()[:300]}")
    return out.stdout


def parse_labels(blob: str) -> dict:
    return {m.group(1): m.group(2)
            for m in re.finditer(r'(\w+)="((?:[^"\\]|\\.)*)"', blob)}


def scrape_node(node: str, namespace: str) -> dict:
    """One node's cAdvisor text -> {(workload, metric): raw value}.

    Container-level series are keyed on the `container` label; pod-level ones
    (network) are SUMMED across interfaces first, which is the only reason this
    returns an accumulator dict rather than a plain mapping.
    """
    text = kubectl("get", "--raw",
                   f"/api/v1/nodes/{node}/proxy/metrics/cadvisor", timeout=60)
    acc: dict[tuple[str, str], float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        name, blob, raw = m.group(1), m.group(2), m.group(3)
        if name not in COUNTERS and name != GAUGE:
            continue
        labels = parse_labels(blob)
        if labels.get("namespace") != namespace:
            continue

        metric = COUNTERS.get(name, "mem_bytes")
        if metric in POD_LEVEL:
            # no container label on these; derive the name from the pod
            workload = workload_name({"pod": labels.get("pod", "")})
        else:
            # container="" is the pod sandbox (the `pause` container), which
            # double-counts the pod if summed in. The frozen queries all carry
            # container!="" for exactly this reason.
            if not labels.get("container"):
                continue
            workload = labels["container"]
        if not workload:
            continue

        key = (workload, metric)
        try:
            value = float(raw)
        except ValueError:
            continue
        if metric in POD_LEVEL:
            acc[key] = acc.get(key, 0.0) + value   # sum across interfaces
        else:
            acc[key] = max(acc.get(key, 0.0), value)
    return acc


def pod_facts(namespace: str) -> tuple[dict, dict]:
    """(memory limit bytes, cumulative restartCount) per workload, from the API.

    Replaces kube-state-metrics, which this cluster no longer runs. The limit
    is read from the pod SPEC -- the thing kube-state-metrics was re-exporting
    -- so `mem_pct` keeps its contract meaning, working set / limit, 0-1.
    """
    data = json.loads(kubectl("get", "pods", "-n", namespace, "-o", "json"))
    limits, restarts = {}, {}
    for pod in data.get("items", []):
        for c in pod["spec"].get("containers", []):
            lim = c.get("resources", {}).get("limits", {}).get("memory")
            if lim:
                limits[c["name"]] = parse_quantity(lim)
        for cs in pod.get("status", {}).get("containerStatuses", []) or []:
            restarts[cs["name"]] = int(cs.get("restartCount", 0))
    return limits, restarts


_SUFFIX = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
           "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def parse_quantity(q: str) -> float:
    for suffix, mult in _SUFFIX.items():
        if q.endswith(suffix):
            return float(q[: -len(suffix)]) * mult
    return float(q)


def nodes() -> list[str]:
    out = kubectl("get", "nodes", "-o",
                  "jsonpath={range .items[*]}{.metadata.name}{\"\\n\"}{end}")
    return [n.strip() for n in out.splitlines() if n.strip()]


def tick(node_list: list[str], namespace: str) -> tuple[pd.Timestamp, dict, dict, dict]:
    acc: dict[tuple[str, str], float] = {}
    for node in node_list:
        for key, value in scrape_node(node, namespace).items():
            # a workload lives on exactly one node, so this is a merge not a sum
            acc[key] = acc.get(key, 0.0) + value
    limits, restarts = pod_facts(namespace)
    return pd.Timestamp.now(tz="UTC"), acc, limits, restarts


def rows_from_window(now: pd.Timestamp, acc: dict, limits: dict, restarts: dict,
                     past_ts: pd.Timestamp, past_acc: dict, past_restarts: dict) -> list[dict]:
    """Differentiate `acc` against the sample a rate-window ago -> metric rows."""
    elapsed = (now - past_ts).total_seconds()
    if elapsed <= 0:
        return []

    workloads = sorted({w for (w, _) in acc} | {w for (w, _) in past_acc})
    rows = []
    for w in workloads:
        row = {"ts": now, "workload": w}
        for metric in COUNTERS.values():
            cur, prev = acc.get((w, metric)), past_acc.get((w, metric))
            if cur is None or prev is None:
                row[metric] = pd.NA
            else:
                # a counter that went backwards is a container restart; a rate
                # across a reset is meaningless, so report the post-reset slope
                delta = cur - prev if cur >= prev else cur
                row[metric] = max(delta / elapsed, 0.0)

        mem = acc.get((w, "mem_bytes"))
        row["mem_bytes"] = mem if mem is not None else pd.NA
        limit = limits.get(w)
        row["mem_pct"] = (mem / limit) if (mem is not None and limit) else pd.NA

        # changes(...[1m]): did restartCount move inside the window?
        cur_r, prev_r = restarts.get(w), past_restarts.get(w)
        row["restarts"] = (1 if (cur_r is not None and prev_r is not None and cur_r > prev_r)
                           else 0)
        rows.append(row)
    return rows


def align_to_grid(df: pd.DataFrame, step: int = STEP_SECONDS) -> pd.DataFrame:
    """Snap a polled series onto an exact `step`-second grid, per workload.

    THIS IS NOT COSMETIC. `scrape.py`'s output lands on exact 15s boundaries
    because Prometheus `query_range` does not return raw scrapes -- it
    evaluates the series at each step boundary and hands back a regular grid.
    `data/chaos/metrics.parquet` is consequently 15.00s on every single
    interval. A live poller is not: this one achieved a 14.3s median with a
    12.3-16.3s spread, because each tick costs several kubectl round-trips.

    Left alone that is a 4.7% systematic error in every per-tick slope feature
    (`features.summarize` fits a slope against `np.arange(len(window))`, so it
    assumes the ticks are evenly spaced and 15s apart). A detector fit on
    healthy data at 14.3s/tick and then run against chaos data at 15.0s/tick
    is comparing two slightly different units and nothing in the contract would
    reveal it.

    The resampling rule is Prometheus's own: each grid point takes the most
    recent sample AT OR BEFORE it (`merge_asof` backward), not an
    interpolation, so no value is ever invented -- only repeated. Grid points
    before the first real sample have nothing to carry forward and are dropped.
    """
    out = []
    for workload, g in df.groupby("workload"):
        g = g.sort_values("ts")
        start = g["ts"].min().ceil(f"{step}s")
        end = g["ts"].max().floor(f"{step}s")
        if start > end:
            continue
        grid = pd.DataFrame({"ts": pd.date_range(start, end, freq=f"{step}s")})
        snapped = pd.merge_asof(grid, g.drop(columns=["workload"]), on="ts",
                                direction="backward")
        snapped.insert(1, "workload", workload)
        out.append(snapped.dropna(subset=VALUE_COLUMNS, how="all"))

    aligned = pd.concat(out, ignore_index=True)
    return aligned[METRIC_COLUMNS].sort_values(["workload", "ts"]).reset_index(drop=True)


def collect(minutes: float, namespace: str = NAMESPACE, step: int = STEP_SECONDS,
            rate_window: int = RATE_WINDOW_SECONDS) -> pd.DataFrame:
    node_list = nodes()
    print(f"nodes: {node_list}")
    print(f"collecting {namespace} for {minutes:.0f} min @ {step}s "
          f"(rate window {rate_window}s)")

    depth = max(int(round(rate_window / step)), 1)
    history: deque = deque(maxlen=depth + 1)
    collected: list[dict] = []
    deadline = time.time() + minutes * 60
    n = 0

    while time.time() < deadline:
        start = time.time()
        try:
            now, acc, limits, restarts = tick(node_list, namespace)
        except Exception as exc:                      # a flaky API call is not
            print(f"  tick failed ({exc}); continuing")   # a reason to lose a run
            time.sleep(max(0.0, step - (time.time() - start)))
            continue

        history.append((now, acc, restarts))
        if len(history) > depth:
            past_ts, past_acc, past_restarts = history[0]
            new = rows_from_window(now, acc, limits, restarts,
                                   past_ts, past_acc, past_restarts)
            collected.extend(new)
            n += 1
            if n % 20 == 1:
                left = (deadline - time.time()) / 60
                print(f"  [{now:%H:%M:%S}] {len(collected):5} rows  "
                      f"{len(new)} workloads  {left:.1f} min left")

        time.sleep(max(0.0, step - (time.time() - start)))

    if not collected:
        raise RuntimeError("no rows collected -- was the run shorter than the rate window?")
    df = pd.DataFrame(collected)
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[METRIC_COLUMNS].sort_values(["workload", "ts"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--step", type=int, default=STEP_SECONDS)
    ap.add_argument("--rate-window", type=int, default=RATE_WINDOW_SECONDS)
    ap.add_argument("--out", default="data/healthy/metrics.parquet")
    ap.add_argument("--no-align", action="store_true",
                    help="keep raw poll timestamps instead of snapping to the step grid")
    args = ap.parse_args()

    df = collect(args.minutes, args.namespace, args.step, args.rate_window)
    if not args.no_align:
        before = len(df)
        df = align_to_grid(df, args.step)
        print(f"\naligned to a {args.step}s grid: {before} -> {len(df)} rows")
    df = coerce(df)                               # scrape.py's dtypes, not a copy

    print("\n--- sanity check (guide 3.4) ---")
    print("shape        ", df.shape)
    print("workloads    ", sorted(df["workload"].unique()))
    print("span         ", df["ts"].min(), "->", df["ts"].max())
    print("ticks/workload", df.groupby("workload").size().to_dict())
    print("na fraction:")
    print(df[VALUE_COLUMNS].isna().mean().to_string())
    print("describe:")
    print(df[VALUE_COLUMNS].describe().to_string())

    assert list(df.columns) == METRIC_COLUMNS, df.columns.tolist()
    assert set(df["workload"].unique()) <= {"redis", "nginx", "postgres"}, \
        df["workload"].unique()
    assert df["mem_pct"].notna().any(), "mem_pct entirely missing -- limit lookup broke"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  {df.shape}")


if __name__ == "__main__":
    main()
