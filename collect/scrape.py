"""Export a time range of the eight frozen metrics from Prometheus to Parquet.

Part 3.2 of docs/guides/shaurya.md. Prometheus already stores everything with
7-day retention, so this does NOT poll -- it asks for a window after the fact
with query_range. A crash at 3am costs nothing; you just re-run it in the
morning over the range you wanted.

The eight queries are the frozen list in collect/README.md (which corrects two
that don't work as written in the guide -- see the NOTE there). Output schema
matches SETUP.md section 7:

    ts, workload, cpu_cores, mem_bytes, mem_pct, net_rx, net_tx,
    fs_reads, fs_writes, restarts

one row per (workload, 15s tick). `workload` is a stable name -- `redis`, not
`redis-65c4779958-zgprh` -- so Sagar's per-workload detector doesn't reset every
time a pod restarts (guide section 3.3).

Usage:
    # Prometheus must be reachable (the port-forward from guide section 2.2):
    kubectl port-forward -n monitoring \\
        svc/monitoring-kube-prometheus-prometheus 9090:9090 &

    python collect/scrape.py --minutes 10 --out data/healthy/metrics.parquet
    python collect/scrape.py --start 2026-08-29T02:00:00Z --end 2026-08-29T14:00:00Z \\
        --out data/healthy/metrics.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

PROM_URL = "http://localhost:9090"
NAMESPACE = "prodrome"
STEP_SECONDS = 15

# metric name -> PromQL template ({ns} filled with the namespace).
# Mirrors "The eight metrics -- FROZEN" in collect/README.md. Keep the two in sync.
QUERIES = {
    "cpu_cores": 'rate(container_cpu_usage_seconds_total{{namespace="{ns}",container!=""}}[1m])',
    "mem_bytes": 'container_memory_working_set_bytes{{namespace="{ns}",container!=""}}',
    "mem_pct": (
        'container_memory_working_set_bytes{{namespace="{ns}",container!=""}}'
        " / on(namespace,pod,container) "
        'kube_pod_container_resource_limits{{namespace="{ns}",resource="memory"}}'
    ),
    "net_rx": 'sum by (namespace,pod) (rate(container_network_receive_bytes_total{{namespace="{ns}"}}[1m]))',
    "net_tx": 'sum by (namespace,pod) (rate(container_network_transmit_bytes_total{{namespace="{ns}"}}[1m]))',
    "fs_reads": 'rate(container_fs_reads_bytes_total{{namespace="{ns}",container!=""}}[1m])',
    "fs_writes": 'rate(container_fs_writes_bytes_total{{namespace="{ns}",container!=""}}[1m])',
    "restarts": 'kube_pod_container_status_restarts_total{{namespace="{ns}"}}',
}

METRIC_COLUMNS = ["ts", "workload", "cpu_cores", "mem_bytes", "mem_pct",
                  "net_rx", "net_tx", "fs_reads", "fs_writes", "restarts"]
VALUE_COLUMNS = METRIC_COLUMNS[2:]
# stored as int64 to stay drop-in compatible with the synthetic fixture (data/samples/)
INT_COLUMNS = ["mem_bytes", "net_rx", "net_tx", "fs_reads", "fs_writes", "restarts"]


def workload_name(labels: dict) -> str | None:
    """Stable name: the `container` label if present, else `pod` minus its two hash segments."""
    container = labels.get("container")
    if container:
        return container
    pod = labels.get("pod")
    if not pod:
        return None
    parts = pod.split("-")
    return "-".join(parts[:-2]) if len(parts) > 2 else pod


def query_range(prom_url: str, expr: str, start: float, end: float, step: int) -> list[dict]:
    resp = requests.get(
        f"{prom_url}/api/v1/query_range",
        params={"query": expr, "start": start, "end": end, "step": step},
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "success":
        raise RuntimeError(f"query failed: {body.get('error', body)}")
    return body["data"]["result"]


def scrape(prom_url: str, namespace: str, start: float, end: float, step: int) -> pd.DataFrame:
    """Run all eight query_range calls, flatten to long, pivot to wide."""
    long_rows: list[tuple] = []
    for metric, template in QUERIES.items():
        series = query_range(prom_url, template.format(ns=namespace), start, end, step)
        mapped = 0
        for s in series:
            workload = workload_name(s["metric"])
            if workload is None:
                continue
            mapped += 1
            for ts, value in s["values"]:
                long_rows.append((pd.Timestamp(float(ts), unit="s", tz="UTC"),
                                  workload, metric, float(value)))
        print(f"  {metric:10} {len(series):3} series -> {mapped} workload-mapped")

    long = pd.DataFrame(long_rows, columns=["ts", "workload", "metric", "value"])
    if long.empty:
        raise RuntimeError(
            "no data for any query -- is the port-forward up, and is the range inside "
            "Prometheus retention (7d)?"
        )

    wide = (long.groupby(["ts", "workload", "metric"])["value"].mean()  # collapse any dup
                .unstack("metric")
                .reset_index())
    for col in METRIC_COLUMNS:
        if col not in wide.columns:
            wide[col] = pd.NA
    return wide[METRIC_COLUMNS].sort_values(["workload", "ts"]).reset_index(drop=True)


def fill_gaps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """ffill/bfill short gaps within each workload. Returns (df, na-fraction-before-fill)."""
    na_before = df[VALUE_COLUMNS].isna().mean()
    df = df.copy()
    df[VALUE_COLUMNS] = df.groupby("workload")[VALUE_COLUMNS].transform(lambda g: g.ffill().bfill())
    return df, na_before


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in INT_COLUMNS:
        df[col] = df[col].fillna(0).round().astype("int64")
    df["cpu_cores"] = df["cpu_cores"].astype(float).round(6)
    df["mem_pct"] = df["mem_pct"].astype(float).round(6)
    df["ts"] = df["ts"].astype("datetime64[us, UTC]")  # match the synthetic fixture exactly
    return df


def sanity_check(df: pd.DataFrame, minutes: float) -> None:
    """The guide section 3.4 checks."""
    print("\n--- sanity check (guide 3.4) ---")
    print("shape        ", df.shape)
    print("workloads    ", sorted(df["workload"].unique()))
    print("span         ", df["ts"].min(), "->", df["ts"].max())
    print("na fraction:")
    print(df[VALUE_COLUMNS].isna().mean().to_string())
    print("head:")
    print(df.head().to_string())

    approx = int(minutes * 60 // STEP_SECONDS) * df["workload"].nunique()
    assert set(df["workload"].unique()) <= {"redis", "nginx", "postgres"}, df["workload"].unique()
    assert list(df.columns) == METRIC_COLUMNS, df.columns.tolist()
    assert df["mem_pct"].notna().any(), "mem_pct entirely missing -- query 3 is broken"
    assert 0.5 * approx <= len(df) <= 1.5 * approx, f"expected ~{approx} rows, got {len(df)}"
    print(f"\nOK  (~{approx} rows expected for {minutes:.0f} min x {df['workload'].nunique()} workloads)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prom-url", default=PROM_URL)
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--minutes", type=float, default=10, help="window ending at --end, in minutes")
    ap.add_argument("--start", help="ISO8601; overrides --minutes")
    ap.add_argument("--end", help="ISO8601; defaults to now")
    ap.add_argument("--step", type=int, default=STEP_SECONDS)
    ap.add_argument("--out", default="data/healthy/metrics.parquet")
    ap.add_argument("--no-fill", action="store_true", help="leave gaps as NaN")
    args = ap.parse_args()

    end = pd.Timestamp(args.end).timestamp() if args.end else time.time()
    start = pd.Timestamp(args.start).timestamp() if args.start else end - args.minutes * 60
    span_min = (end - start) / 60

    print(f"scraping {args.namespace}  {pd.Timestamp(start, unit='s', tz='UTC')} -> "
          f"{pd.Timestamp(end, unit='s', tz='UTC')}  @ {args.step}s")
    df = scrape(args.prom_url, args.namespace, start, end, args.step)

    if not args.no_fill:
        df, na_before = fill_gaps(df)
        filled = na_before[na_before > 0]
        if not filled.empty:
            print("\ngaps filled (na fraction before fill) -- tell Sagar which columns:")
            print(filled.to_string())

    df = coerce(df)
    sanity_check(df, span_min)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  {df.shape}")


if __name__ == "__main__":
    main()
