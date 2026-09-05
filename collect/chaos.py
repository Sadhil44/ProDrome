"""Chaos injection + labels -- Part 5 of docs/guides/shaurya.md.

The injector knows exactly what it started and when, so every metric window it
covers is labeled by construction (guide 5.1) -- no hand annotation.

Mechanism: `kubectl exec ... stress-ng` (guide 5.2). No chaos framework.

Four faults, the first three in a `constant` and a `ramp` pattern (guide 5.3-5.4):

    CPU_HOG      stress-ng --cpu 2 --cpu-load P        ramp P: 20..100
    MEMORY_LEAK  stress-ng --vm 1 --vm-bytes NM ...    ramp N: 0.4x..1.3x mem limit
    DISK_STRESS  stress-ng --hdd 1 --hdd-bytes NM      ramp N: 32..512 MB
    POD_KILL     kubectl delete pod                    (unpredictable, ~0 lead time)

`ramp` = successive stress-ng invocations of rising intensity, because stress-ng
cannot ramp on its own (guide 5.4). `constant` jumps straight to full intensity.

Outputs (data/chaos/):

    labels.csv   start_ts,end_ts,workload,fault_type,pattern,run_id   (SETUP.md 7)
                 one row per FAULT run -- this is what Sadhil consumes. FROZEN
                 contract: exactly these six columns.
    runs.csv     full manifest incl. clean runs, restart deltas, and failure_ts
                 (the crisp OOMKill / kill instant for MEMORY_LEAK and POD_KILL;
                 blank for CPU_HOG / DISK_STRESS, which don't fail) -- for the
                 Part 7 eval harness. Not a cross-pair contract; shape is mine.

Both files are APPENDED after every run, so a crash partway through is
recoverable: re-run `--campaign` and it skips the runs already in runs.csv.
To start a fresh campaign, delete data/chaos/ first.

Metrics are NOT collected here. After the campaign, export the window with the
Part 3.2 scraper (this script prints the exact command):

    python collect/scrape.py --start <ISO> --end <ISO> --out data/chaos/metrics.parquet

Run the load generator alongside this -- faults must land against realistic
varying load, not an idle cluster (same reason as the healthy data, guide 4.1).

Usage:
    python collect/chaos.py --campaign                       # full ~2.5h run (guide 5.5)
    python collect/chaos.py --one CPU_HOG redis ramp         # single run, for testing
    python collect/chaos.py --one POD_KILL nginx
    python collect/chaos.py --campaign --duration 60 --gap 20   # fast schedule dry-run
"""
from __future__ import annotations

import argparse
import random
import subprocess
import time
from pathlib import Path

import pandas as pd

NAMESPACE = "prodrome"
WORKLOADS = ["redis", "nginx", "postgres"]
MEM_LIMIT_MI = {"redis": 512, "nginx": 256, "postgres": 512}  # from infra/workloads.yaml
FAULTS = ["CPU_HOG", "MEMORY_LEAK", "DISK_STRESS"]
PATTERNS = ["constant", "ramp"]
RAMP_STEPS = 5

DEFAULT_DURATION = 300   # 5 min per run (guide 5.5)
DEFAULT_GAP = 120        # recovery between runs

LABEL_COLUMNS = ["start_ts", "end_ts", "workload", "fault_type", "pattern", "run_id"]
RUNS_COLUMNS = ["run_id", "run_type", "workload", "fault_type", "pattern",
                "start_ts", "end_ts", "failure_ts", "restarts_delta"]


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _kubectl(*args: str, timeout: float = 60, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True,
                          timeout=timeout, check=check)


def restart_count(workload: str, ns: str) -> int:
    r = _kubectl("get", "pods", "-n", ns, "-l", f"app={workload}",
                 "-o", "jsonpath={.items[*].status.containerStatuses[*].restartCount}")
    return sum(int(x) for x in r.stdout.split()) if r.stdout.strip() else 0


def container_mem_pct(workload: str, ns: str) -> float:
    """cgroup-v2 memory.current / memory.max, read straight from the pod (no
    Prometheus dependency). Used only to mark a MEMORY_LEAK failure instant."""
    r = _kubectl("exec", "-n", ns, f"deploy/{workload}", "--", "sh", "-c",
                 "cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max", timeout=15)
    try:
        cur, mx = (int(x) for x in r.stdout.split()[:2])
        return cur / mx
    except (ValueError, ZeroDivisionError):
        return 0.0


def wait_ready(workload: str, ns: str, timeout: int = 180) -> None:
    _kubectl("wait", "--for=condition=ready", "pod", "-l", f"app={workload}",
             "-n", ns, f"--timeout={timeout}s", timeout=timeout + 10)


def _stress(workload: str, stress_args: list[str], seconds: int, ns: str) -> None:
    """One stress-ng invocation inside the workload pod. Hard-stops if it outlives
    its own --timeout (guide 5.2: stress-ng sometimes does)."""
    try:
        _kubectl("exec", "-n", ns, f"deploy/{workload}", "--", "stress-ng", *stress_args,
                 timeout=seconds + 30)
    except subprocess.TimeoutExpired:
        print(f"      stress-ng exceeded {seconds + 30}s wall -- killing, moving on")
        _kubectl("exec", "-n", ns, f"deploy/{workload}", "--", "pkill", "-9", "stress-ng",
                 timeout=30)


# --- fault plans: (stress_args, seconds) steps -------------------------------

def plan_cpu(workload: str, pattern: str, duration: int) -> list[tuple[list[str], int]]:
    if pattern == "constant":
        return [(["--cpu", "2", "--cpu-load", "100", "--timeout", f"{duration}s"], duration)]
    step = duration // RAMP_STEPS
    return [(["--cpu", "2", "--cpu-load", str(load), "--timeout", f"{step}s"], step)
            for load in (20, 40, 60, 80, 100)]


def plan_mem(workload: str, pattern: str, duration: int) -> list[tuple[list[str], int]]:
    limit = MEM_LIMIT_MI[workload]
    # Both patterns CLIMB toward the ceiling so there's a trajectory to label
    # (guide 6.1) -- a single huge --vm-bytes just OOMKills in ~1s, invisible at
    # 15s scrape resolution. constant climbs fast (front-loaded), ramp climbs
    # slowly across the whole window.
    if pattern == "constant":
        fracs = (0.80, 1.00, 1.20)
        step = max(duration // 6, 20)          # fast onset: ~first half of the window
    else:
        fracs = (0.40, 0.65, 0.90, 1.10, 1.30)
        step = max(duration // RAMP_STEPS, 30)  # slow slide across the window
    # --vm-keep holds RSS pegged at --vm-bytes (vs oscillating), so mem_pct sits
    # at the intended level for the whole step -- a clean trajectory. Above the
    # limit it OOMs; the cgroup killer often reaps stress-ng rather than the app,
    # so `restarts` fires inconsistently for MEMORY_LEAK -- the rising mem_pct is
    # the reliable signal. See collect/README-ish note in the chaos docstring.
    return [(["--vm", "1", "--vm-bytes", f"{int(limit * fr)}M", "--vm-keep",
              "--timeout", f"{step}s"], step)
            for fr in fracs]


def plan_disk(workload: str, pattern: str, duration: int) -> list[tuple[list[str], int]]:
    if pattern == "constant":
        return [(["--hdd", "2", "--hdd-bytes", "512M", "--timeout", f"{duration}s"], duration)]
    step = duration // RAMP_STEPS
    return [(["--hdd", "1", "--hdd-bytes", f"{mb}M", "--timeout", f"{step}s"], step)
            for mb in (32, 96, 192, 320, 512)]


PLANS = {"CPU_HOG": plan_cpu, "MEMORY_LEAK": plan_mem, "DISK_STRESS": plan_disk}


# --- runs -------------------------------------------------------------------

def run_fault(fault: str, workload: str, pattern: str, duration: int, ns: str) -> dict:
    steps = PLANS[fault](workload, pattern, duration)
    r0 = restart_count(workload, ns)
    start = _now()
    print(f"  {fault:12} {workload:9} {pattern:8} -- {len(steps)} step(s), ~{duration}s")

    oomkilled = False
    failure_ts = None   # crisp "about to die / died" instant; runs.csv only, NOT labels.csv
    for i, (args, secs) in enumerate(steps, 1):
        if len(steps) > 1:
            print(f"      step {i}/{len(steps)}: stress-ng {' '.join(args)}")
        t0 = _now()
        _stress(workload, args, secs, ns)
        if fault != "MEMORY_LEAK" or failure_ts is not None:
            continue
        # The cgroup killer often reaps stress-ng rather than the container, and
        # restartCount lags the kubelet, so take the earliest of three signals as
        # the failure instant: a container restart, memory pegged near the
        # ceiling, or stress-ng exiting well before its own --timeout (it died).
        early = (_now() - t0).total_seconds() < secs - 5
        if restart_count(workload, ns) > r0:
            print(f"      OOMKilled at step {i}/{len(steps)}")
            oomkilled = True
            failure_ts = _now()
            break
        if early or container_mem_pct(workload, ns) >= 0.95:
            why = "stress-ng died early" if early else "mem pegged >=95%"
            print(f"      {why} at step {i}/{len(steps)}")
            failure_ts = _now()

    end = _now()
    if fault == "MEMORY_LEAK":
        wait_ready(workload, ns)          # back from an OOMKill before the next run
        time.sleep(3)                     # let the kubelet publish the new restartCount
    delta = max(restart_count(workload, ns) - r0, 0)
    if fault == "MEMORY_LEAK" and failure_ts is None and delta > 0:
        failure_ts = end                  # it restarted at some point in the run
    print(f"      done  ({(end - start).total_seconds():.0f}s, restarts +{delta}"
          f"{', OOMKilled' if oomkilled else ''})")
    return dict(start_ts=start, end_ts=end, workload=workload, fault_type=fault,
               pattern=pattern, restarts_delta=delta, failure_ts=failure_ts)


def run_pod_kill(workload: str, ns: str) -> dict:
    start = _now()
    print(f"  {'POD_KILL':12} {workload:9}")
    _kubectl("delete", "pod", "-n", ns, "-l", f"app={workload}",
             "--grace-period=0", "--force", timeout=60)
    wait_ready(workload, ns, timeout=180)
    end = _now()
    print(f"      recovered in {(end - start).total_seconds():.0f}s")
    return dict(start_ts=start, end_ts=end, workload=workload, fault_type="POD_KILL",
               pattern="none", restarts_delta=0, failure_ts=start)  # the kill IS the failure


def _recover(seconds: int, label: str = "recovery") -> None:
    if seconds > 0:
        print(f"  ... {label} {seconds}s")
        time.sleep(seconds)


def _plan(rounds: int = 1, fault_types: list[str] | None = None) -> list[dict]:
    """The full campaign as an ordered list of steps, each with a stable run_id.
    Deterministic (incl. which 2 workloads get pod-killed) so a re-run resumes
    cleanly instead of duplicating or renumbering.

    `rounds` repeats the whole schedule N times -- run_ids just keep counting
    (CPU_HOG_constant_redis_001, _002, ...), so `--rounds 3` after a `--rounds 1`
    run resumes and adds rounds 2-3. More rounds = multiple runs per fault type,
    which is what lets Sadhil split train/test by run (guide 5.6).

    `fault_types` restricts the FAULTS loop to a subset (e.g. ["DISK_STRESS"]
    to top up just one fault type that turned out contaminated -- see guide
    5.5/eval/README.md) and skips the clean/pod-kill blocks entirely, since a
    targeted top-up shouldn't add more of those. Numbering still starts at
    _001 for that fault, so it resumes past whatever's already in runs.csv
    exactly like a full-schedule rerun does -- existing (contaminated) runs
    for that fault get skipped by campaign()'s normal resume check, and only
    genuinely new ones run."""
    steps: list[dict] = []
    seen: dict[tuple, int] = {}
    clean_i = 0
    faults = fault_types if fault_types else FAULTS
    killed = random.Random(0).sample(WORKLOADS, 2)

    def rid(fault: str, workload: str, pattern: str) -> str:
        key = (fault, pattern, workload)
        seen[key] = seen.get(key, 0) + 1
        return f"{fault}_{pattern}_{workload}_{seen[key]:03d}"

    for _ in range(rounds):
        for workload in WORKLOADS:
            for fault in faults:
                for pattern in PATTERNS:
                    steps.append(dict(kind="fault", fault=fault, workload=workload,
                                      pattern=pattern, run_id=rid(fault, workload, pattern)))
        if fault_types:
            continue                                  # targeted top-up: skip clean/pod-kill
        for _ in range(4):                            # 4 fault-free windows / round
            clean_i += 1
            steps.append(dict(kind="clean", run_id=f"CLEAN_{clean_i:03d}"))
        for workload in killed:                       # 2 pod kills / round
            steps.append(dict(kind="fault", fault="POD_KILL", workload=workload,
                              pattern="none", run_id=rid("POD_KILL", workload, "none")))
    return steps


def _append(rec: dict, outdir: str) -> None:
    """Append one finished run to runs.csv (and labels.csv if it's a fault), writing
    headers on first touch. A crash mid-campaign then keeps everything done so far."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    runs_p = Path(outdir) / "runs.csv"
    pd.DataFrame([{c: rec.get(c) for c in RUNS_COLUMNS}]).to_csv(
        runs_p, mode="a", header=not runs_p.exists(), index=False)
    if rec.get("run_type") == "fault":
        labels_p = Path(outdir) / "labels.csv"       # frozen SETUP.md 7 contract
        pd.DataFrame([{c: rec[c] for c in LABEL_COLUMNS}]).to_csv(
            labels_p, mode="a", header=not labels_p.exists(), index=False)


def campaign(duration: int, gap: int, ns: str, outdir: str, rounds: int = 1,
            fault_types: list[str] | None = None) -> None:
    done: set[str] = set()
    runs_p = Path(outdir) / "runs.csv"
    if runs_p.exists():
        done = set(pd.read_csv(runs_p)["run_id"])
        print(f"resuming -- {len(done)} runs already in {runs_p}, skipping those\n")

    for step in _plan(rounds, fault_types):
        if step["run_id"] in done:
            print(f"  skip {step['run_id']} (done)")
            continue
        if step["kind"] == "clean":
            start = _now()
            _recover(duration, f"{step['run_id']} -- clean window")
            rec = dict(start_ts=start, end_ts=_now(), workload="", fault_type="NONE",
                       pattern="none", run_type="clean", restarts_delta=0, failure_ts=None)
        elif step["fault"] == "POD_KILL":
            rec = run_pod_kill(step["workload"], ns)
            rec["run_type"] = "fault"
        else:
            rec = run_fault(step["fault"], step["workload"], step["pattern"], duration, ns)
            rec["run_type"] = "fault"
        rec["run_id"] = step["run_id"]
        _append(rec, outdir)
        _recover(gap)


def finish(outdir: str) -> None:
    """Read back what the campaign wrote and print the metrics-export command."""
    runs = pd.read_csv(f"{outdir}/runs.csv", parse_dates=["start_ts", "end_ts"])
    labels = pd.read_csv(f"{outdir}/labels.csv", parse_dates=["start_ts", "end_ts"])
    start = runs["start_ts"].min() - pd.Timedelta(seconds=90)   # rate() needs a warmup lead
    end = runs["end_ts"].max() + pd.Timedelta(seconds=60)

    print(f"\n{outdir}/labels.csv  -- {len(labels)} fault runs")
    print(f"{outdir}/runs.csv    -- {len(runs)} runs incl. {(runs.run_type == 'clean').sum()} clean")
    print(f"\ncampaign window (UTC): {start.isoformat()}  ..  {end.isoformat()}")
    print("export the metrics for this window:\n")
    print(f"  python collect/scrape.py --start {start.isoformat()} \\")
    print(f"      --end {end.isoformat()} --out {outdir}/metrics.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", action="store_true", help="run the full campaign (guide 5.5)")
    ap.add_argument("--one", nargs="*", metavar="ARG",
                    help="single run: FAULT WORKLOAD [PATTERN]  (PATTERN omitted for POD_KILL)")
    ap.add_argument("--duration", type=int, default=DEFAULT_DURATION)
    ap.add_argument("--gap", type=int, default=DEFAULT_GAP)
    ap.add_argument("--rounds", type=int, default=1,
                    help="repeat the whole schedule N times (~2.5h each); more runs per fault type")
    ap.add_argument("--fault-types", default=None,
                    help="comma-separated subset (e.g. DISK_STRESS) to top up just that fault "
                         "type; skips clean/pod-kill runs")
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--outdir", default="data/chaos")
    args = ap.parse_args()

    if args.one:
        fault = args.one[0]
        workload = args.one[1]
        if fault == "POD_KILL":
            print(run_pod_kill(workload, args.namespace))
        else:
            pattern = args.one[2] if len(args.one) > 2 else "constant"
            print(run_fault(fault, workload, pattern, args.duration, args.namespace))
    elif args.campaign:
        # runs.csv / labels.csv are appended after every run, so a crash is
        # recoverable: just re-run --campaign and it skips what's already done.
        # To start over, delete data/chaos/ first.
        fault_types = args.fault_types.split(",") if args.fault_types else None
        campaign(args.duration, args.gap, args.namespace, args.outdir, args.rounds, fault_types)
        finish(args.outdir)
    else:
        ap.error("pass --campaign or --one FAULT WORKLOAD [PATTERN]")


if __name__ == "__main__":
    main()
