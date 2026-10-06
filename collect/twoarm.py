"""Two-arm paired fault injection -- the cluster-touching half of the
control-arm harness (SETUP.md ground rule 4).

Injects the SAME fault into `prodrome` (Prodrome's controller active) and
`control` (stock Kubernetes only) at the same instant, samples both arms'
serving health from one loop at one cadence, and writes the two files
`eval/recovery.py` turns into a recovery-time table.

This module is deliberately thin. Every judgement call -- what recovery time
means per fault type, when a pair is admissible, what makes the arms
comparable -- lives in `eval/recovery.py` and is unit-tested offline. What is
left here is the part that genuinely needs a cluster: two `kubectl exec`s and a
polling loop. Everything cluster-shaped enters through an injected callable
(`injector`, `probe_fn`, `clock`), so `collect/test_twoarm.py` drives the whole
scheduler, the barrier, the skew accounting and the file writers with fakes and
no Docker.

-------------------------------------------------------------------------------
WHY SIMULTANEOUS, AND WHY THE SKEW IS RECORDED RATHER THAN ASSUMED
-------------------------------------------------------------------------------

Running the arms sequentially looks equivalent and is not. The cluster is a
shared, drifting thing: the k6 diurnal load curve is at a different point ten
minutes later, the page cache is warmer, another workload's fault may still be
unwinding. Any of that shows up as a recovery-time difference and gets
attributed to Prodrome.

So both arms go through a `threading.Barrier`, and the harness then records the
skew it actually achieved rather than claiming the zero it intended --
`kubectl exec` round-trips are not deterministic. `eval/recovery.py` excludes
any pair outside `SKEW_TOLERANCE_SECONDS` from the paired delta instead of
averaging it in.

-------------------------------------------------------------------------------
THE GAP BETWEEN PAIRS IS THE SAME LANDMINE AS IN collect/chaos.py
-------------------------------------------------------------------------------

`DEFAULT_GAP` is imported from `collect/chaos.py`, which derives it from
`ml.detector.POST_RESTART_SUPPRESS_TICKS`. That is not tidiness: a pair
scheduled too soon after a restart-prone pair lands inside the detector's
post-restart dead zone, the treatment arm cannot fire, and the control arm
"wins" for a reason that is purely scheduling. The control arm has no detector
and no dead zone, so this bias is one-directional -- it makes Prodrome look
worse, which is the direction nobody checks. Same constant, same reason, one
place.

-------------------------------------------------------------------------------
WHAT THIS DOES NOT DO
-------------------------------------------------------------------------------

It does not start the controller and it does not touch `control/controller.py`.
The controller must already be running against the `prodrome` namespace with
`DRY_RUN = False` before a campaign here means anything; in shadow mode the
treatment arm takes no action and the experiment measures stock Kubernetes
twice. `eval/recovery.py`'s `attest_treatment_arm_live()` is what refuses to
publish that run as a comparison, and `--preflight` below prints it before you
spend the time.

It does not generate load. `collect/load/run-load.sh` has to be pointed at BOTH
namespaces first; `--load-attested` is how you tell the harness you did, and it
is recorded per arm-row so a later reader can see whether anyone actually
checked.

Usage:
    python collect/twoarm.py --preflight                  # no cluster needed
    python collect/twoarm.py --campaign --load-attested    # the real thing
    python collect/twoarm.py --one MEMORY_LEAK redis ramp --load-attested
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Run directly (`python collect/twoarm.py`), which only puts collect/ on
# sys.path -- same reason collect/chaos.py does this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collect import chaos  # noqa: E402
from eval.recovery import (  # noqa: E402
    CONTROL_ARM,
    HEALTH_COLUMNS,
    PAIRS_COLUMNS,
    TREATMENT_ARM,
)

ARMS = (TREATMENT_ARM, CONTROL_ARM)
# arm name == namespace name. Kept as a mapping anyway so a cluster that names
# them differently changes one line here and nothing in eval/.
ARM_NAMESPACE = {TREATMENT_ARM: "prodrome", CONTROL_ARM: "control"}

OUTDIR = "data/twoarm"

# Finer than the 15s metrics tick on purpose. A POD_KILL recovery is tens of
# seconds; sampling at 15s quantises it so coarsely that a real 10s difference
# between the arms is invisible or doubled. This series is health only -- it is
# NOT the SETUP.md 7 metrics table and does not touch it.
HEALTH_SAMPLE_SECONDS = 5.0

# How long to keep sampling after injection stops. Recovery is the thing being
# measured, so the window has to be long enough that a slow recovery reads as a
# number rather than as censored. 5 minutes covers a CrashLoop backoff cycle.
TRAIL_SECONDS = 300.0
# Baseline for the degradation band comes from this much quiet time before
# injection, on each arm separately (eval.recovery.latency_band).
LEAD_SECONDS = 90.0


@dataclass(frozen=True)
class PairSpec:
    """One fault, to be injected into both arms at once."""
    pair_id: str
    fault_type: str
    workload: str
    pattern: str
    duration_s: int


@dataclass
class ArmResult:
    arm: str
    namespace: str
    start_ts: pd.Timestamp | None = None
    end_ts: pd.Timestamp | None = None
    failure_ts: pd.Timestamp | None = None
    restarts_delta: int = 0
    inject_ok: bool = False
    error: str = ""


# --- the cluster adapters: the only functions here that need a cluster ------

def kubectl_injector(namespace: str, spec: PairSpec) -> dict:
    """Inject one fault into one namespace, reusing `collect/chaos.py` verbatim.

    Reused rather than reimplemented so the two arms take the fault the chaos
    campaign already validated -- same `stress-ng` arguments, same ramp steps,
    same OOM detection. A second injector would be a second fault definition,
    and then "identical faults" would be an assertion instead of a fact.
    """
    if spec.fault_type == "POD_KILL":
        return chaos.run_pod_kill(spec.workload, namespace)
    return chaos.run_fault(spec.fault_type, spec.workload, spec.pattern,
                           spec.duration_s, namespace)


def kubectl_probe(namespace: str, workload: str) -> dict:
    """One health sample for one workload in one namespace.

    Exactly two `kubectl` calls, the same two in the same order for both arms,
    so the measurement itself cannot favour an arm. Readiness and the restart
    counter come from ONE `get pods` rather than a `get deploy` plus a separate
    `restart_count()`: each call costs 600-900ms of binary startup and API
    round-trip, and at three calls per arm-workload the sampler's achieved
    cadence drifted to ~10s, which quantises a 30s POD_KILL recovery badly
    enough to invent or erase a real 10s difference between the arms.

    `probe_latency_ms` is the wall time of the readiness probe and is the only
    latency SLI this project has anywhere. Note what dominates it: measured
    live against this cluster it sits at 650-920ms on healthy pods, almost all
    of which is `kubectl exec` startup, not the workload. That makes it a blunt
    instrument for CPU_HOG and DISK_STRESS -- see `eval/recovery.py`'s
    `degradation_outcome` for what may and may not be concluded from it.
    """
    r = chaos._kubectl(
        "get", "pods", "-n", namespace, "-l", f"app={workload}", "-o",
        "jsonpath={range .items[*]}{.status.containerStatuses[*].ready}"
        "={.status.containerStatuses[*].restartCount} {end}", timeout=15)
    tokens = [t for t in r.stdout.split() if "=" in t]
    ready = sum(1 for t in tokens if t.split("=")[0] == "true")
    restarts = sum(int(t.split("=")[1]) for t in tokens if t.split("=")[1].isdigit())

    t0 = time.monotonic()
    p = chaos._kubectl("exec", "-n", namespace, f"deploy/{workload}", "--",
                       "true", timeout=15)
    latency_ms = (time.monotonic() - t0) * 1000.0
    return dict(ready_replicas=ready, desired_replicas=len(tokens) or 0,
                restarts_total=restarts,
                probe_ok=(p.returncode == 0), probe_latency_ms=round(latency_ms, 1))


# --- simultaneous injection -------------------------------------------------

def inject_both(spec: PairSpec, injector=kubectl_injector,
                clock=None) -> dict[str, ArmResult]:
    """Fire the same fault into both arms at the same instant.

    Both threads build everything they need, meet at a `Barrier`, and only then
    call the injector -- so the skew is the difference between two `kubectl`
    round-trips rather than the difference between two Python startups. The
    achieved skew is measured and returned; `eval/recovery.py` decides whether
    it was good enough. We do not assert simultaneity, we record it.
    """
    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    results: dict[str, ArmResult] = {
        arm: ArmResult(arm=arm, namespace=ARM_NAMESPACE[arm]) for arm in ARMS}
    barrier = threading.Barrier(len(ARMS))

    def one(arm: str) -> None:
        res = results[arm]
        try:
            barrier.wait(timeout=60)
            res.start_ts = clock()
            rec = injector(res.namespace, spec)
            res.end_ts = clock()
            # Trust the injector's own start/end when it reports them -- it
            # knows when stress-ng really began -- but keep our own as the
            # fallback so a fake or a partial return still produces a pair.
            res.start_ts = rec.get("start_ts") or res.start_ts
            res.end_ts = rec.get("end_ts") or res.end_ts
            res.failure_ts = rec.get("failure_ts")
            res.restarts_delta = int(rec.get("restarts_delta") or 0)
            res.inject_ok = True
        except Exception as exc:                      # noqa: BLE001
            # One arm failing must not take the other down: a half-pair is
            # still worth recording, and recording it as a half-pair is how it
            # gets excluded rather than silently counted.
            #
            # This handler deliberately does NOT call barrier.abort(). It used
            # to, to fail the other arm fast, and that was a race: CPython's
            # Barrier re-checks its state as a released thread resumes, so an
            # abort() from this arm -- after its own wait() had already
            # returned -- could still raise BrokenBarrierError in the OTHER
            # arm before it reached its injector. Observed: the control arm's
            # injector raised, and the prodrome arm then never injected at all
            # and reported BrokenBarrierError, turning a one-arm failure into a
            # zero-arm pair non-deterministically.
            #
            # Without the abort, an arm that dies BEFORE the barrier simply
            # lets the other arm's wait(timeout=60) expire, which is the
            # correct outcome -- never inject into one arm only -- and an arm
            # that dies AFTER the barrier leaves the other arm's real
            # measurement intact.
            res.end_ts = res.end_ts or clock()
            res.error = f"{type(exc).__name__}: {exc}"
            res.inject_ok = False

    threads = [threading.Thread(target=one, args=(arm,), name=f"inject-{arm}")
               for arm in ARMS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def skew_seconds(results: dict[str, ArmResult]) -> float | None:
    starts = [r.start_ts for r in results.values() if r.start_ts is not None]
    if len(starts) < len(ARMS):
        return None
    return abs((max(starts) - min(starts)).total_seconds())


# --- health sampling --------------------------------------------------------

@dataclass
class HealthSampler:
    """Polls both arms' health on one thread at one cadence.

    One thread for both arms, not one per arm: a single loop guarantees the two
    arms are sampled within a few hundred milliseconds of each other for the
    whole experiment, so a recovery-time difference cannot come from one arm
    having been watched more closely than the other.
    """
    workloads: list[str]
    probe_fn = staticmethod(kubectl_probe)
    interval: float = HEALTH_SAMPLE_SECONDS
    clock = staticmethod(lambda: pd.Timestamp.now(tz="UTC"))
    rows: list[dict] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _one(self, arm: str, workload: str) -> dict:
        ts = self.clock()
        try:
            h = self.probe_fn(ARM_NAMESPACE[arm], workload)
        except Exception as exc:                      # noqa: BLE001
            # A probe that throws is itself health information: the API server
            # or the pod is not answering. Record it as not-serving rather than
            # dropping the sample, or an outage looks like a gap in the data --
            # and a gap reads as "nothing happened", hiding the exact event
            # being measured.
            h = dict(ready_replicas=0, desired_replicas=0, restarts_total=None,
                     probe_ok=False, probe_latency_ms=None)
        return dict(ts=ts, arm=arm, workload=workload,
                    **{k: h.get(k) for k in HEALTH_COLUMNS[3:]})

    def sample_once(self) -> None:
        """Probe every (arm, workload) CONCURRENTLY, not in a nested loop.

        Two reasons, both about the comparison rather than about speed. The
        arms end up sampled within milliseconds of each other instead of one
        kubectl round-trip apart, so neither arm is systematically observed
        later than the other -- a sequential loop always reads the control arm
        ~800ms stale, which is a constant bias in a 30s measurement. And the
        achieved cadence stops scaling with the number of workloads, so the
        5s interval is roughly honoured instead of degrading to 10s+.
        """
        targets = [(arm, w) for arm in ARMS for w in self.workloads]
        with ThreadPoolExecutor(max_workers=max(len(targets), 1)) as pool:
            self.rows.extend(pool.map(lambda t: self._one(*t), targets))

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                self.sample_once()
                self._stop.wait(self.interval)
        self._thread = threading.Thread(target=loop, name="health-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=HEALTH_COLUMNS)


# --- the schedule -----------------------------------------------------------

def paired_plan(rounds: int = 1, fault_types: list[str] | None = None,
                workloads: list[str] | None = None) -> list[PairSpec]:
    """The campaign as an ordered list of pairs, each with a stable `pair_id`.

    Deterministic so a crashed campaign resumes instead of renumbering, exactly
    like `collect/chaos.py`'s `_plan()`. POD_KILL is included on every workload
    rather than the two `chaos.py` samples: with only four fault types and two
    of them unable to move (see `eval/recovery.py` on the policy table), the
    cheap instant-failure floor is worth having on all three.
    """
    faults = fault_types or [*chaos.FAULTS, "POD_KILL"]
    wls = workloads or chaos.WORKLOADS
    specs: list[PairSpec] = []
    seen: dict[tuple, int] = {}
    for _ in range(rounds):
        for workload in wls:
            for fault in faults:
                patterns = ["none"] if fault == "POD_KILL" else chaos.PATTERNS
                for pattern in patterns:
                    key = (fault, pattern, workload)
                    seen[key] = seen.get(key, 0) + 1
                    specs.append(PairSpec(
                        pair_id=f"{fault}_{pattern}_{workload}_{seen[key]:03d}",
                        fault_type=fault, workload=workload, pattern=pattern,
                        duration_s=chaos.DEFAULT_DURATION))
    return specs


def pair_rows(spec: PairSpec, results: dict[str, ArmResult],
              load_attested: bool) -> list[dict]:
    """Two rows per pair, one per arm -- the shape `eval/recovery.py` expects."""
    skew = skew_seconds(results)
    return [dict(pair_id=spec.pair_id, arm=r.arm, namespace=r.namespace,
                 fault_type=spec.fault_type, workload=spec.workload,
                 pattern=spec.pattern, duration_s=spec.duration_s,
                 start_ts=r.start_ts, end_ts=r.end_ts, failure_ts=r.failure_ts,
                 restarts_delta=r.restarts_delta, inject_ok=r.inject_ok,
                 skew_s=skew, load_attested=load_attested)
            for r in (results[a] for a in ARMS)]


def _append(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Append after every pair, same reason `collect/chaos.py` does: a campaign
    that dies two hours in keeps everything it already measured."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{c: r.get(c) for c in columns} for r in rows]).to_csv(
        path, mode="a", header=not path.exists(), index=False)


def run_pair(spec: PairSpec, load_attested: bool, outdir: str = OUTDIR,
             injector=kubectl_injector, probe_fn=kubectl_probe,
             sleep=time.sleep) -> dict[str, ArmResult]:
    """One pair end to end: lead-in baseline, simultaneous injection, trailing
    recovery window, then flush both files.

    The lead-in is not padding -- `eval.recovery.latency_band()` needs quiet
    pre-injection samples on each arm to set that arm's own degradation band,
    and without it every CPU_HOG/DISK_STRESS row comes back NO_DATA.
    """
    outdir_p = Path(outdir)
    sampler = HealthSampler(workloads=[spec.workload])
    sampler.probe_fn = probe_fn
    sampler.start()
    try:
        sleep(LEAD_SECONDS)                      # per-arm baseline for the band
        results = inject_both(spec, injector=injector)
        sleep(TRAIL_SECONDS)                     # watch for recovery
    finally:
        sampler.stop()

    _append(outdir_p / "pairs.csv", pair_rows(spec, results, load_attested), PAIRS_COLUMNS)
    _append(outdir_p / "health.csv", sampler.rows, HEALTH_COLUMNS)
    skew = skew_seconds(results)
    print(f"  {spec.pair_id:34} skew={skew if skew is None else round(skew, 2)}s  "
          + "  ".join(f"{a}:{'ok' if results[a].inject_ok else results[a].error}"
                      for a in ARMS))
    return results


def campaign(rounds: int = 1, fault_types: list[str] | None = None,
             outdir: str = OUTDIR, gap: int = chaos.DEFAULT_GAP,
             load_attested: bool = False, injector=kubectl_injector,
             probe_fn=kubectl_probe, sleep=time.sleep) -> None:
    done: set[str] = set()
    pairs_p = Path(outdir) / "pairs.csv"
    if pairs_p.exists():
        done = set(pd.read_csv(pairs_p)["pair_id"])
        print(f"resuming -- {len(done)} pairs already in {pairs_p}, skipping those\n")

    plan = paired_plan(rounds, fault_types)
    print(f"{len(plan)} pairs x 2 arms, gap {gap}s "
          f"(= ml.detector.POST_RESTART_SUPPRESS_TICKS + margin)\n")
    for spec in plan:
        if spec.pair_id in done:
            print(f"  skip {spec.pair_id} (done)")
            continue
        run_pair(spec, load_attested, outdir, injector, probe_fn, sleep)
        sleep(gap)


def preflight() -> bool:
    """Everything checkable without a cluster, printed before anyone spends
    hours. Prints the control-arm attestations and whether a cluster answers.

    Separate entrypoint because the expensive failure mode for this harness is
    not a crash -- it is a campaign that completes and produces two arms that
    were never different, which looks exactly like a negative result.
    """
    from eval import recovery
    from control import controller

    atts = recovery.attest(controller_namespace=controller.NAMESPACE,
                           dry_run=controller.DRY_RUN)
    for a in atts:
        print(f"  [{'ok ' if a.ok else 'FAIL'}] {a.name}: {a.detail}")

    print(f"  [--  ] kubectl-transport: {' '.join(chaos.KUBECTL_CMD)}"
          "   (override with PRODROME_KUBECTL)")

    cluster_ok = True
    try:
        ctx = chaos._kubectl("config", "current-context", timeout=20)
        up = ctx.returncode == 0
        cluster_ok &= up
        print(f"  [{'ok ' if up else 'FAIL'}] cluster-reachable: "
              f"{ctx.stdout.strip() or ctx.stderr.strip().splitlines()[:1]}")
    except (OSError, subprocess.SubprocessError) as exc:
        cluster_ok = False
        print(f"  [FAIL] cluster-reachable: {type(exc).__name__}: {exc}")

    # Per arm, not aggregated: an arm that is half-deployed is the failure mode
    # that produces a recovery-time difference caused entirely by one arm's pod
    # never having been there.
    for arm in ARMS:
        ns = ARM_NAMESPACE[arm]
        try:
            r = chaos._kubectl("get", "deploy", "-n", ns, "-o",
                               "jsonpath={range .items[*]}{.metadata.name}="
                               "{.status.readyReplicas}/{.spec.replicas} {end}", timeout=20)
            got = r.stdout.strip()
            ready = (r.returncode == 0
                     and all(f"{w}=1/1" in got for w in chaos.WORKLOADS))
        except (OSError, subprocess.SubprocessError) as exc:
            got, ready = f"{type(exc).__name__}: {exc}", False
        cluster_ok &= ready
        print(f"  [{'ok ' if ready else 'FAIL'}] arm-{arm}-ready (ns={ns}): "
              f"{got or 'no deployments'}")

    ok = recovery.publishable(atts) and cluster_ok
    print("\nready to run a publishable two-arm campaign: " + ("yes" if ok else "NO"))
    if not ok:
        print("a campaign can still be run -- it just must not be reported as a\n"
              "Prodrome-vs-stock comparison until every line above is ok. A run made\n"
              "while treatment-arm-live FAILs is the NULL experiment: it validates the\n"
              "harness end to end and must come back with no difference between arms.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true",
                    help="check the preconditions; needs no cluster for most of them")
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--one", nargs="*", metavar="ARG",
                    help="single pair: FAULT WORKLOAD [PATTERN]")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--fault-types", default=None)
    ap.add_argument("--gap", type=int, default=chaos.DEFAULT_GAP)
    ap.add_argument("--duration", type=int, default=chaos.DEFAULT_DURATION)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--load-attested", action="store_true",
                    help="you have confirmed identical load is running against BOTH "
                         "namespaces (collect/load/run-load.sh). Recorded per arm-row.")
    ap.add_argument("--null-run", action="store_true",
                    help="run with load_attested=False recorded honestly, to validate the "
                         "harness against a real cluster. The result is NOT a comparison "
                         "and eval/recovery.py will refuse to publish it as one.")
    args = ap.parse_args()

    if args.preflight:
        preflight()
        return
    if not args.load_attested and not args.null_run:
        # Two ways past this, and neither of them is "assume load was running".
        # --load-attested is a claim the operator makes and the harness records;
        # --null-run is the admission that there was none, recorded just as
        # plainly, which makes attest_load_parity() fail and the run
        # unpublishable. What is forbidden is a campaign with no statement
        # either way, because that is the one a later reader cannot assess.
        print("refusing to run without --load-attested: both arms need identical load or "
              "the comparison is meaningless (docs/guides/shaurya.md Part 4). Start "
              "collect/load/run-load.sh against BOTH namespaces, then pass the flag -- "
              "or pass --null-run to validate the harness and record that there was no "
              "load, which makes the run unpublishable as a comparison.")
        raise SystemExit(2)
    if args.null_run and args.load_attested:
        ap.error("--null-run and --load-attested contradict each other")

    if args.one:
        fault, workload = args.one[0], args.one[1]
        pattern = args.one[2] if len(args.one) > 2 else ("none" if fault == "POD_KILL"
                                                         else "constant")
        spec = PairSpec(pair_id=f"{fault}_{pattern}_{workload}_001", fault_type=fault,
                        workload=workload, pattern=pattern, duration_s=args.duration)
        run_pair(spec, args.load_attested, args.outdir)
    elif args.campaign:
        campaign(args.rounds,
                 args.fault_types.split(",") if args.fault_types else None,
                 args.outdir, args.gap, args.load_attested)
        print(f"\nnow measure it:  python -m eval.recovery --indir {args.outdir}")
    else:
        ap.error("pass --preflight, --campaign or --one FAULT WORKLOAD [PATTERN]")


if __name__ == "__main__":
    main()
