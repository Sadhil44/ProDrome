"""LIVE positive control: can the COLLECTION half see an effect that is there?

-------------------------------------------------------------------------------
THE HALF `eval/sensitivity.py` CANNOT COVER
-------------------------------------------------------------------------------

`eval/sensitivity.py` holds one arm back by rewriting health bits in a finished
campaign's `health.csv`. That is a complete test of the analysis -- outcome
classification, the sustained-hold, the per-arm latency band, the tie band, the
paired delta -- and no test at all of the thing that produced `health.csv`: the
two-arm sampler, the `kubectl` round-trip, the injection barrier, the cadence
the cluster actually gives us.

This module closes that. It puts a difference of known size into the CLUSTER
and lets the ordinary collection path discover it, so a pair that comes out of
here has been through every component a real campaign uses.

-------------------------------------------------------------------------------
HOW A KNOWN DELTA IS MADE WITHOUT TOUCHING control/ OR FAKING DATA
-------------------------------------------------------------------------------

`DRY_RUN` stays True and `control/controller.py` is not read, written or
imported by this file. The difference does not come from Prodrome. It comes
from a deliberate, symmetric handicap:

    both arms, at the same instant (threading.Barrier):  replicas -> 0
    prodrome, after  `base_hold_s`          seconds:     replicas -> 1
    control,  after  `base_hold_s + effect_s` seconds:   replicas -> 1

Both arms suffer the same KIND of outage and the same structural overheads --
pod termination, scheduling, image start, readiness gate -- so those cancel in
the per-pair difference. What does not cancel is `effect_s`, which is the only
asymmetry in the design. The true delta is `effect_s` BY CONSTRUCTION, and it
is the thing the harness has to come back with.

Note the direction: the CONTROL arm is the one held down longer, so a working
harness reports the TREATMENT arm faster -- the sign a real Prodrome win would
have. `--hurt-arm prodrome` flips it, which is worth one run: a harness that
derived the sign from which arm it polled first rather than from the data would
pass the control-arm version and fail this one.

`effect_s = 0` is a pair worth running too. It is a live null: the same outage
on both arms with no asymmetry at all, so anything other than a tie is the
harness inventing a difference.

-------------------------------------------------------------------------------
WHAT THIS IS NOT
-------------------------------------------------------------------------------

Stated in code so it survives summarising: the effect here is MANUFACTURED by
this file. It is evidence about the instrument, never about Prodrome. The rows
are written with `fault_type=POSITIVE_CONTROL`, which `control/policy.py` has
no entry for, so `eval.recovery.lever_for` resolves it to "nothing" and every
table marks it `comparable=False` -- a positive-control pair cannot be read as
a Prodrome win even by a reader who only sees the table. They are also written
to a SEPARATE directory (`data/twoarm-poscontrol/` by default) so they can
never be concatenated into the real campaign by accident.

A scale-to-zero is also not a fault. It is an outage of a chosen length, which
is exactly what a sensitivity test needs and exactly what a chaos experiment
does not: nothing here tells you how the system behaves under stress, only
whether the instrument can measure a difference between two outages.

-------------------------------------------------------------------------------
SAFETY
-------------------------------------------------------------------------------

Every arm is scaled back up in a `finally`, and `restore()` runs at the end of
the campaign and after any exception, so a killed run leaves both namespaces at
one replica rather than at zero. `--restore-only` is the manual version for the
case where the process was killed outright.

Usage:
    python -m collect.poscontrol --preflight
    python -m collect.poscontrol --campaign --sizes 0,15,45 --repeats 3
    python -m collect.poscontrol --restore-only
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import pandas as pd

from collect import chaos
from collect.twoarm import (ARM_NAMESPACE, ARMS, CONTROL_ARM, HEALTH_COLUMNS,
                            PAIRS_COLUMNS, TREATMENT_ARM, ArmResult,
                            HealthSampler, _append, kubectl_probe,
                            skew_seconds, wait_settled)

OUTDIR = "data/twoarm-poscontrol"

# The sentinel fault type. See the module docstring: it has no row in
# control/policy.py on purpose, so every report marks it not-comparable.
FAULT_TYPE = "POSITIVE_CONTROL"

# Both arms are held down at least this long before either is released. It is
# not padding: the per-pair difference only cancels the structural overheads if
# both arms actually incur them, and a zero base hold would race the release
# against the scale-down the API server has not finished applying.
BASE_HOLD_SECONDS = 20.0

# Pre-injection quiet samples. Shorter than the campaign's 90s because
# POSITIVE_CONTROL is an OUTAGE fault and does not need a latency band -- it
# needs enough samples to show the arm was serving before the outage.
LEAD_SECONDS = 45.0

# Trailing observation after the LAST arm is released. The measurement needs
# `HOLD_SECONDS` (30s) of sustained serving plus pod start-up, so this has
# margin over both. It is recorded per pair in `trail_s`, which is the
# censoring horizon `eval/recovery.py` reads.
TRAIL_SECONDS = 120.0

WORKLOAD = "nginx"          # fastest, most deterministic restart of the three


# --- the cluster adapter ----------------------------------------------------

def kubectl_scale(namespace: str, workload: str, replicas: int) -> bool:
    r = chaos._kubectl("scale", f"deploy/{workload}", "-n", namespace,
                       f"--replicas={replicas}", timeout=30)
    return r.returncode == 0


def hold_both(workload: str, base_hold_s: float, effect_s: float,
              hurt_arm: str = CONTROL_ARM, scale_fn=kubectl_scale,
              sleep=time.sleep, clock=None) -> dict[str, ArmResult]:
    """Take both arms down together; bring one back `effect_s` later.

    The barrier is the same device `collect/twoarm.inject_both` uses and for
    the same reason: both threads do all their setup first and meet, so the
    skew between the two scale-downs is two `kubectl` round-trips rather than
    two Python startups. The skew is measured and written down, not asserted.

    `end_ts` is the instant this module asked for the arm back, which is the
    last moment it controlled. The recovery time the harness reports is
    measured from the health series, not from here -- these timestamps exist so
    a reader can check what was requested against what was measured.
    """
    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    results = {arm: ArmResult(arm=arm, namespace=ARM_NAMESPACE[arm])
               for arm in ARMS}
    barrier = threading.Barrier(len(ARMS))
    holds = {arm: base_hold_s + (effect_s if arm == hurt_arm else 0.0)
             for arm in ARMS}

    def one(arm: str) -> None:
        res, ns = results[arm], ARM_NAMESPACE[arm]
        try:
            barrier.wait(timeout=60)
            res.start_ts = clock()
            if not scale_fn(ns, workload, 0):
                raise RuntimeError(f"scale-to-0 failed for {ns}/{workload}")
            # The outage instant we CAUSED. The harness must find its own in
            # the health series; this is only the note beside it.
            res.failure_ts = res.start_ts
            sleep(holds[arm])
            res.end_ts = clock()
            if not scale_fn(ns, workload, 1):
                raise RuntimeError(f"scale-to-1 failed for {ns}/{workload}")
            res.inject_ok = True
        except Exception as exc:                      # noqa: BLE001
            res.error = f"{type(exc).__name__}: {exc}"
            res.inject_ok = False
        finally:
            # Belt and braces. If anything above threw after the scale-down,
            # this arm must not be left at zero replicas.
            res.end_ts = res.end_ts or clock()
            if not res.inject_ok:
                try:
                    scale_fn(ns, workload, 1)
                except Exception:                     # noqa: BLE001, S110
                    pass

    threads = [threading.Thread(target=one, args=(a,), name=f"hold-{a}")
               for a in ARMS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def restore(workloads=None, scale_fn=kubectl_scale) -> dict[str, bool]:
    """Put every arm back to one replica. Idempotent, and run unconditionally
    at the end of a campaign and after any exception."""
    out = {}
    for arm in ARMS:
        for w in (workloads or [WORKLOAD]):
            out[f"{ARM_NAMESPACE[arm]}/{w}"] = scale_fn(ARM_NAMESPACE[arm], w, 1)
    return out


# --- one pair ---------------------------------------------------------------

def pair_id_for(effect_s: float, n: int, hurt_arm: str) -> str:
    hurt = "c" if hurt_arm == CONTROL_ARM else "p";  # which arm was held back
    return f"POSCTL_{hurt}{int(round(effect_s)):03d}s_{n:03d}"


def poscontrol_rows(pair_id: str, workload: str, effect_s: float,
                    base_hold_s: float, hurt_arm: str,
                    results: dict[str, ArmResult], settle: dict | None,
                    trail_s: float) -> list[dict]:
    """Two rows, same columns `eval/recovery.py` already reads.

    `duration_s` carries that ARM's own hold, so the file records the asymmetry
    rather than leaving it in this module's arguments. `pattern` carries the
    manufactured effect size, which is how the analysis groups the curve: it is
    the one piece of information a positive-control pair has that a real pair
    does not, and inventing a new column for it would make these rows
    unreadable by the existing loader.
    """
    skew = skew_seconds(results)
    settle = settle or {}
    rows = []
    for arm in ARMS:
        r = results[arm]
        hold = base_hold_s + (effect_s if arm == hurt_arm else 0.0)
        rows.append(dict(
            pair_id=pair_id, arm=arm, namespace=r.namespace,
            fault_type=FAULT_TYPE, workload=workload,
            pattern=f"effect{int(round(effect_s))}s_hurt_{hurt_arm}",
            duration_s=int(round(hold)),
            start_ts=r.start_ts, end_ts=r.end_ts, failure_ts=r.failure_ts,
            restarts_delta=0, inject_ok=r.inject_ok, skew_s=skew,
            # Never attested and never could be: no load generator is running
            # and this is not a Prodrome comparison in the first place.
            load_attested=False,
            settled_ok=settle.get("settled"), settled_s=settle.get("waited_s"),
            trail_s=trail_s))
    return rows


def run_pair(pair_id: str, effect_s: float, outdir: str = OUTDIR,
             workload: str = WORKLOAD, base_hold_s: float = BASE_HOLD_SECONDS,
             hurt_arm: str = CONTROL_ARM, lead_s: float = LEAD_SECONDS,
             trail_s: float | None = None, probe_fn=kubectl_probe,
             scale_fn=kubectl_scale, sleep=time.sleep,
             settle: dict | None = None) -> dict[str, ArmResult]:
    """Lead-in, synchronised take-down, asymmetric release, trailing watch.

    `trail_s` defaults to `TRAIL_SECONDS + effect_s`: the observation window has
    to outlast the effect it is observing, and a fixed trail would censor the
    large sizes and only the large sizes -- which looks exactly like the
    harness going blind at large effects.
    """
    trail_s = TRAIL_SECONDS + effect_s if trail_s is None else trail_s
    sampler = HealthSampler(workloads=[workload])
    sampler.probe_fn = probe_fn
    sampler.start()
    try:
        sleep(lead_s)
        results = hold_both(workload, base_hold_s, effect_s, hurt_arm,
                            scale_fn=scale_fn, sleep=sleep)
        sleep(trail_s)
    finally:
        sampler.stop()
        restore([workload], scale_fn=scale_fn)

    out = Path(outdir)
    _append(out / "pairs.csv",
            poscontrol_rows(pair_id, workload, effect_s, base_hold_s, hurt_arm,
                            results, settle, trail_s), PAIRS_COLUMNS)
    _append(out / "health.csv", sampler.rows, HEALTH_COLUMNS)
    skew = skew_seconds(results)
    print(f"  {pair_id:22} effect={effect_s:5.0f}s  "
          f"skew={skew if skew is None else round(skew, 2)}s  "
          + "  ".join(f"{a}:{'ok' if results[a].inject_ok else results[a].error}"
                      for a in ARMS), flush=True)
    return results


def campaign(sizes=(0.0, 15.0, 45.0), repeats: int = 3, outdir: str = OUTDIR,
             workload: str = WORKLOAD, base_hold_s: float = BASE_HOLD_SECONDS,
             hurt_arm: str = CONTROL_ARM, lead_s: float = LEAD_SECONDS,
             gap_s: float = 20.0, probe_fn=kubectl_probe,
             scale_fn=kubectl_scale, sleep=time.sleep, settle_fn=None) -> None:
    """Interleaved, not blocked: all sizes once, then again.

    Blocking by size would confound effect size with time -- every large-effect
    pair would have run after every small one, so cluster drift over the run
    would be indistinguishable from the thing being measured. Interleaving also
    means a campaign killed halfway still has every size, which for a run on a
    memory-starved host is the case to plan for.
    """
    done: set[str] = set()
    pairs_p = Path(outdir) / "pairs.csv"
    if pairs_p.exists():
        done = set(pd.read_csv(pairs_p)["pair_id"])
        print(f"resuming -- {len(done)} pairs already in {pairs_p}\n")

    plan = [(s, n) for n in range(1, repeats + 1) for s in sizes]
    todo = [(s, n) for s, n in plan
            if pair_id_for(s, n, hurt_arm) not in done]
    est = sum(lead_s + base_hold_s + s + TRAIL_SECONDS + s + gap_s
              for s, _ in todo) / 60.0
    print(f"{len(plan)} pairs x 2 arms ({len(todo)} to run), sizes={list(sizes)}, "
          f"hurt arm={hurt_arm}, base hold={base_hold_s:.0f}s, "
          f"lead={lead_s:.0f}s, gap={gap_s:.0f}s\n"
          f"estimated wall clock: {est:.0f} min\n", flush=True)
    try:
        for i, (s, n) in enumerate(todo, 1):
            settle = settle_fn([workload]) if settle_fn else None
            if settle is not None and not settle["settled"]:
                print(f"  [warn] settle gate: {settle['detail']}", flush=True)
            pid = pair_id_for(s, n, hurt_arm)
            print(f"  [{i}/{len(todo)}] {pid}", flush=True)
            run_pair(pid, s, outdir, workload, base_hold_s, hurt_arm, lead_s,
                     probe_fn=probe_fn, scale_fn=scale_fn, sleep=sleep,
                     settle=settle)
            sleep(gap_s)
    finally:
        print(f"restore: {restore([workload], scale_fn=scale_fn)}", flush=True)


def preflight(scale_fn=kubectl_scale) -> bool:
    """Cheap checks before spending wall clock, plus the standing refusal.

    Deliberately prints the not-a-Prodrome-result line every time. The expensive
    failure mode of this file is not a crash, it is a table of manufactured
    wins being quoted a month from now by someone who did not read the module.
    """
    from control import controller
    from eval import recovery

    print(__doc__.split("Usage:")[0].strip()[:0] or "", end="")
    ok = True
    print(f"  [--  ] DRY_RUN is {controller.DRY_RUN} and this file never "
          "reads or writes control/ -- the effect below is manufactured here")
    comparable = recovery.lever_for(FAULT_TYPE) not in ("nothing", "alert_only")
    print(f"  [{'ok ' if not comparable else 'FAIL'}] not-comparable: "
          f"{FAULT_TYPE} -> lever {recovery.lever_for(FAULT_TYPE)!r} "
          f"-> comparable={comparable} (must be False)")
    ok &= not comparable
    kind = recovery.RECOVERY_KIND.get(FAULT_TYPE)
    print(f"  [{'ok ' if kind == 'outage' else 'FAIL'}] declared kind: {kind}")
    ok &= kind == "outage"

    for arm in ARMS:
        ns = ARM_NAMESPACE[arm]
        r = chaos._kubectl("get", "deploy", WORKLOAD, "-n", ns, "-o",
                           "jsonpath={.status.readyReplicas}/{.spec.replicas}",
                           timeout=20)
        got = r.stdout.strip()
        up = r.returncode == 0 and got == "1/1"
        ok &= up
        print(f"  [{'ok ' if up else 'FAIL'}] arm-{arm} {ns}/{WORKLOAD}: "
              f"{got or 'absent'}")
    print("\nready: " + ("yes" if ok else "NO"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--restore-only", action="store_true",
                    help="scale both arms back to 1 replica and exit")
    ap.add_argument("--sizes", default="0,15,45",
                    help="manufactured effect sizes in seconds")
    ap.add_argument("--repeats", type=int, default=3, help="pairs per size")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--workload", default=WORKLOAD)
    ap.add_argument("--base-hold", type=float, default=BASE_HOLD_SECONDS)
    ap.add_argument("--lead", type=float, default=LEAD_SECONDS)
    ap.add_argument("--gap", type=float, default=20.0)
    ap.add_argument("--hurt-arm", choices=list(ARMS), default=CONTROL_ARM)
    ap.add_argument("--settle-gate", action="store_true")
    args = ap.parse_args()

    if args.restore_only:
        print(restore([args.workload]))
        return
    if args.preflight or not args.campaign:
        preflight()
        if not args.campaign:
            return
    campaign(sizes=tuple(float(s) for s in args.sizes.split(",")),
             repeats=args.repeats, outdir=args.outdir, workload=args.workload,
             base_hold_s=args.base_hold, hurt_arm=args.hurt_arm,
             lead_s=args.lead, gap_s=args.gap,
             settle_fn=wait_settled if args.settle_gate else None)


if __name__ == "__main__":
    main()
