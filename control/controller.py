# Prodrome controller
import csv
import os
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from control.policy import decide
from ml.features import METRICS, WINDOW_SIZE

# NOTE: `kubernetes` is deliberately NOT imported here. At module scope it made
# this whole module unimportable without the package, so the action wiring, the
# cooldown, the kill switch and the decision-log format could not be tested
# offline or in CI and had to be checked by parsing this file's source. Only
# connect_to_kubernetes() needs it, and it imports it there.


# This file's own location, which is the one thing about the process that does
# not depend on where an operator happened to be standing when they started it.
REPO_ROOT = Path(__file__).resolve().parent.parent

NAMESPACE = "prodrome"
# SETUP.md S8 names this path, and the dashboard and eval harness both expect it.
#
# Deliberately still RELATIVE, which is a known and currently unfixable gap:
# tests/test_integration.py asserts `controller.LOG_FILE == Path(
# "data/decisions/log.csv")` to pin that the writer and the dashboard name the
# same file, and tests/ is Aahan's. So the decision log still resolves against
# the working directory: start the controller from elsewhere and it writes an
# audit trail the dashboard and the eval harness never find. Same class of
# defect as the STOP bug below, same severity-one-step-lower consequence (a lost
# audit trail rather than a dead kill switch). `startup_report()` prints the
# RESOLVED path so at least it is visible. Raised for testing rather than fixed
# unilaterally.
LOG_FILE = Path("data/decisions/log.csv")
DRY_RUN = True
MAX_REPLICAS = 5
COOLDOWN_SECONDS = 120

# Global cap on actions across ALL workloads, not per workload.
#
# The per-workload cooldown is individually satisfied by six workloads firing at
# once and producing six simultaneous remediations -- which is the shape of a
# node-level or cluster-level problem, and exactly the situation where
# remediating everything at once is itself the destabilising event. Three
# actions per window means at most half the six-workload fleet can be under
# remediation simultaneously, and the fourth refusal is the signal that the
# problem is not per workload. The number is a judgement call, not a measurement;
# it is here as a named constant so it can be argued with.
#
# RATE_LIMIT_WINDOW_SECONDS == COOLDOWN_SECONDS is LOAD-BEARING, not cosmetic:
# because a workload cannot act twice inside its own cooldown, one lease file per
# workload is an exact count of actions in the window. Make the window longer
# than the cooldown and the count silently starts undercounting.
MAX_ACTIONS_PER_WINDOW = 3
RATE_LIMIT_WINDOW_SECONDS = COOLDOWN_SECONDS

# The kill switch, anchored to the repo root rather than the working directory.
#
# This was `Path("STOP")`, resolved against whatever directory the controller was
# started from. An operator who creates STOP in the repo root while the
# controller runs from somewhere else gets a kill switch that silently does
# nothing -- the worst possible failure for the one control whose entire job is
# to be found in a hurry, by someone in a hurry. It is now absolute.
STOP_FILE = REPO_ROOT / "STOP"

# ...and, additionally, cwd-relative STOP is still honoured. A kill switch should
# fail in the direction of stopping, so the rule is "ANY of these files stops the
# controller", not "exactly this one". The cost is that a stray file named STOP
# in the working directory halts remediation; that is the safe direction, and
# `startup_report()` prints every path that is checked so it is never a mystery.
# A list so tests can redirect it; without that they would read the repo's real
# working directory.
EXTRA_STOP_PATHS = [Path("STOP")]

# Per-workload cooldown state for THIS process. Mirrored to the lease files
# below, which are what make the rail hold between processes; see
# `last_action()` for how the two are combined.
last_action_time = {}

def connect_to_kubernetes():
    """Import the kubernetes client lazily, here, rather than at module scope.

    At module scope it made this entire module unimportable without the
    kubernetes package installed -- so the policy/action wiring, the cooldown,
    the kill switch and the decision-log format could not be tested offline or
    in CI at all, and had to be checked by parsing this file's source. None of
    that logic needs a cluster; only this function does.
    """
    from kubernetes import client, config

    config.load_kube_config()
    return client.AppsV1Api()


def get_replicas(apps_api, deployment_name):
    deployment = apps_api.read_namespaced_deployment(
        name=deployment_name,
        namespace=NAMESPACE,
    )

    replicas = deployment.spec.replicas
    # `spec.replicas` is optional in the Kubernetes API and an unset value means
    # one replica. Returned raw, `min(None + 1, MAX_REPLICAS)` in execute_action
    # raises TypeError part-way through a remediation -- the controller would die
    # on a perfectly legal deployment. This is the documented default, not a guess.
    return 1 if replicas is None else replicas


def scale(apps_api, deployment_name, replicas):
    if replicas > MAX_REPLICAS:
        raise ValueError(
            f"Replica count {replicas} exceeds maximum of {MAX_REPLICAS}"
        )

    body = {
        "spec": {
            "replicas": replicas
        }
    }

    apps_api.patch_namespaced_deployment_scale(
        name=deployment_name,
        namespace=NAMESPACE,
        body=body,
    )

def restart(apps_api, deployment_name):
    timestamp = datetime.now(timezone.utc).isoformat()

    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "prodrome/restarted-at": timestamp
                    }
                }
            }
        }
    }

    apps_api.patch_namespaced_deployment(
        name=deployment_name,
        namespace=NAMESPACE,
        body=body,
    )


# SETUP.md S7, verbatim and in this order. The dashboard and the evaluation
# harness both read this file by column name, so the order and the spelling are
# a contract, not a local choice -- changing either means announcing it.
DECISION_LOG_COLUMNS = [
    "ts",
    "workload",
    "detector_score",
    "fired",
    "predicted_class",
    "confidence",
    "top_features",
    "action",
    "result",
    "mode",
]


def log_decision(
    workload,
    detector_score,
    fired,
    predicted_class,
    confidence,
    action,
    result,
    top_features=None,
    mode=None,
):
    """Append one row per controller evaluation, matching SETUP.md S7.

    Previously wrote `timestamp` instead of `ts` and omitted `top_features` and
    `mode` entirely. `mode` is the column that separates a shadow observation
    from an executed action, so without it ground rule 4's control-arm
    comparison cannot be answered from the log at all -- you cannot tell what
    Prodrome did from what it merely would have done.

    `mode` defaults to the live DRY_RUN setting rather than to a literal, so a
    row can never claim to be live while the process is dry-running.
    """
    if mode is None:
        mode = "shadow" if DRY_RUN else "live"

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()

    with LOG_FILE.open("a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(DECISION_LOG_COLUMNS)

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            workload,
            detector_score,
            fired,
            predicted_class,
            confidence,
            "" if top_features is None else top_features,
            action,
            result,
            mode,
        ])

def execute_action(apps_api, workload, action):
    """Carry out one policy action, or refuse to.

    Returns one of: blocked-kill-switch, dry-run, blocked-cooldown,
    blocked-rate-limit, executed, at-max, no-op, unknown-action. The decision
    log's `result` column is this string, so the set is part of the decision-log
    contract.

    EVERY rail that can refuse lives HERE, at the point of action, and not only
    in evaluate_once -- because evaluate_once is not the only caller: run_loop,
    the eval harness and any head-to-head script can call this directly, and a
    rail enforced one level up is bypassed by every one of them. That argument
    was already accepted for the kill switch; it applies verbatim to the cooldown
    and the global rate limit, which until now a direct caller could ignore
    entirely while every row it wrote read `executed`.

    Order: kill switch, dry-run, cooldown, rate limit. The kill switch is first
    so STOP wins over everything, including the print. Dry-run stays AHEAD of the
    two budget rails, and that ordering is load-bearing for collect: the shadow
    arm must neither spend nor observe a budget, or its rows stop describing what
    Prodrome would have done on its own. The consequence, unchanged, is that
    shadow action counts are an UPPER BOUND on live ones rather than
    like-for-like.

    The cooldown is recorded here too, where the write happens, rather than by
    the caller -- including on an exception, whose outcome is unknown and which
    therefore has to count as an action.

    `action` is matched against exactly the vocabulary control/policy.py emits.
    The previous `("restart", "rolling_restart")` alias accepted a value no
    policy row can produce -- an untested code path whose effect is restarting
    production pods -- so it is gone. Anything unrecognised returns
    unknown-action and touches nothing: that string in the log is how cluster
    finds out diagnosis shipped a lever nobody implemented, which is why it must
    not be flattened into `no-op`.
    """
    if kill_switch_active():
        return "blocked-kill-switch"

    if DRY_RUN:
        print(f"[DRY RUN] Would {action} {workload}")
        return "dry-run"

    if cooldown_active(workload):
        return "blocked-cooldown"

    if rate_limit_active():
        # A refusal nobody can see is not a rail. The caller logs the result, so
        # this reaches the decision log as `blocked-rate-limit`; the print is for
        # whoever is watching the process, because "six workloads are failing and
        # Prodrome stopped after three" is not something to discover later.
        print(
            f"[prodrome][RATE LIMIT] refusing {action} on {workload}: "
            f"{MAX_ACTIONS_PER_WINDOW} actions already taken in the last "
            f"{RATE_LIMIT_WINDOW_SECONDS}s across {sorted(recent_actions())}"
        )
        return "blocked-rate-limit"

    try:
        if action == "rolling_restart":
            restart(apps_api, workload)
            result = "executed"
        elif action == "scale_out":
            current = get_replicas(apps_api, workload)
            # At or above the cap there is nothing a scale_out can legally do,
            # and it must not patch anyway. `min(current + 1, MAX_REPLICAS)` on
            # a deployment an operator or HPA has already taken above the cap
            # resolves DOWNWARDS -- 8 replicas becomes 5 -- so a decision named
            # scale_out removed capacity from a workload mid-incident on a
            # model's guess. Returning at-max also keeps a no-op patch from
            # logging as `executed`, which an operator cannot tell apart from a
            # real scale-up.
            if current >= MAX_REPLICAS:
                result = "at-max"
            else:
                scale(apps_api, workload, min(current + 1, MAX_REPLICAS))
                result = "executed"
        elif action in ("alert_only", "nothing"):
            result = "no-op"
        else:
            result = "unknown-action"
    except Exception:
        # A patch that raised may still have landed -- a 409, a 403, an API
        # server restart. Its outcome is unknown, so it counts against the
        # cooldown and the fleet budget before the exception goes anywhere.
        record_action(workload)
        raise

    # Recorded iff the cluster may have changed. Never on at-max, no-op or
    # unknown-action, which issue no write at all, and never on a block -- a
    # blocked action that recorded a timestamp would push the window forward on
    # every firing and, at this detector's false-positive rate, lock the workload
    # out permanently.
    if result == "executed":
        record_action(workload)

    return result

# --- the cooldown, and why it is a file ---------------------------------------
#
# `last_action_time` was a plain module-level dict, so the cooldown held inside
# one process and nowhere else. Two controllers -- or one controller plus the
# eval harness or a head-to-head script calling `execute_action` directly --
# each believed independently that it was within budget. At the detector's
# measured 5.0% false-positive rate with a ~300s post-fault firing tail, two
# processes could stack remediations on the same workload with neither aware of
# the other, and every row in both logs would read `executed` and look correct.
#
# Options considered:
#
# 1. Leave it process-local and document it. Rejected: the harness ALREADY calls
#    execute_action directly, so this is a live hole, not a theoretical one.
# 2. A lock service, or a Kubernetes Lease object. Rejected: SETUP.md S8 is
#    "files, not services" for the whole project, and a rail that needs the
#    cluster up cannot protect the cluster when it is not. A Lease also could
#    not be read by a controller that is not talking to that cluster yet.
# 3. One shared JSON/CSV file of all cooldowns plus a lock around it. Rejected:
#    every read is a read-modify-write of one file shared by every workload, so
#    it needs real locking, and a crash mid-write corrupts the state for all six
#    workloads at once instead of one.
# 4. Derive it from the decision log: scan back for the last `executed`/`failed`
#    row per workload. Attractive -- zero new state, and the log is already the
#    record of what happened. Rejected: it makes a safety rail depend on parsing
#    an ever-growing CSV on every tick, and on nobody ever rotating, truncating
#    or filtering that file. A rail should not be one `head -n 1000` away from
#    being disabled.
# 5. CHOSEN: one small lease file per workload, written atomically. Independent
#    files, so a corrupt one affects one workload; no lock needed for a
#    whole-file replace; readable with `cat` by an operator who wants to know
#    why Prodrome is not acting; and `os.replace` is atomic on POSIX and on
#    Windows, which is where this repo is actually edited.
#
# WHAT THIS DOES NOT COVER, precisely:
#
# - Two processes on DIFFERENT MACHINES, or in different containers without a
#   shared volume. They do not share a filesystem, so they do not share a
#   cooldown. Nothing here detects that; it fails silently and open.
# - A network filesystem where `os.replace` is not atomic, or where mtimes lie
#   (NFS, some SMB mounts). Same failure, same silence.
# - The check-then-act race. This is shared STATE, not a lock: two processes can
#   both read "no cooldown" and both act. The window is the gap between the
#   check and the write -- one API call wide, not one tick wide -- so it removes
#   the realistic failure (two controllers on 15s cycles stacking inside a 120s
#   window) but not the simultaneous one. Closing it properly needs an atomic
#   claim-before-act, which is a bigger change to the cooldown's recorded
#   semantics (a claim that is then refused has to be released) and is not worth
#   it until something is actually running two controllers.
# - Anything that writes a lease without acting, or acts without writing one.
#   The lease records an ATTEMPT, not a confirmed cluster change.
# - Shadow mode does not read leases at all, on purpose -- see `last_action()`.


def cooldown_dir():
    """Where the per-workload lease files live: beside the decision log.

    Derived from LOG_FILE rather than named independently, and that is
    deliberate. The decision log is the record of what the controller did; these
    leases are the index of it that the rails read. Anything that redirects one
    -- a test, an isolated harness run -- has to redirect the other, or the two
    disagree about what happened. Deriving it makes that automatic instead of
    something every caller has to remember.

    Consequence to be aware of: a process pointed at a different LOG_FILE gets a
    different cooldown namespace and shares no cooldown with the production
    controller. That is right for an isolated experiment and wrong for a second
    live controller, so a second live controller must share the log path.
    """
    return LOG_FILE.parent / "cooldowns"


def _lease_name(workload):
    """A filename for `workload` that is safe and collision-free.

    Percent-encoded rather than sanitised: sanitising maps two different
    workloads onto one file, and two workloads sharing a cooldown is a rail
    quietly doing the wrong thing. Normal Kubernetes names come through
    unchanged, so `data/decisions/cooldowns/nginx.lease` stays greppable.
    """
    return quote(str(workload), safe="") + ".lease"


def _read_lease(path):
    """The timestamp in one lease file, or None if there is no usable one.

    An unreadable or unparseable lease falls back to the file's own mtime rather
    than being ignored. Ignoring it fails OPEN -- a corrupt lease would mean no
    cooldown at all. Falling back to mtime fails closed but BOUNDED: the
    workload is held for one cooldown window from when the file was last
    touched, not forever, so a corrupt file cannot become a permanent lockout
    that looks exactly like a working rail.
    """
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return None
    except OSError:
        text = ""

    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return None

    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _write_lease(workload, when):
    """Publish `workload`'s action time for other processes, atomically.

    Written to a pid-unique temporary and `os.replace`d into position, so a
    concurrent reader sees either the old timestamp or the new one and never a
    half-written file.

    A filesystem that refuses the write is NOT fatal: the action it describes has
    already happened, and raising here would make `evaluate_once` log a
    successful remediation as `failed`. It is also not silent -- the cross-process
    guarantee is gone the moment this fails, so it says so every time, loudly,
    rather than degrading to the process-local behaviour this code exists to
    replace.
    """
    directory = cooldown_dir()
    target = directory / _lease_name(workload)
    temporary = directory / f"{_lease_name(workload)}.{os.getpid()}.tmp"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        temporary.write_text(when.isoformat())
        os.replace(temporary, target)
    except OSError as error:
        print(
            f"[prodrome][WARN] could not write the cooldown lease {target}: {error}. "
            "The cooldown is now PROCESS-LOCAL ONLY -- a second controller will not "
            "see this action and can remediate the same workload again."
        )


def _lease_times():
    """Every workload's last action time as recorded on disk, keyed by filename."""
    out = {}
    try:
        entries = list(cooldown_dir().iterdir())
    except OSError:
        return out

    for path in entries:
        if path.suffix != ".lease":
            continue
        when = _read_lease(path)
        if when is not None:
            out[path.name] = when
    return out


def last_action(workload):
    """When `workload` was last acted on, by ANY process, or None.

    The LATER of this process's memory and the shared lease, not either on its
    own: a longer cooldown is the fail-safe direction, and taking the max means
    neither a stale file nor a restarted process can shorten the window.

    In shadow mode the shared lease is deliberately not consulted. The shadow arm
    must describe what Prodrome would have done ON ITS OWN -- if a live
    controller's leases could make shadow rows read `blocked-cooldown`, the
    control-arm comparison collect runs would be measuring the live arm's
    side effects. Shadow writes no leases either (nothing records under dry-run),
    so this is a read-side gate only.
    """
    local = last_action_time.get(workload)
    shared = None if DRY_RUN else _read_lease(cooldown_dir() / _lease_name(workload))

    if local is None:
        return shared
    if shared is None:
        return local
    return max(local, shared)


def cooldown_active(workload):
    when = last_action(workload)
    if when is None:
        return False

    elapsed = (datetime.now(timezone.utc) - when).total_seconds()

    return elapsed < COOLDOWN_SECONDS


def recent_actions():
    """Workloads acted on inside the rate-limit window, across every process.

    One entry per workload because a workload cannot act twice inside its own
    cooldown and the window is that same length -- so this is an exact count of
    actions, not an approximation. Returned as names rather than a bare number so
    a refusal can say WHICH workloads spent the budget.
    """
    now = datetime.now(timezone.utc)
    seen = {_lease_name(workload): when for workload, when in last_action_time.items()}

    if not DRY_RUN:
        for name, when in _lease_times().items():
            if name not in seen or when > seen[name]:
                seen[name] = when

    return sorted(
        name
        for name, when in seen.items()
        if (now - when).total_seconds() < RATE_LIMIT_WINDOW_SECONDS
    )


def rate_limit_active():
    """True when the fleet-wide action budget for this window is spent.

    Checked AFTER the per-workload cooldown everywhere it is used, because when
    both apply the workload-specific reason is the more useful one for whoever is
    reading the log.
    """
    return len(recent_actions()) >= MAX_ACTIONS_PER_WINDOW


def record_action(workload):
    """Record an issued action, in memory and on disk.

    Called from `execute_action`, where the write actually happens, so that a
    direct caller cannot issue an action without spending the budget for it.
    """
    now = datetime.now(timezone.utc)
    last_action_time[workload] = now
    _write_lease(workload, now)


def kill_switch_paths():
    """Every path whose existence stops the controller, canonical one first.

    Exists so an operator can be TOLD which file to create rather than having to
    infer it from the source. `startup_report()` prints it.
    """
    paths = [STOP_FILE]
    for extra in EXTRA_STOP_PATHS:
        resolved = Path(extra).resolve()
        if resolved not in paths:
            paths.append(resolved)
    return paths


def kill_switch_active():
    return any(path.exists() for path in kill_switch_paths())


def startup_report():
    """The resolved location of every rail an operator has to be able to find.

    A kill switch nobody can locate is not a kill switch, and these paths used to
    be inferable only by reading the source and knowing the working directory.
    Printed by `run_loop` and by `__main__`. LOG_FILE is printed RESOLVED on
    purpose: it is still relative, so this line is how you find out it is not
    where you assumed.
    """
    stops = "\n".join(f"[prodrome]     {path}" for path in kill_switch_paths())
    return "\n".join([
        f"[prodrome] mode={'shadow (DRY_RUN)' if DRY_RUN else 'LIVE'}  namespace={NAMESPACE}",
        "[prodrome] kill switch -- creating ANY of these files stops all remediation:",
        stops,
        f"[prodrome] kill switch currently {'ENGAGED' if kill_switch_active() else 'open'}",
        f"[prodrome] decision log:    {LOG_FILE.resolve()}",
        f"[prodrome] cooldown leases: {cooldown_dir().resolve()}",
        f"[prodrome] rails: <={MAX_REPLICAS} replicas, one action per workload per "
        f"{COOLDOWN_SECONDS}s, <={MAX_ACTIONS_PER_WINDOW} actions across all workloads per "
        f"{RATE_LIMIT_WINDOW_SECONDS}s",
    ])


def attribution(classifier, window):
    """The decision log's `top_features` cell for one prediction, or "".

    Called only where the window is exactly WINDOW_SIZE deep, because
    `classifier.explain()` refuses a short one on purpose -- a confident
    attribution computed over the wrong amount of history is worse than none,
    for the same reason predict() refuses it.

    `top_features` is read through the published classifier interface and
    nothing else, and it is treated as optional: a classifier artifact predating
    the capability writes "" rather than taking the control loop down over an
    audit string.

    What is deliberately NOT caught is a classifier that HAS the capability and
    raises. That is a real defect on the diagnosis side, and swallowing it would
    recreate exactly the bug this column just came out of: a silently empty
    field that looks like "no features mattered". It is safe to fail loudly
    here, too -- this runs before any rail and before any cluster call, so an
    exception cannot leave a half-applied remediation behind.
    """
    describe = getattr(classifier, "top_features", None)
    if describe is None:
        return ""

    return describe(window)


def evaluate_once(apps_api, workload, metrics, detector, classifier, history=None):
    """Run one detector/classifier/policy decision and append its audit row.

    ``metrics`` is one named metric tick. The detector receives the frozen
    ordered vector; the classifier receives a full history DataFrame once it
    has enough ticks to calculate window features.

    Rail order is deliberate: kill switch, then the per-workload cooldown, then
    the fleet-wide rate limit, then execute. The kill switch first so an operator
    reading the log can see STOP was engaged rather than a cooldown that happened
    to be running; the workload-specific reason ahead of the fleet-wide one for
    the same reason, since it is the more useful of the two when both apply.

    These checks are duplicated inside `execute_action`, which enforces all of
    them again. That is deliberate defence in depth, not an oversight: this
    function is not the only caller, and the duplication is what lets the log say
    which rail refused while still guaranteeing the rail holds for callers that
    never come through here.

    `top_features` is filled from the same window the prediction was made on, on
    the only branch where a prediction happens at all. It is written on every
    row that has one, including a `failed` row -- the attribution for an action
    that may have half-landed is the row an operator wants most.

    The cooldown is recorded iff the cluster may have changed -- `executed`, and
    `failed` whose outcome is unknown. Never on `dry-run` (the shadow arm must
    accumulate no state the live arm would, or the control-arm comparison is
    comparing two different policies), never on a block (a blocked action that
    recorded a timestamp would push the window forward on every firing and, at
    this detector's false-positive rate, lock the workload out permanently),
    and never on `at-max`/`no-op`, which issue no write at all. The recording
    itself now happens in `execute_action`, at the point of the write.
    """
    score, fired = detector.score(workload, [metrics.get(metric) for metric in METRICS])
    predicted_class, confidence = ("NORMAL", 1.0)
    action = "nothing"
    top_features = ""
    window_ready = history is not None and len(history) >= WINDOW_SIZE
    if fired and window_ready:
        window = pd.DataFrame(list(history), columns=METRICS)
        predicted_class, confidence = classifier.predict(window)
        top_features = attribution(classifier, window)
        action = decide(predicted_class, confidence)
    elif fired:
        action = "nothing"

    if not fired:
        result = "not-fired"
    elif action == "nothing":
        # The detector fired and the policy declined to act -- below the
        # confidence floor, an unmapped class, or not yet WINDOW_SIZE ticks of
        # history. This used to log `not-fired` next to `fired=True`, two columns
        # flatly contradicting each other, on what is one of the most common rows
        # in the log at a 5.0% false-positive rate with a ~300s firing tail.
        # Announced as a decision-log contract change before being made.
        result = "abstained"
    else:
        if kill_switch_active():
            result = "blocked-kill-switch"
        elif cooldown_active(workload):
            result = "blocked-cooldown"
        elif rate_limit_active():
            result = "blocked-rate-limit"
        else:
            try:
                result = execute_action(apps_api, workload, action)
            except Exception:
                # A patch that raised may still have landed -- a 409, a 403, an
                # API server restart. Previously the exception escaped before
                # log_decision, so the attempt left NO audit row at all and
                # recorded NO cooldown: the loop then retried every tick,
                # forever, invisibly. At a 61% false-positive rate that is an
                # unbounded action storm with no trace. So: log the attempt and
                # re-raise rather than carry on blind. The cooldown for it is
                # recorded inside execute_action, where the write happened.
                log_decision(
                    workload, score, fired, predicted_class, confidence, action, "failed",
                    top_features=top_features,
                )
                raise
            if result == "executed":
                if action == "rolling_restart" and hasattr(detector, "on_restart"):
                    detector.on_restart(workload)

    log_decision(
        workload, score, fired, predicted_class, confidence, action, result,
        top_features=top_features,
    )
    return result


def run_loop(apps_api, workloads, metrics_source, detector, classifier, interval_seconds=15):
    """Continuously evaluate workloads with per-workload 20-tick histories.

    Prints `startup_report()` first and unconditionally. An operator who has to
    stop this thing needs to know which file to create before they need to
    create it, not while they are looking for it.
    """
    import time
    print(startup_report(), flush=True)
    histories = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
    while True:
        for workload in workloads:
            metrics = metrics_source(workload)
            histories[workload].append([metrics.get(metric) for metric in METRICS])
            evaluate_once(
                apps_api, workload, metrics, detector, classifier, histories[workload]
            )
        time.sleep(interval_seconds)

if __name__ == "__main__":
    # `python -m control.controller` is what an operator runs to answer "where
    # is the kill switch and is it on", so it prints the resolved paths rather
    # than a bare yes/no that is only meaningful if you already knew.
    print(startup_report())
