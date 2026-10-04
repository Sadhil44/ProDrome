# Prodrome controller
import csv
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from control.policy import decide
from ml.features import METRICS, WINDOW_SIZE

# NOTE: `kubernetes` is deliberately NOT imported here. At module scope it made
# this whole module unimportable without the package, so the action wiring, the
# cooldown, the kill switch and the decision-log format could not be tested
# offline or in CI and had to be checked by parsing this file's source. Only
# connect_to_kubernetes() needs it, and it imports it there.


NAMESPACE = "prodrome"
# SETUP.md S8 names this path, and the dashboard and eval harness both expect it.
LOG_FILE = Path("data/decisions/log.csv")
DRY_RUN = True
MAX_REPLICAS = 5
COOLDOWN_SECONDS = 120
STOP_FILE = Path("STOP")

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

    Returns one of: blocked-kill-switch, dry-run, executed, at-max, no-op,
    unknown-action. The decision log's `result` column is this string, so the set
    is part of the decision-log contract.

    The kill switch is re-checked HERE and not only in evaluate_once, because
    evaluate_once is not the only caller: run_loop, the eval harness and any
    head-to-head script can call this directly, and a rail enforced one level up
    is bypassed by every one of them. It is checked before the DRY_RUN branch so
    that STOP wins over everything, including the print.

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

    if action == "rolling_restart":
        restart(apps_api, workload)
        return "executed"

    if action == "scale_out":
        current = get_replicas(apps_api, workload)
        # At or above the cap there is nothing a scale_out can legally do, and
        # it must not patch anyway. `min(current + 1, MAX_REPLICAS)` on a
        # deployment an operator or HPA has already taken above the cap resolves
        # DOWNWARDS -- 8 replicas becomes 5 -- so a decision named scale_out
        # removed capacity from a workload mid-incident on a model's guess.
        # Returning at-max also keeps a no-op patch from logging as `executed`,
        # which an operator cannot tell apart from a real scale-up.
        if current >= MAX_REPLICAS:
            return "at-max"
        scale(apps_api, workload, min(current + 1, MAX_REPLICAS))
        return "executed"

    if action in ("alert_only", "nothing"):
        return "no-op"

    return "unknown-action"

def cooldown_active(workload):
    if workload not in last_action_time:
        return False

    elapsed = (
        datetime.now(timezone.utc) - last_action_time[workload]
    ).total_seconds()

    return elapsed < COOLDOWN_SECONDS

def record_action(workload):
    last_action_time[workload] = datetime.now(timezone.utc)

def kill_switch_active():
    return STOP_FILE.exists()


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

    Rail order is deliberate: kill switch, then cooldown, then execute. The kill
    switch first so an operator reading the log can see STOP was engaged rather
    than a cooldown that happened to be running.

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
    and never on `at-max`/`no-op`, which issue no write at all.
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

    result = "not-fired"
    if fired and action != "nothing":
        if kill_switch_active():
            result = "blocked-kill-switch"
        elif cooldown_active(workload):
            result = "blocked-cooldown"
        else:
            try:
                result = execute_action(apps_api, workload, action)
            except Exception:
                # A patch that raised may still have landed -- a 409, a 403, an
                # API server restart. Previously the exception escaped before
                # log_decision, so the attempt left NO audit row at all and
                # recorded NO cooldown: the loop then retried every tick,
                # forever, invisibly. At a 61% false-positive rate that is an
                # unbounded action storm with no trace. So: log the attempt,
                # count it against the cooldown because its outcome is unknown,
                # and re-raise rather than carry on blind.
                log_decision(
                    workload, score, fired, predicted_class, confidence, action, "failed",
                    top_features=top_features,
                )
                record_action(workload)
                raise
            if result == "executed":
                record_action(workload)
                if action == "rolling_restart" and hasattr(detector, "on_restart"):
                    detector.on_restart(workload)

    log_decision(
        workload, score, fired, predicted_class, confidence, action, result,
        top_features=top_features,
    )
    return result


def run_loop(apps_api, workloads, metrics_source, detector, classifier, interval_seconds=15):
    """Continuously evaluate workloads with per-workload 20-tick histories."""
    import time
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
    if kill_switch_active():
        print("[STOP] Kill switch is active!")
    else:
        print("[OK] Kill switch is not active.")
