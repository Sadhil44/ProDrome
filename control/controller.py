import csv
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from kubernetes import client, config
from control.policy import decide
from ml.features import METRICS, WINDOW_SIZE


NAMESPACE = "prodrome"
LOG_FILE = Path("control/decisions.csv")
DRY_RUN = True
MAX_REPLICAS = 5
COOLDOWN_SECONDS = 120
STOP_FILE = Path("STOP")

last_action_time = {}

def connect_to_kubernetes():
    config.load_kube_config()
    return client.AppsV1Api()


def get_replicas(apps_api, deployment_name):
    deployment = apps_api.read_namespaced_deployment(
        name=deployment_name,
        namespace=NAMESPACE,
    )

    return deployment.spec.replicas


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


def log_decision(
    workload,
    detector_score,
    fired,
    predicted_class,
    confidence,
    action,
    result,
):
    file_exists = LOG_FILE.exists()

    with LOG_FILE.open("a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "workload",
                "detector_score",
                "fired",
                "predicted_class",
                "confidence",
                "action",
                "result",
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            workload,
            detector_score,
            fired,
            predicted_class,
            confidence,
            action,
            result,
        ])

def execute_action(apps_api, workload, action):
    if DRY_RUN:
        print(f"[DRY RUN] Would {action} {workload}")
        return "dry-run"

    if action in ("restart", "rolling_restart"):
        restart(apps_api, workload)
        return "executed"

    if action == "scale_out":
        current = get_replicas(apps_api, workload)
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


def evaluate_once(apps_api, workload, metrics, detector, classifier, history=None):
    """Run one detector/classifier/policy decision and append its audit row.

    ``metrics`` is one named metric tick. The detector receives the frozen
    ordered vector; the classifier receives a full history DataFrame once it
    has enough ticks to calculate window features.
    """
    score, fired = detector.score(workload, [metrics.get(metric) for metric in METRICS])
    predicted_class, confidence = ("NORMAL", 1.0)
    action = "nothing"
    window_ready = history is not None and len(history) >= WINDOW_SIZE
    if fired and window_ready:
        predicted_class, confidence = classifier.predict(
            pd.DataFrame(list(history), columns=METRICS)
        )
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
            result = execute_action(apps_api, workload, action)
            if result == "executed":
                record_action(workload)
                if action in ("restart", "rolling_restart") and hasattr(detector, "on_restart"):
                    detector.on_restart(workload)

    log_decision(workload, score, fired, predicted_class, confidence, action, result)
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
