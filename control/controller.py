# Prodrome controller
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path

from kubernetes import client, config


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

    if action == "restart":
        restart(apps_api, workload)
        return "executed"

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


if __name__ == "__main__":
    if kill_switch_active():
        print("[STOP] Kill switch is active!")
    else:
        print("[OK] Kill switch is not active.")
