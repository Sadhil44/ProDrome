# Prodrome controller
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path

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
