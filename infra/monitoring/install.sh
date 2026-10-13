#!/usr/bin/env bash
#
# Install (or re-install) the Prodrome monitoring stack. Idempotent.
#
#   bash infra/monitoring/install.sh            # install at the pinned version
#   bash infra/monitoring/install.sh --resolve  # pin to the current chart version
#   bash infra/monitoring/install.sh --verify   # check an existing install only
#
# WHY THIS FILE EXISTS
#
# The stack used to exist only as a helm command in docs/guides/shaurya.md
# section 2.1. On 2026-10-12 the `monitoring` namespace was empty, `helm list -A`
# returned nothing, and kube-state-metrics was gone -- and nothing in the repo
# could say what had been there. collect/scrape.py had nothing to query, the
# healthy baseline had to be polled live through the kubelet cAdvisor endpoint
# by collect/baseline.py with rate(), the mem_pct limit join and
# changes(restarts[1m]) all reconstructed by hand, and there was no retention
# window to recover the lost range from.
#
# The ruling (crosstalk/decisions, 2026-10-13): RESTORE. Prometheus is the
# current collection mechanism and collect/scrape.py is the current collector.
# collect/baseline.py's cAdvisor path stays supported as the documented
# fallback. See infra/README.md for the reasoning and for the data-provenance
# caveat that comes with it.
#
# NOT VERIFIED AGAINST A LIVE CLUSTER. This script was written without a
# cluster to run it on. Read it before you run it.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE="monitoring"
NAMESPACE="monitoring"
CHART="prometheus-community/kube-prometheus-stack"
REPO_URL="https://prometheus-community.github.io/helm-charts"
VALUES="$HERE/values.yaml"
VERSION_FILE="$HERE/chart-version.txt"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1 (SETUP.md section 1)" >&2; exit 1; }; }

add_repo() {
  helm repo add prometheus-community "$REPO_URL" >/dev/null 2>&1 || true
  helm repo update prometheus-community >/dev/null
}

# The eight frozen metrics come from exactly two places. This is the check that
# answers "is this cluster the one the data was collected on", which is the
# question nobody could answer in October.
verify() {
  local fail=0
  echo "--- helm releases"
  helm list -n "$NAMESPACE"

  echo "--- pods"
  kubectl get pods -n "$NAMESPACE"

  echo "--- kube-state-metrics (the ONLY source of the mem_pct limit join)"
  if kubectl get deploy -n "$NAMESPACE" -l app.kubernetes.io/name=kube-state-metrics \
       -o name 2>/dev/null | grep -q .; then
    echo "    present"
  else
    echo "    MISSING -- mem_pct cannot be computed; collect/scrape.py will produce nulls" >&2
    fail=1
  fi

  echo "--- scrape interval (must be 15s; a different one changes every rate([1m]))"
  kubectl get prometheus -n "$NAMESPACE" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"  scrapeInterval="}{.spec.scrapeInterval}{"  retention="}{.spec.retention}{"\n"}{end}' \
    2>/dev/null || { echo "    no Prometheus resource found" >&2; fail=1; }

  echo
  echo "To query it:  kubectl port-forward -n $NAMESPACE svc/${RELEASE}-kube-prometheus-prometheus 9090:9090"
  echo "Then:         python -m collect.scrape --minutes 10 --out data/healthy/metrics.parquet"
  return "$fail"
}

case "${1:-}" in
  --verify)
    need kubectl; need helm
    verify
    exit $?
    ;;
  --resolve)
    # Pin to whatever the repo currently offers, and WRITE IT BACK so the next
    # person installs the same chart. An unpinned chart does not reproduce an
    # environment; it reproduces a date.
    need helm
    add_repo
    resolved="$(helm search repo "$CHART" --output json \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["version"])')"
    [ -n "$resolved" ] || { echo "could not resolve a chart version" >&2; exit 1; }
    printf '%s\n' "$resolved" > "$VERSION_FILE"
    echo "pinned $CHART to $resolved in $VERSION_FILE -- COMMIT THIS FILE."
    exit 0
    ;;
esac

need kubectl; need helm

kubectl cluster-info >/dev/null || { echo "no reachable cluster (kind create cluster --config infra/kind-config.yaml)" >&2; exit 1; }

version="$(tr -d '[:space:]' < "$VERSION_FILE" 2>/dev/null || true)"
if [ -z "$version" ] || [ "$version" = "UNPINNED" ]; then
  cat >&2 <<'MSG'
refusing to install an unpinned chart.

infra/monitoring/chart-version.txt says UNPINNED, which is the honest state:
this file was committed from a seat with no cluster and no network, so no
version could be verified. A floating chart would mean the data collected next
month came from a different monitoring stack than the data collected today,
with nothing in the repo recording the difference -- which is the class of
problem this directory exists to prevent.

Resolve it once, commit the result, and the environment is reproducible:

    bash infra/monitoring/install.sh --resolve
    # then commit infra/monitoring/chart-version.txt and re-run this script
MSG
  exit 1
fi

add_repo

# `upgrade --install`, not `install`: re-running this must converge rather than
# fail on "release already exists". The cluster is disposable and this script
# gets run again every time one is rebuilt.
helm upgrade --install "$RELEASE" "$CHART" \
  --version "$version" \
  --namespace "$NAMESPACE" --create-namespace \
  --values "$VALUES" \
  --wait --timeout 10m

echo
verify
