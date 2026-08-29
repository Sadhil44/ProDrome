#!/usr/bin/env bash
# Diurnal load for redis OR postgres in one namespace, driven via `kubectl exec`
# into the workload pod (no port-forward needed -- robust for a 12h run).
# Follows the same 10-minute compressed day as the k6 nginx profile:
# night / morning ramp / midday dip / evening peak / night.
#
#   collect/load/rhythm.sh <namespace> <redis|postgres>
#
# Runs until killed. The benchmark client runs INSIDE the workload container, so
# a small part of that container's measured CPU is generator overhead -- an
# acceptable tradeoff for a synthetic testbed, and the same one the stress-ng
# chaos injector makes (guide 5.2).
set -uo pipefail

ns="${1:?usage: rhythm.sh <namespace> <redis|postgres>}"
kind="${2:?usage: rhythm.sh <namespace> <redis|postgres>}"

# phase within the 600-second day -> number of concurrent clients
concurrency() {
  local p=$(( $(date +%s) % 600 ))
  if   (( p < 120 )); then echo 2    # night
  elif (( p < 240 )); then echo 16   # morning ramp
  elif (( p < 360 )); then echo 8    # midday dip
  elif (( p < 480 )); then echo 24   # evening peak
  else echo 2; fi                    # night again
}

echo "rhythm: $kind @ $ns  (pid $$)"
while true; do
  c="$(concurrency)"
  case "$kind" in
    redis)
      kubectl exec -n "$ns" deploy/redis -- \
        redis-benchmark -q -t get,set -c "$c" -n "$(( c * 300 ))" >/dev/null 2>&1 || true
      ;;
    postgres)
      # trailing `postgres` is the DB name -- pgbench otherwise defaults it to
      # the in-container OS user ("root"), which has no database.
      kubectl exec -n "$ns" deploy/postgres -- \
        pgbench -U postgres -n -c "$c" -T 12 postgres >/dev/null 2>&1 || true
      ;;
    *)
      echo "unknown kind: $kind" >&2; exit 2 ;;
  esac
  sleep 1
done
