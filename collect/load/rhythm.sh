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
#
# redis    -> redis-benchmark, load set by CLIENT COUNT.
# postgres -> pgbench, load set by TRANSACTION RATE (`-R`). Client count can't be
#             used: pgbench's default workload saturates whatever CPU it's given
#             (0.5 core, 1 core, doesn't matter), so CPU pegs flat with no diurnal
#             rhythm. `-R` throttles tps, so CPU and fs_writes track the daily
#             curve and stay well under the limit -- leaving headroom for a
#             CPU_HOG fault to register. (See the 500m->1000m limit bump in
#             infra/workloads.yaml; needed so peak load isn't near the ceiling.)
set -uo pipefail

ns="${1:?usage: rhythm.sh <namespace> <redis|postgres>}"
kind="${2:?usage: rhythm.sh <namespace> <redis|postgres>}"

# phase within the 600-second day -> load level.
# redis:    concurrent clients.   postgres: pgbench transactions/sec (-R).
level() {
  local p=$(( $(date +%s) % 600 ))
  if [[ "$kind" == postgres ]]; then
    if   (( p < 120 )); then echo 120   # night
    elif (( p < 240 )); then echo 350   # morning ramp
    elif (( p < 360 )); then echo 220   # midday dip
    elif (( p < 480 )); then echo 600   # evening peak
    else echo 120; fi                   # night again
  else
    if   (( p < 120 )); then echo 2     # night
    elif (( p < 240 )); then echo 16    # morning ramp
    elif (( p < 360 )); then echo 8     # midday dip
    elif (( p < 480 )); then echo 24    # evening peak
    else echo 2; fi                     # night again
  fi
}

echo "rhythm: $kind @ $ns  (pid $$)"
while true; do
  v="$(level)"
  case "$kind" in
    redis)
      kubectl exec -n "$ns" deploy/redis -- \
        redis-benchmark -q -t get,set -c "$v" -n "$(( v * 300 ))" >/dev/null 2>&1 || true
      ;;
    postgres)
      # -R v: throttle to v transactions/sec across 8 client threads. Default
      # (UPDATE-heavy) workload so WAL + heap writes give fs_writes a rhythm too.
      # Trailing `postgres` is the DB name (pgbench otherwise picks OS user "root").
      kubectl exec -n "$ns" deploy/postgres -- \
        pgbench -R "$v" -c 8 -U postgres -n -T 12 postgres >/dev/null 2>&1 || true
      ;;
    *)
      echo "unknown kind: $kind" >&2; exit 2 ;;
  esac
  sleep 1
done
