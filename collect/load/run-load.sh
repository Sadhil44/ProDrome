#!/usr/bin/env bash
# Part 4 load generation orchestrator (docs/guides/shaurya.md).
#
# Drives diurnal load against redis, nginx, and postgres in BOTH the `prodrome`
# and `control` namespaces -- identical load on both arms, or the eventual
# comparison is meaningless (guide 4.2). Run it in its own terminal tab.
# It self-terminates when the k6 runs finish; Ctrl+C stops everything early.
#
#   DAYS=84 bash collect/load/run-load.sh     # ~14h -- the overnight run
#   DAYS=3  bash collect/load/run-load.sh     # ~30 min -- smoke test
#
# Keep the Mac awake for the whole run:  caffeinate -i -w $$   (in another tab)
set -uo pipefail
cd "$(dirname "$0")/../.."                    # repo root

DAYS="${DAYS:-84}"
NSES=(prodrome control)
pids=()
k6pids=()

cleanup() {
  echo; echo "stopping load ..."
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
  pkill -P $$ 2>/dev/null || true
  wait 2>/dev/null || true
  echo "done."
}
trap cleanup EXIT INT TERM

# Resilient port-forward: kubectl port-forward dies silently (guide 2.2), so loop it.
pf_retry() {
  while true; do
    kubectl port-forward -n "$1" svc/nginx "$2:80" >/dev/null 2>&1
    sleep 2
  done
}

echo "one-time pgbench init (scale 5) ..."
for ns in "${NSES[@]}"; do
  # trailing `postgres` is the DB name (pgbench defaults it to the OS user otherwise)
  if kubectl exec -n "$ns" deploy/postgres -- pgbench -i -s 5 -U postgres postgres >/dev/null 2>&1; then
    echo "  $ns ok"
  else
    echo "  $ns -- init failed, continuing (postgres load will be weak)"
  fi
done

port=8080
for ns in "${NSES[@]}"; do
  pf_retry "$ns" "$port" &
  pids+=("$!")
  sleep 3

  k6 run -e TARGET="http://localhost:${port}" -e DAYS="$DAYS" \
    collect/load/k6-diurnal.js >"collect/load/k6-${ns}.log" 2>&1 &
  kp="$!"; pids+=("$kp"); k6pids+=("$kp")
  echo "nginx    @ ${ns}  -> k6 via localhost:${port}  (log: collect/load/k6-${ns}.log)"

  bash collect/load/rhythm.sh "$ns" redis &
  pids+=("$!")
  bash collect/load/rhythm.sh "$ns" postgres &
  pids+=("$!")
  echo "redis    @ ${ns}  -> rhythm loop"
  echo "postgres @ ${ns}  -> rhythm loop"

  port=$(( port + 1 ))
done

echo
echo "load running: ${#pids[@]} processes, ~$(( DAYS * 10 )) min of compressed days."
echo "check it landed:  python collect/scrape.py --minutes 10 --out /tmp/peek.parquet"
echo "Ctrl+C to stop early."
echo

wait "${k6pids[@]}"     # exit (and clean up) once every k6 run has finished
