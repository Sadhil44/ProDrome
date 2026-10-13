#!/usr/bin/env bash
# Start (or restart) the read-only web view in the background.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORUM_HOME="${FORUM_HOME:-$HOME/.ccp-forum}"; FORUM_HOME="${FORUM_HOME/#\~/$HOME}"
PIDFILE="$FORUM_HOME/viewer.pid"; LOG="$FORUM_HOME/viewer.log"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then kill "$(cat "$PIDFILE")"; sleep 0.5; fi
nohup python3 "$HERE/viewer.py" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
port="$(grep -E '^VIEWER_PORT=' "${FORUM_ENV:-$FORUM_HOME/forum.env}" | cut -d= -f2)"; port="${port:-8010}"
for _ in $(seq 1 40); do curl -fs -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break; sleep 0.25; done
curl -fs -m 2 "http://127.0.0.1:$port/health" >/dev/null || { echo "viewer did not come up; log:"; tail -20 "$LOG"; exit 1; }
echo "viewer up  pid $(cat "$PIDFILE")  http://127.0.0.1:$port  log $LOG"
