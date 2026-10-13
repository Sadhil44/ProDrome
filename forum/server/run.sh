#!/usr/bin/env bash
# Start (or restart) the prodrome CCP forum server on this Linux/WSL host.
#   forum/server/run.sh            start in background, print URLs
#   forum/server/run.sh --fg       run in the foreground
#   FORUM_PUBLIC_URL=https://... forum/server/run.sh   override + persist the public URL
# Config: ~/.ccp-forum/forum.env (created from forum.env.example). Binaries: ~/.ccp-forum/bin.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORUM_HOME="${FORUM_HOME:-$HOME/.ccp-forum}"; FORUM_HOME="${FORUM_HOME/#\~/$HOME}"
ENV_FILE="${FORUM_ENV:-$FORUM_HOME/forum.env}"
UPSTREAM="${CCP_UPSTREAM:-https://ccp.spl.team}"
mkdir -p "$FORUM_HOME"
[ -f "$ENV_FILE" ] || cp "$HERE/forum.env.example" "$ENV_FILE"

# env vars given on the command line win over the file, and get persisted
override_url="${FORUM_PUBLIC_URL:-}"; override_vurl="${VIEWER_PUBLIC_URL:-}"
set -a; . "$ENV_FILE"; set +a
FORUM_HOME="${FORUM_HOME/#\~/$HOME}"
BIN="$FORUM_HOME/bin"; DATA="$FORUM_HOME/data"; DL="$FORUM_HOME/downloads"; CH="$FORUM_HOME/client-home"
mkdir -p "$BIN" "$DATA" "$DL" "$CH"

setenv() { # setenv KEY VALUE -> persist into forum.env and export
  if grep -qE "^$1=" "$ENV_FILE"; then sed -i "s|^$1=.*|$1=$2|" "$ENV_FILE"; else echo "$1=$2" >> "$ENV_FILE"; fi
  export "$1=$2"
}
[ -z "$override_url" ]  || setenv FORUM_PUBLIC_URL "$override_url"
[ -z "$override_vurl" ] || setenv VIEWER_PUBLIC_URL "$override_vurl"

# --- keys --------------------------------------------------------------------
if [ -z "${CCP_ADMIN_KEY:-}" ]; then
  setenv CCP_ADMIN_KEY "ccp-admin-$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  echo "generated admin key (in $ENV_FILE)"
fi
if [ -z "${CCP_CLIENT_KEY:-}" ]; then
  k=""
  for f in "$HOME/.claude.json" /mnt/c/Users/*/.claude.json; do
    [ -f "$f" ] || continue
    k="$(grep -oE 'CCP_CLIENT_KEY=[A-Za-z0-9._-]+' "$f" | head -1 | cut -d= -f2 || true)"; [ -n "$k" ] && break
  done
  [ -n "$k" ] || k="$(curl -fsSL -m 10 "$UPSTREAM/setup-client.sh" 2>/dev/null | sed -n 's/^CLIENT_KEY="\(.*\)"/\1/p' | head -1 || true)"
  [ -n "$k" ] || k="ccp-client-$(head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  setenv CCP_CLIENT_KEY "$k"
fi

# --- URLs (placeholders until a tunnel / box IP is known) ---------------------
[ -n "${FORUM_PUBLIC_URL:-}" ]  || setenv FORUM_PUBLIC_URL "http://127.0.0.1:${CCP_PORT:-1338}"
[ -n "${VIEWER_PUBLIC_URL:-}" ] || setenv VIEWER_PUBLIC_URL "http://127.0.0.1:${VIEWER_PORT:-8010}"

# --- binaries ------------------------------------------------------------------
case "$(uname -m)" in aarch64|arm64) arch=aarch64;; *) arch=x86_64;; esac
fetch() { curl -fsSL -m 180 "$UPSTREAM/downloads/$1" -o "$2" && chmod 0755 "$2"; }
if [ ! -x "$BIN/ccp-server" ]; then
  if [ -x "$HOME/.local/bin/ccp-server" ]; then cp "$HOME/.local/bin/ccp-server" "$BIN/"; else fetch "ccp-server-linux-$arch" "$BIN/ccp-server"; fi
fi
if [ ! -x "$BIN/ccp-client" ]; then
  if [ -x "$HOME/.local/bin/ccp-client" ]; then cp "$HOME/.local/bin/ccp-client" "$BIN/"; else fetch "ccp-client-linux-$arch" "$BIN/ccp-client"; fi
fi
for f in ccp-client-linux-x86_64 ccp-client-linux-aarch64 ccp-client-darwin-x86_64 ccp-client-darwin-aarch64 ccp-server-linux-x86_64 ccp-mcp.tar.gz; do
  [ -s "$DL/$f" ] || curl -fsSL -m 180 "$UPSTREAM/downloads/$f" -o "$DL/$f" 2>/dev/null || rm -f "$DL/$f"
done

# --- (re)start -----------------------------------------------------------------
PIDFILE="$FORUM_HOME/server.pid"; LOG="$FORUM_HOME/server.log"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then kill "$(cat "$PIDFILE")"; sleep 1; fi
export CCP_SERVER_DATA_DIR="$DATA" CCP_HTTP_BASE_URL="$FORUM_PUBLIC_URL" CCP_SERVER_URL="$FORUM_PUBLIC_URL"
export CCP_HTTP_LISTENER_ADDR="${CCP_HTTP_LISTENER_ADDR:-0.0.0.0:${CCP_PORT:-1338}}"
export CCP_CLIENT_KEY CCP_ADMIN_KEY CCP_SESSION_VISIBILITY CCP_SESSION_PURPOSE CCP_SESSION_LABELS CCP_AGENT_STATUS_TTL_SECONDS
[ "${1:-}" != "--fg" ] || exec "$BIN/ccp-server" "$CCP_SESSION"
nohup "$BIN/ccp-server" "$CCP_SESSION" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"

LOCAL="http://127.0.0.1:${CCP_PORT:-1338}"
for _ in $(seq 1 40); do curl -fs -m 2 "$LOCAL/health" >/dev/null 2>&1 && break; sleep 0.25; done
curl -fs -m 2 "$LOCAL/health" >/dev/null || { echo "server did not come up; log:"; tail -20 "$LOG"; exit 1; }
CCP_CLIENT_HOME="$CH" CCP_SERVER_URL="$LOCAL" "$BIN/ccp-client" subscribe-all >/dev/null   # local enrollment for viewer/seed

cat <<EOF
forum server up   pid $(cat "$PIDFILE")   log $LOG
  agents connect to:  $FORUM_PUBLIC_URL   (local $LOCAL)   session $CCP_SESSION
  admin page:         $FORUM_PUBLIC_URL/admin   (key in $ENV_FILE)
EOF
