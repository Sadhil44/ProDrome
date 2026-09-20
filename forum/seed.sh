#!/usr/bin/env bash
# Create the forum structure: one shelf per aspect (forum/aspects.txt) plus the shared
# `crosstalk` shelf, each with its books. Idempotent (add-shelf/add-book create or re-describe).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORUM_HOME="${FORUM_HOME:-$HOME/.ccp-forum}"; FORUM_HOME="${FORUM_HOME/#\~/$HOME}"
set -a; . "${FORUM_ENV:-$FORUM_HOME/forum.env}"; set +a
FORUM_HOME="${FORUM_HOME/#\~/$HOME}"
export CCP_CLIENT_HOME="$FORUM_HOME/client-home" CCP_SERVER_URL="http://127.0.0.1:${CCP_PORT:-1338}" CCP_AGENT_NAME="${CCP_AGENT_NAME:-forum-seed}"
C="$FORUM_HOME/bin/ccp-client"; S="$CCP_SESSION"
q() { "$@" >/dev/null; }

aspects=()
while read -r name desc; do
  case "$name" in ''|'#'*) continue;; esac
  aspects+=("$name")
  q "$C" add-shelf "$S" "$name" "$desc"
  q "$C" add-book "$S" --shelf "$name" updates    "Progress notes from $name agents: what changed, where, how to run it. One entry per agent per day."
  q "$C" add-book "$S" --shelf "$name" interfaces "Contracts $name exposes to other aspects: APIs, schemas, ports, env vars, file formats. One entry per contract, appended on change."
  echo "shelf $name  (updates, interfaces)"
done < "$HERE/aspects.txt"

q "$C" add-shelf "$S" crosstalk "Shared by every aspect. Read announcements and blockers at the start of every task."
q "$C" add-book "$S" --shelf crosstalk announcements "Changes and decisions that affect other aspects. Label the affected aspects."
q "$C" add-book "$S" --shelf crosstalk blockers      "Stuck on another aspect. Label the aspect you need; the owner appends the answer; the poster appends 'resolved'."
q "$C" add-book "$S" --shelf crosstalk decisions     "Architecture and product decisions with the options considered and why one won."
q "$C" add-book "$S" --shelf crosstalk questions     "Anything cross-cutting that is not an announcement, blocker or decision."
echo "shelf crosstalk  (announcements, blockers, decisions, questions)"

# the full project plan, once (edit forum/master/prodrome-plan-v1.md and append via the client to revise)
if [ -f "$HERE/master/prodrome-plan-v1.md" ] && ! "$C" search-entries "$S" prodrome-plan-v1 2>/dev/null | grep -q '"name": "prodrome-plan-v1"'; then
  "$C" add-entry "$S" --shelf crosstalk --book decisions prodrome-plan-v1 \
    "Prodrome project plan v1: idea, the four-stage pipeline, the data contracts, aspects and owners, ground rules, risks" \
    --labels "$(IFS=,; echo "${aspects[*]}"),crosstalk,plan" \
    "$(cat "$HERE/master/prodrome-plan-v1.md")" >/dev/null
  echo "posted crosstalk/decisions/prodrome-plan-v1"
fi

if ! "$C" search-entries "$S" forum-opened 2>/dev/null | grep -q '"name": "forum-opened"'; then
  "$C" add-entry "$S" --shelf crosstalk --book announcements forum-opened \
    "The forum is open: aspects, books, and the rules every agent must follow" \
    --labels "$(IFS=,; echo "${aspects[*]}"),crosstalk" \
    "Forum endpoint: $FORUM_PUBLIC_URL (session $S). Web view: $VIEWER_PUBLIC_URL. Aspects: ${aspects[*]}. Rules: master_instructions('$S'). Each aspect posts progress to <aspect>/updates, contracts to <aspect>/interfaces, and anything that affects another aspect to crosstalk/announcements labeled with that aspect." >/dev/null
  echo "posted crosstalk/announcements/forum-opened"
elif ! "$C" get "$S" forum-opened --shelf crosstalk --book announcements | grep -qF "$FORUM_PUBLIC_URL"; then
  CCP_APPEND_REASON="public URL changed" "$C" append "$S" forum-opened --shelf crosstalk --book announcements \
    "UPDATE $(date -u +%Y-%m-%dT%H:%MZ): forum endpoint is now $FORUM_PUBLIC_URL, web view $VIEWER_PUBLIC_URL. Re-subscribe with: ccp-client subscribe $S --server $FORUM_PUBLIC_URL" >/dev/null
  echo "appended new public URL to crosstalk/announcements/forum-opened"
fi
