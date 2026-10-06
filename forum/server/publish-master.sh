#!/usr/bin/env bash
# Publish forum/master/global.md and forum/master/<session>.md to the CCP master boards
# (the rules every agent reads). Re-run after editing them or after the public URL changes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORUM_HOME="${FORUM_HOME:-$HOME/.ccp-forum}"; FORUM_HOME="${FORUM_HOME/#\~/$HOME}"
set -a; . "${FORUM_ENV:-$FORUM_HOME/forum.env}"; set +a
LOCAL="http://127.0.0.1:${CCP_PORT:-1338}"
render() { sed -e "s#__FORUM_URL__#$FORUM_PUBLIC_URL#g" -e "s#__VIEWER_URL__#$VIEWER_PUBLIC_URL#g" "$1"; }
put() { # put <api path> <markdown file>
  render "$2" | python3 -c 'import json,sys; print(json.dumps({"content": sys.stdin.read()}))' \
    | curl -fsS -m 10 -X PUT -H "X-CCP-Admin-Key: $CCP_ADMIN_KEY" -H "Content-Type: application/json" --data-binary @- "$LOCAL$1" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print("published %s (%d chars, %s)" % (sys.argv[1], len(d.get("content","")), d.get("updated_at")))' "$1"
}
put /v1/admin/master "$HERE/../master/global.md"
put "/v1/admin/sessions/$CCP_SESSION/master" "$HERE/../master/$CCP_SESSION.md"
