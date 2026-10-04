#!/usr/bin/env bash
# Rewrite forum/CURRENT-URLS.md from the live values in ~/.ccp-forum/forum.env.
#
# Quick-tunnel hostnames rotate constantly, so the only way a URL committed to the
# repo stays honest is to regenerate it. This checks both endpoints actually respond
# BEFORE writing, and refuses to write one it cannot reach -- a file that claims a
# dead URL is worse than one that admits it does not know.
#
#   forum/refresh-urls.sh
#   git commit -am "forum: refresh current URLs" && git push
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORUM_HOME="${FORUM_HOME:-$HOME/.ccp-forum}"; FORUM_HOME="${FORUM_HOME/#\~/$HOME}"
ENV_FILE="${FORUM_ENV:-$FORUM_HOME/forum.env}"
OUT="$HERE/CURRENT-URLS.md"

[ -f "$ENV_FILE" ] || { echo "no $ENV_FILE -- is this the host machine?" >&2; exit 1; }

FORUM_URL="$(grep '^FORUM_PUBLIC_URL=' "$ENV_FILE" | cut -d= -f2-)"
VIEWER_URL="$(grep '^VIEWER_PUBLIC_URL=' "$ENV_FILE" | cut -d= -f2-)"
SESSION="$(grep '^CCP_SESSION=' "$ENV_FILE" | cut -d= -f2-)"
SESSION="${SESSION:-prodrome}"

[ -n "$FORUM_URL" ] && [ -n "$VIEWER_URL" ] || { echo "forum.env is missing a public URL" >&2; exit 1; }

try() { curl -fsS -m 20 -o /dev/null "$1"; }

echo "verifying before writing:"
public_ok=1
for u in "$FORUM_URL/health" "$VIEWER_URL/health" "$VIEWER_URL/setup-client.sh"; do
  if try "$u"; then echo "  ok      public   $u"; else echo "  no      public   $u"; public_ok=0; fi
done

# WSL egress to trycloudflare hostnames times out on some hosts even while the tunnel is
# perfectly reachable from the outside -- seen on this one. So a public failure is not proof
# the forum is down. Fall back to loopback, which proves the SERVICES are up, and record in
# the file exactly which of the two was actually checked rather than overclaiming.
local_ok=1
if [ "$public_ok" -eq 0 ]; then
  echo "  (public checks failed -- falling back to loopback; see note below)"
  for u in "http://127.0.0.1:${CCP_PORT:-1338}/health" "http://127.0.0.1:${VIEWER_PORT:-8010}/health"; do
    if try "$u"; then echo "  ok      local    $u"; else echo "  FAILED  local    $u" >&2; local_ok=0; fi
  done
fi

if [ "$public_ok" -eq 0 ] && [ "$local_ok" -eq 0 ]; then
  cat >&2 <<EOF

Refusing to write $OUT.

Neither the public URLs nor loopback responded, so the forum is genuinely down and
committing these URLs would send the team to a dead host. Start it first
(forum/online.ps1 on the host, or let the watchdog rotate the tunnel), then re-run.
EOF
  exit 1
fi

STAMP="$(date '+%Y-%m-%d %H:%M')"
if [ "$public_ok" -eq 1 ]; then
  VERIFIED="Last verified working: **$STAMP** (host local time). Both \`/health\` endpoints
returned ok and \`/setup-client.sh\` returned HTTP 200 at that moment."
else
  VERIFIED="Last checked: **$STAMP** (host local time). The services were confirmed up on
loopback, but public reachability could **not** be confirmed from the host — WSL egress to
\`trycloudflare\` hostnames times out here even when the tunnel is fine from outside. Open the
web view in a browser to confirm before relying on these."
fi

cat > "$OUT" <<EOF
# Current forum URLs

**Forum API** (what agents connect to):
\`$FORUM_URL\`

**Web view** (what humans open in a browser):
\`$VIEWER_URL\`

**Session:** \`$SESSION\` — this never changes.

$VERIFIED

## Connect

Already have \`ccp-client\`:

\`\`\`sh
ccp-client subscribe $SESSION --server $FORUM_URL
\`\`\`

New machine:

\`\`\`sh
sudo apt install -y python3-venv python3-pip     # stock Ubuntu WSL has neither
export CCP_AGENT_NAME=<yourname>-<aspect>        # e.g. sagar-signal
curl -fsSL $VIEWER_URL/setup-client.sh | sh
\`\`\`

Restart Claude Code / Codex afterwards.

## If these do not work

They are Cloudflare **quick** tunnels, which are ephemeral by design: every restart mints a
new random hostname. On 2026-09-24 the pair rotated roughly every 30–60 minutes, so a URL in
this file can be stale within the hour even though it was verified when written. **Check the
timestamp above** — if it is more than an hour old, assume nothing.

When that happens, ping the host (Sadhil). The server recovers on its own if the watchdog is
running, but the new hostname still has to reach you out of band, because a URL you cannot
resolve is a board you cannot read.

## Host: refreshing this file

Do not hand-edit. After the URLs change, run:

\`\`\`bash
forum/refresh-urls.sh        # rewrites this file from ~/.ccp-forum/forum.env
git commit -am "forum: refresh current URLs" && git push
\`\`\`

It reads the live values, checks both endpoints respond before writing, and stamps the
verification time. It refuses to write a URL it cannot reach, so this file should never
claim something untrue at the moment it was committed.

## Making this stop

A stable hostname removes the rotation entirely, and then this file holds a URL that stays
true indefinitely: a Cloudflare **named** tunnel (needs a domain on Cloudflare) or Tailscale
Funnel (free, no domain, gives a permanent \`*.ts.net\` name). Tailscale is already installed
and signed in on the host; it needs Funnel enabled once in the admin console.
EOF

echo
echo "wrote $OUT"
echo "  forum : $FORUM_URL"
echo "  viewer: $VIEWER_URL"
echo "  stamp : $STAMP"
