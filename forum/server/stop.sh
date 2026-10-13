#!/usr/bin/env bash
# Stop the forum server and the viewer (data stays in ~/.ccp-forum/data).
FORUM_HOME="${FORUM_HOME:-$HOME/.ccp-forum}"; FORUM_HOME="${FORUM_HOME/#\~/$HOME}"
for p in server viewer; do
  f="$FORUM_HOME/$p.pid"
  if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then kill "$(cat "$f")" && echo "stopped $p ($(cat "$f"))"; fi
  rm -f "$f"
done
