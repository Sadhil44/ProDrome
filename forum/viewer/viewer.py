#!/usr/bin/env python3
"""prodrome forum: read-only web view of the CCP session, plus the agent installer and
client downloads. Standard library only.

Config comes from ~/.ccp-forum/forum.env (or env vars): CCP_SESSION, CCP_PORT, CCP_ADMIN_KEY,
CCP_CLIENT_KEY, FORUM_PUBLIC_URL, VIEWER_PORT, VIEWER_PUBLIC_URL, VIEWER_POLL_SECONDS.

Routes:
  /                 the forum page (auto-refreshes)
  /api/state        everything the page renders, as JSON
  /rules            master boards as plain text
  /setup-client.sh  installer for agent machines (templated with this forum's URL + key)
  /downloads/<f>    ccp-client binaries and the MCP package
  /health
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent


def _load_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


_HOME = Path(os.path.expanduser(os.environ.get("FORUM_HOME", "~/.ccp-forum")))
ENV = _load_env(Path(os.path.expanduser(os.environ.get("FORUM_ENV", str(_HOME / "forum.env")))))


def cfg(key: str, default: str = "") -> str:
    return os.environ.get(key) or ENV.get(key) or default


FORUM_HOME = Path(os.path.expanduser(cfg("FORUM_HOME", str(_HOME))))
SESSION = cfg("CCP_SESSION", "prodrome")
LOCAL_URL = f"http://127.0.0.1:{cfg('CCP_PORT', '1338')}"
PUBLIC_URL = cfg("FORUM_PUBLIC_URL", LOCAL_URL)
VIEWER_PORT = int(cfg("VIEWER_PORT", "8000"))
VIEWER_PUBLIC_URL = cfg("VIEWER_PUBLIC_URL", f"http://127.0.0.1:{VIEWER_PORT}")
ADMIN_KEY = cfg("CCP_ADMIN_KEY")
CLIENT_KEY = cfg("CCP_CLIENT_KEY")
CLIENT_BIN = cfg("CCP_CLIENT_BIN", str(FORUM_HOME / "bin" / "ccp-client"))
CLIENT_HOME = cfg("VIEWER_CLIENT_HOME", str(FORUM_HOME / "client-home"))
DOWNLOADS = Path(cfg("CCP_DOWNLOAD_DIR", str(FORUM_HOME / "downloads")))
POLL = float(cfg("VIEWER_POLL_SECONDS", "5"))
INSTALLER = HERE.parent / "agent" / "install.sh"


# --- CCP access ---------------------------------------------------------------
def client(*args: str) -> str:
    env = dict(os.environ, CCP_CLIENT_HOME=CLIENT_HOME, CCP_CLIENT_KEY=CLIENT_KEY, CCP_SERVER_URL=LOCAL_URL)
    p = subprocess.run([CLIENT_BIN, *args], capture_output=True, text=True, env=env, timeout=30)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout).strip() or f"ccp-client {args[0]} failed")
    return p.stdout


def client_json(*args: str):
    out = client(*args).strip()
    return json.loads(out) if out else {}


def export_bundle() -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "session.droplet"
        client("export", SESSION, "--output", str(out))
        return json.loads(out.read_text())


def admin(path: str):
    req = urllib.request.Request(LOCAL_URL + path, headers={"X-CCP-Admin-Key": ADMIN_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _ts(v) -> int:
    """Normalize CCP timestamps (unix seconds or milliseconds, str or int) to milliseconds."""
    try:
        n = int(str(v))
    except (TypeError, ValueError):
        return 0
    return n if n > 10**12 else n * 1000


def build_state() -> dict:
    bundle = export_bundle()
    brief = client_json("brief-me", SESSION)
    master = client_json("master-instructions", SESSION)
    try:
        books_meta = client_json("search-books", SESSION, " ")
    except Exception:
        books_meta = []

    shelves: dict[str, dict] = {}
    for s in brief.get("shelves", []):
        shelves[s["shelf_name"]] = {"name": s["shelf_name"], "description": s.get("description", ""), "books": {}}
    for b in books_meta if isinstance(books_meta, list) else []:
        sh = shelves.setdefault(b.get("shelf_name", "main"), {"name": b.get("shelf_name", "main"), "description": b.get("shelf_description", ""), "books": {}})
        sh["books"].setdefault(b["book_name"], {"name": b["book_name"], "description": b.get("description", "") or b.get("book_description", ""), "entries": []})
    for e in bundle.get("entries", []):
        sh = shelves.setdefault(e["shelf_name"], {"name": e["shelf_name"], "description": e.get("shelf_description", ""), "books": {}})
        sh["description"] = sh["description"] or e.get("shelf_description", "")
        bk = sh["books"].setdefault(e["book_name"], {"name": e["book_name"], "description": e.get("book_description", ""), "entries": []})
        bk["description"] = bk["description"] or e.get("book_description", "")
        history = [
            {
                "at": _ts(h.get("created_at")),
                "by": h.get("agent_name") or h.get("client_common_name") or "?",
                "host": h.get("host_name") or "",
                "reason": h.get("reason") or "",
                "content": h.get("appended_content") or "",
            }
            for h in (e.get("history") or [])
        ]
        bk["entries"].append(
            {
                "name": e["name"],
                "description": e.get("description", ""),
                "labels": e.get("labels") or [],
                "context": e.get("context", ""),
                "history": history,
                "updated_at": max([h["at"] for h in history] + [0]),
            }
        )
    for sh in shelves.values():
        for bk in sh["books"].values():
            bk["entries"].sort(key=lambda x: -x["updated_at"])
        sh["books"] = sorted(sh["books"].values(), key=lambda b: b["name"])

    statuses = []
    for name in shelves:
        try:
            statuses += client_json("team-status", SESSION, "--team", name)
        except Exception:
            pass
    for s in statuses:
        s["updated_at"] = _ts(s.get("updated_at"))
        s["expires_at"] = _ts(s.get("expires_at"))
    statuses.sort(key=lambda s: -s["updated_at"])

    activity, overview = [], {}
    if ADMIN_KEY:
        for path in (f"/v1/admin/activity?session={SESSION}&limit=300", "/v1/admin/activity?limit=300"):
            try:
                activity = admin(path)
                break
            except Exception:
                continue
        try:
            overview = admin("/v1/admin/overview")
        except Exception:
            overview = {}
    for a in activity:
        a["at"] = _ts(a.get("created_at"))
    activity.sort(key=lambda a: -a["at"])

    return {
        "ok": True,
        "error": None,
        "generated_at": int(time.time() * 1000),
        "session": SESSION,
        "forum_url": PUBLIC_URL,
        "viewer_url": VIEWER_PUBLIC_URL,
        "master": master,
        "shelves": sorted(shelves.values(), key=lambda s: (s["name"] != "crosstalk", s["name"])),
        "statuses": statuses,
        "activity": activity,
        "totals": {"shelves": brief.get("total_shelves", 0), "books": brief.get("total_books", 0), "entries": brief.get("total_entries", 0)},
        "overview": overview,
    }


STATE: dict = {"ok": False, "error": "starting", "generated_at": 0}
LOCK = threading.Lock()


def poller() -> None:
    global STATE
    while True:
        try:
            new = build_state()
        except Exception as ex:  # keep the last good state, surface the error
            with LOCK:
                STATE = dict(STATE, ok=False, error=str(ex))
        else:
            with LOCK:
                STATE = new
        time.sleep(POLL)


# --- HTTP ---------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>prodrome forum</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#2a2f3a;--fg:#e6e8ee;--dim:#9aa3b2;--acc:#7cc4ff;--warn:#ffb86b;--ok:#7ee787}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif}
a{color:var(--acc);text-decoration:none}code,pre{font:12.5px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace}
header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;z-index:2}
header h1{font-size:16px;margin:0}header .m{color:var(--dim)}header input{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px;min-width:220px}
main{padding:12px 16px;display:grid;gap:14px;grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
@media(max-width:1000px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;min-width:0}
section h2{font-size:13px;margin:0 0 8px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;display:flex;justify-content:space-between;align-items:center}
.wide{grid-column:1/-1}
.chip{display:inline-block;padding:1px 7px;border-radius:999px;border:1px solid var(--line);color:var(--dim);font-size:11px;margin:0 4px 2px 0}
.chip.a{border-color:var(--acc);color:var(--acc)}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;margin:6px 0;max-height:360px;overflow:auto}
details{border-top:1px solid var(--line);padding:6px 0}details>summary{cursor:pointer;list-style:none}details>summary::-webkit-details-marker{display:none}
summary .n{font-weight:600}summary .d{color:var(--dim)}
.row{display:flex;gap:10px;padding:5px 0;border-top:1px solid var(--line);align-items:baseline}.row:first-child{border-top:0}
.t{color:var(--dim);font-size:11.5px;white-space:nowrap}.who{color:var(--warn)}.kind{color:var(--acc);font-size:11.5px}
.hist{margin:6px 0 0 10px;padding-left:10px;border-left:2px solid var(--line)}
.err{background:#3a1d1d;border:1px solid #7a3030;color:#ffb4b4;padding:8px 12px;border-radius:8px;margin:12px 16px 0}
.shelf{margin-bottom:10px}.shelf>h3{margin:8px 0 2px;font-size:14px}.shelf>h3 span{color:var(--dim);font-weight:400;margin-left:8px}
.book{margin:4px 0 4px 12px}.book>h4{margin:6px 0 2px;font-size:13px}.book>h4 span{color:var(--dim);font-weight:400;margin-left:8px}
.empty{color:var(--dim);font-style:italic;padding:4px 0}
.copy{cursor:pointer;color:var(--dim)}.copy:hover{color:var(--fg)}
select{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:3px 6px}
</style></head><body>
<header>
  <h1>prodrome forum</h1>
  <span class="m">CCP endpoint: <code id="furl"></code> <span class="copy" title="copy" onclick="cp('furl')">&#x2398;</span></span>
  <span class="m">session <code id="sess"></code></span>
  <span class="m" id="tot"></span>
  <input id="q" placeholder="filter entries, labels, activity..." oninput="render()">
  <span class="m" id="upd"></span>
</header>
<div id="err" class="err" hidden></div>
<main>
  <section class="wide"><h2>connect an agent <span class="m">one line per machine</span></h2>
    <pre id="connect"></pre></section>
  <section><h2>who is working on what <span id="stc" class="m"></span></h2><div id="status"></div></section>
  <section><h2>activity <select id="kind" onchange="render()"><option value="">all kinds</option></select></h2><div id="act"></div></section>
  <section class="wide"><h2>the board <span class="m">shelf = aspect &rsaquo; book &rsaquo; entry (newest first)</span></h2><div id="board"></div></section>
  <section class="wide"><h2>rules (master boards)</h2><details><summary>session <code>prodrome</code> master board</summary><pre id="mSess"></pre></details><details><summary>global master board</summary><pre id="mGlob"></pre></details></section>
</main>
<script>
let S=null;const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ago=ms=>{if(!ms)return'';const d=(Date.now()-ms)/1000;if(d<60)return Math.floor(d)+'s ago';if(d<3600)return Math.floor(d/60)+'m ago';if(d<86400)return Math.floor(d/3600)+'h ago';return new Date(ms).toLocaleString()};
function cp(id){navigator.clipboard&&navigator.clipboard.writeText($(id).textContent)}
const open=new Set();document.addEventListener('toggle',e=>{const k=e.target.dataset&&e.target.dataset.k;if(!k)return;e.target.open?open.add(k):open.delete(k)},true);
async function load(){try{const r=await fetch('/api/state',{cache:'no-store'});S=await r.json();render()}catch(e){$('err').hidden=false;$('err').textContent='viewer unreachable: '+e}}
function match(t){const q=$('q').value.trim().toLowerCase();return !q||t.toLowerCase().includes(q)}
function render(){if(!S)return;$('err').hidden=!S.error;$('err').textContent=S.error?('last refresh failed: '+S.error):'';
 $('furl').textContent=S.forum_url;$('sess').textContent=S.session;$('upd').textContent='refreshed '+ago(S.generated_at);
 $('tot').textContent=`${S.totals.shelves} shelves / ${S.totals.books} books / ${S.totals.entries} entries`;
 $('connect').textContent=`curl -fsSL ${S.viewer_url}/setup-client.sh | sh\n# already have ccp-client?  ccp-client subscribe ${S.session} --server ${S.forum_url}\n# then: master-instructions ${S.session}  (the rules)   brief-me ${S.session}  (overview)`;
 $('mSess').textContent=(S.master.session||{}).content||'(empty)';$('mGlob').textContent=(S.master.global||{}).content||'(empty)';
 const st=S.statuses.filter(s=>match(s.team+' '+s.agent_name+' '+s.status));$('stc').textContent=st.length+' active';
 $('status').innerHTML=st.length?st.map(s=>`<div class="row"><span class="chip a">${esc(s.team)}</span><span class="who">${esc(s.agent_name)}</span><span>${esc(s.status)}</span><span class="t">${ago(s.updated_at)}</span></div>`).join(''):'<div class="empty">nobody has set_status yet</div>';
 const kinds=[...new Set(S.activity.map(a=>a.kind))].sort();const ks=$('kind');const cur=ks.value;ks.innerHTML='<option value="">all kinds</option>'+kinds.map(k=>`<option ${k===cur?'selected':''}>${esc(k)}</option>`).join('');
 const act=S.activity.filter(a=>(!ks.value||a.kind===ks.value)&&match([a.kind,a.shelf_name,a.book_name,a.entry_name,a.agent_name,a.actor,a.content,a.reason].join(' '))).slice(0,150);
 $('act').innerHTML=act.length?act.map(a=>`<div class="row"><span class="t">${ago(a.at)}</span><span class="kind">${esc(a.kind)}</span><span class="who">${esc(a.agent_name||a.actor||'')}</span><span><b>${esc([a.shelf_name,a.book_name,a.entry_name].filter(Boolean).join('/'))}</b>${a.reason?' <i class="t">'+esc(a.reason)+'</i>':''}${a.content?'<div class="t">'+esc(String(a.content).slice(0,240))+'</div>':''}</span></div>`).join(''):'<div class="empty">no activity'+(S.overview&&S.overview.sessions?'':' (admin key not configured: activity feed unavailable)')+'</div>';
 $('board').innerHTML=S.shelves.map(sh=>{const books=sh.books.map(bk=>{const es=bk.entries.filter(e=>match([e.name,e.description,e.labels.join(' '),e.context].join(' ')));
   const body=es.length?es.map(e=>{const k=sh.name+'/'+bk.name+'/'+e.name;return `<details data-k="${esc(k)}" ${open.has(k)?'open':''}><summary><span class="n">${esc(e.name)}</span> <span class="d">${esc(e.description)}</span> <span class="t">${ago(e.updated_at)}</span><div>${e.labels.map(l=>`<span class="chip">${esc(l)}</span>`).join('')}</div></summary><pre>${esc(e.context)}</pre>${e.history.length?'<div class="hist">'+e.history.map(h=>`<div class="row"><span class="t">${ago(h.at)}</span><span class="who">${esc(h.by)}${h.host?' @'+esc(h.host):''}</span><span>${h.reason?'<i>'+esc(h.reason)+'</i> ':''}<span class="t">+${h.content.length} chars</span></span></div>`).join('')+'</div>':''}</details>`}).join(''):'<div class="empty">no entries</div>';
   return `<div class="book"><h4>${esc(bk.name)}<span>${esc(bk.description)}</span></h4>${body}</div>`}).join('');
   return `<div class="shelf"><h3>${esc(sh.name)}<span>${esc(sh.description)}</span></h3>${books||'<div class="empty">no books</div>'}</div>`}).join('');
}
load();setInterval(load,__POLL__);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8", extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            with LOCK:
                ok = STATE.get("ok", False)
            return self._send(200, json.dumps({"status": "ok" if ok else "degraded", "error": STATE.get("error")}).encode(), "application/json")
        if path == "/":
            return self._send(200, PAGE.replace("__POLL__", str(int(POLL * 1000))).encode(), "text/html; charset=utf-8")
        if path == "/api/state":
            with LOCK:
                body = json.dumps(STATE).encode()
            return self._send(200, body, "application/json")
        if path == "/rules":
            with LOCK:
                m = STATE.get("master") or {}
            text = f"# session master board ({SESSION})\n\n{(m.get('session') or {}).get('content', '')}\n\n# global master board\n\n{(m.get('global') or {}).get('content', '')}\n"
            return self._send(200, text.encode())
        if path == "/setup-client.sh":
            if not INSTALLER.exists():
                return self._send(404, b"installer missing")
            script = (
                INSTALLER.read_text()
                .replace("__FORUM_URL__", PUBLIC_URL)
                .replace("__VIEWER_URL__", VIEWER_PUBLIC_URL)
                .replace("__CLIENT_KEY__", CLIENT_KEY)
                .replace("__SESSION__", SESSION)
            )
            return self._send(200, script.encode(), "text/x-shellscript; charset=utf-8")
        if path.startswith("/downloads/"):
            name = unquote(path[len("/downloads/"):])
            f = (DOWNLOADS / name).resolve()
            if "/" in name or not f.is_file() or DOWNLOADS.resolve() not in f.parents:
                return self._send(404, b"no such download")
            return self._send(200, f.read_bytes(), "application/octet-stream", {"Content-Disposition": f'attachment; filename="{name}"'})
        return self._send(404, b"not found")

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    threading.Thread(target=poller, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", VIEWER_PORT), Handler)
    print(f"viewer on http://0.0.0.0:{VIEWER_PORT}  (public {VIEWER_PUBLIC_URL})  forum {PUBLIC_URL}  session {SESSION}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
