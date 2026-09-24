# Keeps the forum alive: every 30 s check the CCP server and the viewer inside WSL, AND the two
# public tunnels, and repair whatever is down.
#
# WSL itself can stop (it does when Windows sleeps) and take the server and viewer with it. The
# cloudflared tunnels on Windows usually survive that -- but not always: if a quick tunnel loses its
# origin for long enough, Cloudflare tears it down server-side and the cloudflared process is left
# running against a dead hostname. That is invisible to a local-only health check, which is why an
# earlier version of this script could report "restarting: server, viewer", succeed, and leave the
# forum unreachable from outside.
#
# So this script checks BOTH layers. When a tunnel is dead it opens a new one, and because a quick
# tunnel gets a new random hostname every time, it then republishes the master boards and re-seeds
# so the board itself carries the new URL, and re-points this machine's `ccp-forum` MCP at it.
# Teammates still have to re-subscribe -- nothing can fix that except a stable hostname.
#
#   Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File forum\watchdog.ps1"
param([int]$IntervalSeconds = 30, [int]$CcpPort = 1338, [int]$ViewerPort = 8000)

$repo = Split-Path -Parent $PSScriptRoot
$wslRepo = ((wsl.exe -e wslpath -a ($repo -replace '\\', '/')) -replace "`0", "").Trim()
$home_ = $env:USERPROFILE
$logDir = Join-Path $home_ ".ccp-forum"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "watchdog.log"
$cloudflared = Join-Path $home_ "bin\cloudflared.exe"
if (-not (Test-Path $cloudflared)) { $cloudflared = "cloudflared" }

function Say([string]$m) { Add-Content $log "$(Get-Date -Format s) $m" }

function LocalUp([int]$port) {
    try { $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://localhost:$port/health"; return $true }
    catch { return $false }
}

function PublicUp([string]$url) {
    if (-not $url) { return $false }
    try { $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 15 "$url/health"; return $true }
    catch { return $false }
}

function ForumEnv([string]$key) {
    $v = wsl.exe -e bash -lc "grep ^$key= ~/.ccp-forum/forum.env | cut -d= -f2"
    return (($v -replace "`0", "").Trim())
}

# Kill any cloudflared bound to $port, start a fresh tunnel, return its new public URL.
function NewTunnel([int]$port, [string]$name) {
    $tlog = Join-Path $logDir "tunnel-$name.log"
    Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" |
        Where-Object { $_.CommandLine -like "*localhost:$port*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Remove-Item $tlog -ErrorAction SilentlyContinue
    Start-Process $cloudflared -ArgumentList "tunnel --url http://localhost:$port --no-autoupdate" `
        -RedirectStandardError $tlog -WindowStyle Hidden | Out-Null
    for ($i = 0; $i -lt 80; $i++) {
        Start-Sleep -Milliseconds 500
        # Test-Path goes true the instant Start-Process creates the redirect file, but the file
        # is empty for a moment after that -- and Get-Content -Raw on an empty file returns $null,
        # which makes [regex]::Matches throw "Value cannot be null". That aborted a real recovery
        # on 2026-09-24: the watchdog survived on its outer catch, but the dead tunnel was never
        # replaced. Read first, guard for null, and only then match.
        $raw = $null
        if (Test-Path $tlog) { $raw = Get-Content $tlog -Raw -ErrorAction SilentlyContinue }
        if ($raw) {
            # cloudflared's log also mentions its OWN control endpoint, api.trycloudflare.com, and
            # a naive match picks that up instead of the assigned hostname -- which silently writes
            # a bogus URL into forum.env and republishes the boards with it. Exclude the known
            # non-tunnel hosts rather than trusting the first match.
            $u = [regex]::Matches($raw, 'https://[a-z0-9-]+\.trycloudflare\.com') |
                 ForEach-Object { $_.Value } |
                 Where-Object { $_ -notmatch '^https://(api|update)\.' } |
                 Select-Object -Unique -First 1
            if ($u) { return $u }
        }
    }
    return $null
}

Say "watchdog started (pid $PID), interval ${IntervalSeconds}s"

# WSL itself can wedge -- the whole hvsocket layer starts refusing connections
# (Wsl/Service/0x80072747, WSAENOBUFS) and every wsl.exe call fails instantly.
# Retrying a restart every 30 s against that achieves nothing, floods the log,
# and adds load to a subsystem already short of resources. Back off instead, and
# say plainly that this needs a human, because it does: clearing it takes
# `wsl --shutdown`, which also restarts Docker's distro and anything running in it.
$wslFailures = 0
$backoffUntil = [datetime]::MinValue

while ($true) {
    try {
        if ((Get-Date) -lt $backoffUntil) {
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        # --- layer 1: the services inside WSL -------------------------------------------------
        $ccpLocal = LocalUp $CcpPort
        $viewerLocal = LocalUp $ViewerPort
        if (-not $ccpLocal -or -not $viewerLocal) {
            $what = @(); if (-not $ccpLocal) { $what += "server" }; if (-not $viewerLocal) { $what += "viewer" }
            Say "local down: $($what -join ', ') -- restarting in WSL"
            $cmd = "cd '$wslRepo'"
            if (-not $ccpLocal) { $cmd += " && forum/server/run.sh" }
            if (-not $viewerLocal) { $cmd += " && forum/viewer/run.sh" }
            $out = wsl.exe -e bash -lc $cmd 2>&1
            $text = ($out -join "`n")
            Say $text

            if ($LASTEXITCODE -ne 0 -and $text -match 'Wsl/Service|HCS_E_|could not be performed because the system lacked') {
                $wslFailures++
                if ($wslFailures -ge 3) {
                    $wait = [Math]::Min(30, 5 * [Math]::Pow(2, $wslFailures - 3))
                    $backoffUntil = (Get-Date).AddMinutes($wait)
                    Say ("WSL itself is failing ($wslFailures consecutive). Backing off $wait min. " +
                         "This does not self-heal: it needs `wsl --shutdown` from a human, which also " +
                         "restarts Docker's distro and any containers in it.")
                }
            } else {
                $wslFailures = 0
            }
            Start-Sleep 3
        } else {
            $wslFailures = 0
        }

        # --- layer 2: the public tunnels ------------------------------------------------------
        # Only meaningful once the services are actually up locally; a dead origin looks like a
        # dead tunnel and we do not want to churn hostnames for no reason.
        if ((LocalUp $CcpPort) -and (LocalUp $ViewerPort)) {
            $forumUrl = ForumEnv "FORUM_PUBLIC_URL"
            $viewerUrl = ForumEnv "VIEWER_PUBLIC_URL"
            $forumOk = PublicUp $forumUrl
            $viewerOk = PublicUp $viewerUrl

            if (-not $forumOk -or -not $viewerOk) {
                $what = @(); if (-not $forumOk) { $what += "forum" }; if (-not $viewerOk) { $what += "viewer" }
                Say "PUBLIC tunnel down: $($what -join ', ') -- reopening (hostname will change)"

                $newForum = if (-not $forumOk) { NewTunnel $CcpPort "ccp" } else { $forumUrl }
                $newViewer = if (-not $viewerOk) { NewTunnel $ViewerPort "viewer" } else { $viewerUrl }

                if (-not $newForum -or -not $newViewer) {
                    Say "tunnel reopen FAILED (forum=$newForum viewer=$newViewer); will retry next tick"
                } else {
                    Say "new URLs: forum=$newForum viewer=$newViewer"
                    # Persist, restart the server against them, republish the rules and reseed so
                    # the board itself advertises the new endpoint.
                    $reload = "cd '$wslRepo' && FORUM_PUBLIC_URL='$newForum' VIEWER_PUBLIC_URL='$newViewer' " +
                              "forum/server/run.sh && forum/server/publish-master.sh && forum/seed.sh && forum/viewer/run.sh"
                    $out = wsl.exe -e bash -lc $reload 2>&1
                    Say ($out -join "`n")

                    # Re-point this machine's own client + MCP so the host agent keeps working.
                    $key = ForumEnv "CCP_CLIENT_KEY"
                    $wslHome = ((wsl.exe -e bash -lc "echo `$HOME") -replace "`0", "").Trim()
                    wsl.exe -e bash -lc "CCP_CLIENT_KEY=$key $wslHome/.local/bin/ccp-client subscribe prodrome --server $newForum" 2>&1 | Out-Null
                    if (Get-Command claude -ErrorAction SilentlyContinue) {
                        $agent = if ($env:CCP_AGENT_NAME) { $env:CCP_AGENT_NAME } else { "$env:USERNAME-$env:COMPUTERNAME".ToLower() }
                        $inner = "CCP_SERVER_URL=$newForum CCP_CLIENT_KEY=$key CCP_CLIENT_BIN=$wslHome/.local/bin/ccp-client " +
                                 "CCP_AGENT_NAME=$agent exec $wslHome/.ccp-client/mcp-venv/bin/ccp-mcp-server"
                        claude mcp remove ccp-forum --scope user 2>$null | Out-Null
                        claude mcp add ccp-forum --scope user -- wsl.exe -e bash -lc $inner 2>&1 | Out-Null
                        Say "re-pointed ccp-forum MCP at $newForum (restart Claude Code to reload)"
                    }
                    Say "RECOVERED. Teammates must re-subscribe: ccp-client subscribe prodrome --server $newForum"
                }
            }
        }
    } catch {
        Say "watchdog iteration error: $_"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
