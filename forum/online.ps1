# Put the forum on the public internet from this Windows machine (the server runs in WSL).
# Opens two Cloudflare quick tunnels (no account needed): one for the CCP API (1338) and one
# for the web view (8010), writes the public URLs into the forum config, restarts the server
# with them, republishes the rules, seeds the structure and starts the viewer.
#   powershell -ExecutionPolicy Bypass -File forum\online.ps1
# Quick-tunnel URLs change every time cloudflared restarts: re-run this and hand out the new URL.
# Stop everything: forum\offline.ps1
param([int]$CcpPort = 1338, [int]$ViewerPort = 8010)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$wslRepo = ((wsl.exe -e wslpath -a ($repo -replace '\\', '/')) -replace "`0", "").Trim()
$logDir = Join-Path $env:USERPROFILE ".ccp-forum"; New-Item -ItemType Directory -Force $logDir | Out-Null

function Start-Tunnel([int]$port, [string]$name) {
    $log = Join-Path $logDir "tunnel-$name.log"
    Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" | Where-Object { $_.CommandLine -like "*localhost:$port*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Remove-Item $log -ErrorAction SilentlyContinue
    Start-Process cloudflared -ArgumentList "tunnel --url http://localhost:$port --no-autoupdate" -RedirectStandardError $log -WindowStyle Hidden | Out-Null
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Path $log) {
            # Skip cloudflared's own control endpoint (api.trycloudflare.com) and the updater host;
            # matching those instead of the assigned hostname yields a URL that resolves but is not
            # this tunnel, which is worse than failing outright.
            $u = [regex]::Matches((Get-Content $log -Raw), 'https://[a-z0-9-]+\.trycloudflare\.com') |
                 ForEach-Object { $_.Value } |
                 Where-Object { $_ -notmatch '^https://(api|update)\.' } |
                 Select-Object -Unique -First 1
            if ($u) { return $u }
        }
    }
    throw "cloudflared did not report a URL for port $port (see $log)"
}

wsl.exe -e bash -lc "cd '$wslRepo' && forum/server/run.sh" | Out-Host          # 1. server up (local URL for now)
$forumUrl  = Start-Tunnel $CcpPort "ccp"
$viewerUrl = Start-Tunnel $ViewerPort "viewer"
Write-Host "tunnels: $forumUrl -> :$CcpPort    $viewerUrl -> :$ViewerPort"
# 2. restart with the public URLs, publish the rules, seed, start the viewer
wsl.exe -e bash -lc "cd '$wslRepo' && FORUM_PUBLIC_URL='$forumUrl' VIEWER_PUBLIC_URL='$viewerUrl' forum/server/run.sh && forum/server/publish-master.sh && forum/seed.sh && forum/viewer/run.sh" | Out-Host
Start-Sleep 2
try { $h = Invoke-RestMethod -TimeoutSec 30 "$forumUrl/health"; Write-Host "public health: $($h.status)" } catch { Write-Warning "public health check failed: $_" }

# 3. point this machine's own agents at the new URL (Claude Code MCP `ccp-forum` + WSL client enrollment).
#    Paths are read out of WSL rather than hardcoded, so this works on whoever's laptop is hosting.
$key     = ((wsl.exe -e bash -lc "grep ^CCP_CLIENT_KEY= ~/.ccp-forum/forum.env | cut -d= -f2") -replace "`0", "").Trim()
$wslHome = ((wsl.exe -e bash -lc "echo `$HOME") -replace "`0", "").Trim()
wsl.exe -e bash -lc "CCP_CLIENT_KEY=$key $wslHome/.local/bin/ccp-client subscribe prodrome --server $forumUrl 2>/dev/null || true" | Out-Host
if (Get-Command claude -ErrorAction SilentlyContinue) {
    $agent = if ($env:CCP_AGENT_NAME) { $env:CCP_AGENT_NAME } else { "$env:USERNAME-$env:COMPUTERNAME".ToLower() }
    $mcp   = "$wslHome/.ccp-client/mcp-venv/bin/ccp-mcp-server"
    $client = "$wslHome/.local/bin/ccp-client"
    claude mcp remove ccp-forum --scope user 2>$null | Out-Null
    claude mcp add ccp-forum --scope user -- wsl.exe -e bash -lc "CCP_SERVER_URL=$forumUrl CCP_CLIENT_KEY=$key CCP_CLIENT_BIN=$client CCP_AGENT_NAME=$agent exec $mcp" | Out-Null
    Write-Host "Claude Code MCP 'ccp-forum' now points at $forumUrl (restart Claude Code to reload)"
}
# 4. watchdog: restarts the server/viewer inside WSL if they die (tunnels stay up, URLs unchanged)
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -like "*watchdog.ps1*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\watchdog.ps1`"" | Out-Null
Write-Host ""
Write-Host "FORUM (agents, CCP API):  $forumUrl     session: prodrome"
Write-Host "WEB VIEW (humans):        $viewerUrl"
Write-Host "connect an agent machine: curl -fsSL $viewerUrl/setup-client.sh | sh"
