# Keeps the forum alive: every 30 s check the CCP server and the viewer inside WSL and restart
# whatever is down. WSL itself can stop (it did once when the Windows disk filled up) and take
# both with it; the cloudflared tunnels on Windows survive and only need their origins back,
# so the public URLs do not change. online.ps1 starts this hidden; offline.ps1 stops it.
#   Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File forum\watchdog.ps1"
param([int]$IntervalSeconds = 30, [int]$CcpPort = 1338, [int]$ViewerPort = 8000)
$repo = Split-Path -Parent $PSScriptRoot
$wslRepo = ((wsl.exe -e wslpath -a ($repo -replace '\\', '/')) -replace "`0", "").Trim()
$log = Join-Path $env:USERPROFILE ".ccp-forum\watchdog.log"

function Up([int]$port) {
    try { $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://localhost:$port/health"; return $true } catch { return $false }
}
while ($true) {
    $ccp = Up $CcpPort; $viewer = Up $ViewerPort
    if (-not $ccp -or -not $viewer) {
        $what = @(); if (-not $ccp) { $what += "server" }; if (-not $viewer) { $what += "viewer" }
        Add-Content $log "$(Get-Date -Format s) restarting: $($what -join ', ')"
        $cmd = "cd '$wslRepo'"
        if (-not $ccp) { $cmd += " && forum/server/run.sh" }
        if (-not $viewer) { $cmd += " && forum/viewer/run.sh" }
        $out = wsl.exe -e bash -lc $cmd 2>&1
        Add-Content $log ($out -join "`n")
    }
    Start-Sleep -Seconds $IntervalSeconds
}
