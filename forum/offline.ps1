# Take the forum offline: stop the tunnels, the viewer and the server. Data stays in WSL ~/.ccp-forum/data.
$repo = Split-Path -Parent $PSScriptRoot
$wslRepo = ((wsl.exe -e wslpath -a ($repo -replace '\\', '/')) -replace "`0", "").Trim()
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -like "*watchdog.ps1*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
wsl.exe -e bash -lc "cd '$wslRepo' && forum/server/stop.sh"
