#Requires -Version 5.1
<#
  Stop every RAG Assistant service by freeing its port.
  Useful if the launcher window was closed without pressing Enter.
#>

$ErrorActionPreference = "SilentlyContinue"

$ports = @(
  @{ Name = "Frontend"; Port = 3000 },
  @{ Name = "Backend";  Port = 8000 },
  @{ Name = "Qdrant";   Port = 6333 },
  @{ Name = "Redis";    Port = 6379 }
  # Ollama (11434) is left running - it is a shared background service.
)

foreach ($svc in $ports) {
  $conns = Get-NetTCPConnection -LocalPort $svc.Port -State Listen -ErrorAction SilentlyContinue
  if (-not $conns) {
    Write-Host "  [--] $($svc.Name) not running (port $($svc.Port))" -ForegroundColor DarkGray
    continue
  }
  $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($procId in $pids) {
    taskkill /PID $procId /T /F | Out-Null
    Write-Host "  [ok] Stopped $($svc.Name) (pid $procId)" -ForegroundColor Green
  }
}

Write-Host "`nDone." -ForegroundColor Green
