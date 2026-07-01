#Requires -Version 5.1
<#
  RAG Assistant - one-click launcher.

  Installs any missing dependencies (uv, Node.js, Ollama, Qdrant, Redis),
  installs Python + frontend packages, pulls the local LLM, then starts the
  whole pipeline (Redis, Qdrant, Ollama, FastAPI backend, Next.js frontend),
  ingests the knowledge base if empty, and opens the app in its own window.

  Usage (from repo root):
    Double-click RAG-Assistant.bat
      - or -
    powershell -ExecutionPolicy Bypass -File scripts\launch.ps1
  Switches:
    -SkipInstall   Skip dependency install/build steps (faster restarts)
    -CheckOnly     Only report tool/service status, then exit
#>

param(
  [switch]$SkipInstall,
  [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  [ok] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [!]  $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "  [x]  $m" -ForegroundColor Red }

function Have($name) {
  return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Test-Port($port) {
  try {
    $c = New-Object Net.Sockets.TcpClient
    $c.Connect("127.0.0.1", [int]$port)
    $c.Close()
    return $true
  } catch { return $false }
}

function Wait-Http($url, $timeoutSec = 180) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  while ($sw.Elapsed.TotalSeconds -lt $timeoutSec) {
    try {
      $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
      if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { return $true }
    } catch {}
    Start-Sleep -Seconds 2
  }
  return $false
}

function Resolve-Uv {
  $cmd = Get-Command uv -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $p = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
  if (Test-Path $p) { return $p }
  return $null
}

# --------------------------------------------------------------------------
# Dependency provisioning
# --------------------------------------------------------------------------
function Ensure-Uv {
  $uv = Resolve-Uv
  if (-not $uv -and -not $SkipInstall) {
    Step "Installing uv (Python toolchain)"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $uv = Resolve-Uv
  }
  if (-not $uv) { Fail "uv is required. See https://docs.astral.sh/uv/"; exit 1 }
  Ok "uv -> $uv"
  return $uv
}

function Ensure-Node {
  if (Have node) { Ok "node -> $((Get-Command node).Source)"; return }
  if ($SkipInstall) { Fail "Node.js not found"; exit 1 }
  Step "Installing Node.js LTS"
  if (Have winget) {
    winget install -e --id OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements
  } else {
    Fail "Install Node.js LTS from https://nodejs.org then re-run."
    exit 1
  }
  if (-not (Have node)) {
    Warn "Node installed but not on PATH yet - please re-open this launcher."
    exit 1
  }
  Ok "node ready"
}

function Ensure-Ollama {
  if (Have ollama) { Ok "ollama -> $((Get-Command ollama).Source)"; return }
  if ($SkipInstall) { Fail "Ollama not found"; exit 1 }
  Step "Installing Ollama"
  $bundled = Join-Path $Root ".services\OllamaSetup.exe"
  if (Test-Path $bundled) {
    Start-Process -FilePath $bundled -ArgumentList "/VERYSILENT" -Wait
  } elseif (Have winget) {
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
  } else {
    Fail "Install Ollama from https://ollama.com/download then re-run."
    exit 1
  }
  if (-not (Have ollama)) {
    Warn "Ollama installed but not on PATH yet - please re-open this launcher."
    exit 1
  }
  Ok "ollama ready"
}

function Ensure-Binary($name, $exePath, $url) {
  if (Test-Path $exePath) { Ok "$name -> $exePath"; return }
  if ($SkipInstall) { Fail "$name binary missing at $exePath"; exit 1 }
  Step "Downloading $name"
  $destDir = Split-Path $exePath
  New-Item -ItemType Directory -Force $destDir | Out-Null
  $zip = Join-Path $env:TEMP "$name.zip"
  Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
  Expand-Archive -Path $zip -DestinationPath $destDir -Force
  Remove-Item $zip -ErrorAction SilentlyContinue
  if (-not (Test-Path $exePath)) {
    # Some archives nest a folder; find the exe and flatten.
    $found = Get-ChildItem -Path $destDir -Recurse -Filter (Split-Path $exePath -Leaf) |
      Select-Object -First 1
    if ($found) { Copy-Item $found.FullName $exePath -Force }
  }
  if (Test-Path $exePath) { Ok "$name ready" } else { Fail "$name download failed"; exit 1 }
}

function Ensure-QdrantConfig {
  $cfg = Join-Path $Root ".services\qdrant_config.yaml"
  if (Test-Path $cfg) { return }
  @'
storage:
  storage_path: ./qdrant_storage
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
'@ | Set-Content -Path $cfg -Encoding utf8
}

function Ensure-Env {
  $envFile = Join-Path $Root ".env"
  if (Test-Path $envFile) { Ok ".env present"; return }
  Step "Creating .env (local Ollama defaults)"
  @'
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
ROUTER_MODEL=llama3.2
GENERATOR_MODEL=llama3.2
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
ALLOWED_ORIGINS=http://localhost:3000
'@ | Set-Content -Path $envFile -Encoding utf8
  Ok ".env created"
}

# --------------------------------------------------------------------------
# Process management
# --------------------------------------------------------------------------
$script:Started = @()

function Start-Bg($name, $file, $argList, $workdir) {
  New-Item -ItemType Directory -Force (Join-Path $Root "logs") | Out-Null
  $out = Join-Path $Root "logs\$name.out.log"
  $err = Join-Path $Root "logs\$name.err.log"
  $p = Start-Process -FilePath $file -ArgumentList $argList -WorkingDirectory $workdir `
    -PassThru -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
  $script:Started += $p
  Ok "$name started (pid $($p.Id)) - logs\$name.*.log"
}

function Stop-All {
  Step "Shutting down"
  foreach ($p in $script:Started) {
    try {
      if ($p -and -not $p.HasExited) {
        Start-Process -FilePath "taskkill" -ArgumentList "/PID $($p.Id) /T /F" `
          -NoNewWindow -Wait -ErrorAction SilentlyContinue | Out-Null
      }
    } catch {}
  }
  Ok "All launched services stopped."
}

# --------------------------------------------------------------------------
# App launch
# --------------------------------------------------------------------------
function Open-App($url) {
  $candidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
  )
  $browser = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if ($browser) {
    # --app opens a dedicated, chrome-less window (separate from the browser).
    Start-Process $browser "--app=$url --window-size=1240,880"
    Ok "Opened app window ($([IO.Path]::GetFileNameWithoutExtension($browser)) --app)"
  } else {
    Start-Process $url
    Ok "Opened $url in default browser"
  }
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "  RAG Assistant launcher" -ForegroundColor Magenta
Write-Host "  ----------------------" -ForegroundColor Magenta

Step "Checking dependencies"
$Uv = Ensure-Uv
Ensure-Node
Ensure-Ollama
$QdrantExe = Join-Path $Root ".services\qdrant\qdrant.exe"
$RedisExe  = Join-Path $Root ".services\redis\redis-server.exe"
Ensure-Binary "qdrant" $QdrantExe "https://github.com/qdrant/qdrant/releases/download/v1.9.7/qdrant-x86_64-pc-windows-msvc.zip"
Ensure-Binary "redis"  $RedisExe  "https://github.com/tporadowski/redis/releases/download/v5.0.14.1/Redis-x64-5.0.14.1.zip"
Ensure-QdrantConfig
Ensure-Env

if ($CheckOnly) {
  Step "Service ports"
  foreach ($svc in @(@("Redis",6379), @("Qdrant",6333), @("Ollama",11434), @("Backend",8000), @("Frontend",3000))) {
    if (Test-Port $svc[1]) { Ok "$($svc[0]) listening on $($svc[1])" } else { Warn "$($svc[0]) not running (port $($svc[1]))" }
  }
  Write-Host "`nCheck complete." -ForegroundColor Green
  exit 0
}

if (-not $SkipInstall) {
  Step "Installing Python dependencies (uv sync)"
  & $Uv sync
  if ($LASTEXITCODE -ne 0) { Fail "uv sync failed"; exit 1 }
  Ok "Python deps ready"

  Step "Installing frontend dependencies (npm install)"
  Push-Location (Join-Path $Root "frontend")
  npm install --no-fund --no-audit
  $npmCode = $LASTEXITCODE
  Pop-Location
  if ($npmCode -ne 0) { Fail "npm install failed"; exit 1 }
  Ok "Frontend deps ready"

  Step "Pulling local LLM (llama3.2) - first run may take a few minutes"
  ollama pull llama3.2
}

# Build the frontend once (production) if not already built.
$buildId = Join-Path $Root "frontend\.next\BUILD_ID"
if (-not (Test-Path $buildId)) {
  Step "Building frontend (one-time)"
  Push-Location (Join-Path $Root "frontend")
  npm run build
  $buildCode = $LASTEXITCODE
  Pop-Location
  if ($buildCode -ne 0) { Fail "frontend build failed"; exit 1 }
  Ok "Frontend built"
}

try {
  Step "Starting services"

  if (Test-Port 6379) { Ok "Redis already running (6379)" }
  else { Start-Bg "redis" $RedisExe @("--port","6379") $Root }

  if (Test-Port 6333) { Ok "Qdrant already running (6333)" }
  else { Start-Bg "qdrant" $QdrantExe @("--config-path",".services\qdrant_config.yaml") $Root }

  if (Test-Port 11434) { Ok "Ollama already running (11434)" }
  else { Start-Bg "ollama" ((Get-Command ollama).Source) @("serve") $Root }

  Step "Waiting for Qdrant"
  if (-not (Wait-Http "http://localhost:6333/collections" 60)) { Fail "Qdrant did not start"; Stop-All; exit 1 }
  Ok "Qdrant is up"

  # Ingest the knowledge base if the collection is empty / missing.
  $needIngest = $true
  try {
    $r = Invoke-RestMethod "http://localhost:6333/collections/rag_chunks" -TimeoutSec 5
    if ($r.result.points_count -gt 0) { $needIngest = $false; Ok "Knowledge base ready ($($r.result.points_count) chunks)" }
  } catch {}
  if ($needIngest) {
    Step "Ingesting knowledge base (first run)"
    & $Uv run python ingest_markdown.py
  }

  $Npm = (Get-Command npm).Source
  Start-Bg "backend" $Uv @("run","uvicorn","api.main:app","--host","127.0.0.1","--port","8000") $Root
  Start-Bg "frontend" $Npm @("run","start") (Join-Path $Root "frontend")

  Step "Waiting for backend"
  if (-not (Wait-Http "http://localhost:8000/health" 120)) { Warn "Backend health check timed out (it may still be loading models)" }
  else { Ok "Backend is up" }

  Step "Waiting for frontend"
  if (-not (Wait-Http "http://localhost:3000" 120)) { Warn "Frontend did not respond in time" }
  else { Ok "Frontend is up" }

  Open-App "http://localhost:3000"

  Write-Host ""
  Write-Host "  RAG Assistant is running:" -ForegroundColor Green
  Write-Host "    Frontend : http://localhost:3000" -ForegroundColor Gray
  Write-Host "    Backend  : http://localhost:8000/health" -ForegroundColor Gray
  Write-Host ""
  Write-Host "  Press Enter in this window to stop everything." -ForegroundColor Yellow
  [void](Read-Host)
}
finally {
  Stop-All
}
