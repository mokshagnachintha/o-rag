<#
.SYNOPSIS
    Build the Offline RAG Android APK using Docker + Buildozer.

.DESCRIPTION
    This script:
      1. Waits for Docker Desktop to be running (starts it if needed).
      2. Pulls the kivy/buildozer Docker image.
      3. Mounts this project folder into the container.
      4. Runs `buildozer android debug` inside the container.
      5. Copies the resulting APK to the project root so you can
         sideload it directly onto your phone.

    FIRST BUILD: takes 30–90 minutes (NDK + p4a compilation).
    SUBSEQUENT BUILDS: ~5 minutes (everything is cached in .buildozer/).

.NOTES
    Requirements: Docker Desktop for Windows must be installed.
    This script has been tested with Docker Desktop 4.x+ on Windows 10/11.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colour helpers ────────────────────────────────────────────────── #
function Write-Step  { param($msg) Write-Host "`n>>> $msg" -ForegroundColor Cyan   }
function Write-OK    { param($msg) Write-Host "    [OK] $msg"   -ForegroundColor Green  }
function Write-Warn  { param($msg) Write-Host "    [!!] $msg"   -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "`n[FAIL] $msg"   -ForegroundColor Red; exit 1 }

# ── Paths ─────────────────────────────────────────────────────────── #
$ProjectDir  = $PSScriptRoot
$BinDir      = Join-Path $ProjectDir "bin"

# Docker image — use the official Buildozer image maintained by Kivy team
$DockerImage  = "kivy/buildozer:latest"

# ── 1. Ensure Docker is running ───────────────────────────────────── #
Write-Step "Checking Docker Desktop …"

$dockerRunning = $false
try {
    $null = docker info 2>&1
    $dockerRunning = $true
    Write-OK "Docker Desktop is already running."
} catch {
    Write-Warn "Docker Desktop is not running.  Attempting to start it …"
}

if (-not $dockerRunning) {
    # Try to start Docker Desktop
    $desktopExe = "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $desktopExe)) {
        Write-Fail "Docker Desktop not found at '$desktopExe'.`nInstall it from https://www.docker.com/products/docker-desktop/"
    }
    Start-Process $desktopExe
    Write-Host "    Waiting for Docker Engine to start (up to 120 s) …" -ForegroundColor Yellow
    $attempts = 0
    do {
        Start-Sleep -Seconds 5
        $attempts++
        try { $null = docker info 2>&1; $dockerRunning = $true } catch {}
    } while (-not $dockerRunning -and $attempts -lt 24)

    if (-not $dockerRunning) {
        Write-Fail "Docker did not start within 120 seconds.`nPlease open Docker Desktop manually, wait for the whale icon in the tray, then re-run this script."
    }
    Write-OK "Docker Desktop started."
}

# ── 2. Pull the Buildozer image ───────────────────────────────────── #
Write-Step "Pulling Docker image: $DockerImage …"
Write-Host "    (This downloads ~2 GB on the first run)" -ForegroundColor Yellow
docker pull $DockerImage
if ($LASTEXITCODE -ne 0) { Write-Fail "docker pull failed." }
Write-OK "Image is up to date."

# ── 3. Prepare the Docker volume mounts ──────────────────────────── #
# Docker Desktop on Windows WSL2 translates Windows paths automatically.
Write-Step "Project directory: $ProjectDir"

# ── 4. Run Buildozer ─────────────────────────────────────────────── #
Write-Step "Starting Buildozer container …"
Write-Host  "    Command: buildozer android debug" -ForegroundColor Gray
Write-Host  "    The first build downloads the Android NDK/SDK and compiles all"
Write-Host  "    native libraries.  This takes 30–90 minutes on first run." -ForegroundColor Yellow
Write-Host  "    Subsequent builds use the .buildozer/ cache and finish in ~5 min." -ForegroundColor Yellow
Write-Host  ""

# Docker Desktop (WSL2 backend) accepts Windows-style paths directly.
# We mount the whole project at the kivy/buildozer default working directory.
docker run --rm `
    --volume "${ProjectDir}:/home/user/hostcwd" `
    -e BUILDOZER_WARN_ON_ROOT=0 `
    $DockerImage `
    android debug

if ($LASTEXITCODE -ne 0) {
    Write-Fail @"
Buildozer exited with code $LASTEXITCODE.
Common fixes:
  • Scroll up for the first 'ERROR' line in the build log.
  • Delete '.buildozer/' and re-run to do a clean build.
  • Make sure all Python files have no import errors.
"@
}

# ── 5. Find and copy the APK ─────────────────────────────────────── #
Write-Step "Locating APK …"
$apkFiles = Get-ChildItem -Path $BinDir -Filter "*.apk" -ErrorAction SilentlyContinue
if ($null -eq $apkFiles -or $apkFiles.Count -eq 0) {
    Write-Fail "No APK found in $BinDir — check the build log above."
}

# Sort by LastWriteTime and take the newest
$latestApk = ($apkFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
$destApk   = Join-Path $ProjectDir "OfflineRAG-debug.apk"
Copy-Item $latestApk.FullName $destApk -Force

Write-OK "APK copied to: $destApk"

# ── 6. Done ──────────────────────────────────────────────────────── #
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " BUILD COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  APK: $destApk"
Write-Host ""
Write-Host "  To install on your phone:"
Write-Host "    1. Enable 'Unknown sources' / 'Install unknown apps' in"
Write-Host "       Android Settings → Security (or Apps → Special app access)."
Write-Host "    2. Copy OfflineRAG-debug.apk to your phone via USB / cloud."
Write-Host "    3. Open the file manager on your phone and tap the APK."
Write-Host ""
Write-Host "  To install via ADB (USB cable):"
Write-Host "    adb install -r OfflineRAG-debug.apk"
Write-Host ""
Write-Host "  Model file (GGUF):"
Write-Host "    The app will auto-download the model on first launch."
Write-Host "    Or copy your .gguf file to /sdcard/Android/data/"
Write-Host "    com.yourname.offlinerag/files/ with adb push."
Write-Host "============================================================" -ForegroundColor Green
