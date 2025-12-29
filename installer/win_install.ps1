# SAM3-Tools Windows Installer (uv, .venv only)
Write-Host "=== SAM3-Tools Installer (Windows) ==="

# ---------------------------------------------------------
# Paths / URLs
# ---------------------------------------------------------
$RepoURL    = "https://github.com/AyedaOk/sam3-tools.git"
$LuaRepoURL = "https://github.com/AyedaOk/DT_custom_script.git"
$InstallDir = Join-Path $env:USERPROFILE "sam3-tools"

$VenvDir    = Join-Path $InstallDir ".venv"
$PythonExe  = Join-Path $VenvDir "Scripts\python.exe"

$LauncherDir = Join-Path $InstallDir "launcher"
$LauncherBat = Join-Path $LauncherDir "sam3-tools.bat"

# Darktable Lua Custom scripts folder (Windows)
$PluginDir  = Join-Path $env:LOCALAPPDATA "darktable\lua\Custom"

# Hugging Face (gated model)
$HFModelPage = "https://huggingface.co/facebook/sam3"
$HFTokenPage = "https://huggingface.co/settings/tokens"

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
$ErrorActionPreference = "Stop"

function Confirm-YesNo {
  param(
    [Parameter(Mandatory=$true)][string]$Prompt,
    [ValidateSet("Y","N")][string]$Default = "Y"
  )

  $suffix = if ($Default -eq "Y") { "[Y/n]" } else { "[y/N]" }
  $reply = Read-Host "$Prompt $suffix"
  if ([string]::IsNullOrWhiteSpace($reply)) { $reply = $Default }
  return ($reply -match "^[Yy]")
}

function Has-Cmd {
  param([Parameter(Mandatory=$true)][string]$Name)
  return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Write-Warn($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host $msg -ForegroundColor Red }
function Write-Ok($msg)   { Write-Host $msg -ForegroundColor Green }

# ---------------------------------------------------------
# 1) Dependency checks (winget auto-install)
# ---------------------------------------------------------
$MissingGit = -not (Has-Cmd "git")
$MissingUv  = -not (Has-Cmd "uv")

# VC++ 2015-2022 x64 detection
$vcInstalled = $false
$VCRedistKeys = @(
  "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
  "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
)
foreach ($key in $VCRedistKeys) {
  if (Test-Path $key) {
    try {
      if ((Get-ItemProperty $key -ErrorAction Stop).Installed -eq 1) {
        $vcInstalled = $true
        break
      }
    } catch { }
  }
}
$MissingVC = -not $vcInstalled

if ($MissingUv)  { Write-Warn "[Missing] uv" }
if ($MissingGit) { Write-Warn "[Missing] Git" }
if ($MissingVC)  { Write-Warn "[Missing] Microsoft Visual C++ Redistributable (x64)" }

if ($MissingUv -or $MissingGit -or $MissingVC) {
  if (-not (Has-Cmd "winget")) {
    Write-Err "winget not available. Install missing dependencies manually:"
    if ($MissingUv)  { Write-Err "  - uv (Astral)" }
    if ($MissingGit) { Write-Err "  - Git" }
    if ($MissingVC)  { Write-Err "  - Microsoft VC++ 2015-2022 Redistributable (x64)" }
    exit 1
  }

  Write-Host ""
  if (-not (Confirm-YesNo "Install missing dependencies using winget?" "Y")) {
    exit 1
  }

  if ($MissingVC) {
    winget install -e --id Microsoft.VCRedist.2015+.x64 --source winget --accept-package-agreements --accept-source-agreements
  }
  if ($MissingGit) {
    winget install -e --id Git.Git --source winget --accept-package-agreements --accept-source-agreements
  }
  if ($MissingUv) {
    winget install -e --id astral-sh.uv --source winget --accept-package-agreements --accept-source-agreements
  }

  Write-Host ""
  Write-Warn "Please close this terminal, reopen PowerShell, and re-run the installer."
  exit 0
}

# ---------------------------------------------------------
# 2) Clone or update repo
# ---------------------------------------------------------
if (Test-Path (Join-Path $InstallDir ".git")) {
  Write-Host "Repository exists - updating..."
  git -C $InstallDir pull
} elseif (Test-Path $InstallDir) {
  Write-Err "Install path exists but is not a git repo:"
  Write-Err "  $InstallDir"
  Write-Warn "Move it aside or delete it, then re-run."
  exit 1
} else {
  Write-Host "Cloning repository..."
  git clone $RepoURL $InstallDir
}

Set-Location $InstallDir

# ---------------------------------------------------------
# 3) Create uv environment (.venv)
# ---------------------------------------------------------
if (-not (Test-Path $VenvDir)) {
  Write-Host "Creating virtual environment with uv (.venv)..."
  uv venv
}

if (-not (Test-Path $PythonExe)) {
  Write-Err "uv environment python not found at: $PythonExe"
  Write-Warn "Try deleting .venv and re-running the installer."
  exit 1
}

# Optional quick sanity info (non-fatal)
try {
  $pyver = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
  Write-Host "Using Python: $pyver"
} catch { }

# ---------------------------------------------------------
# 4) Install Python dependencies (uv pip)
# ---------------------------------------------------------
Write-Host "Installing Python dependencies..."

# Use a dedicated temp dir (helps with large wheels / long paths)
$TmpRoot = Join-Path $env:LOCALAPPDATA "sam3-tools\tmp"
New-Item -ItemType Directory -Force -Path $TmpRoot | Out-Null
$origTEMP = $env:TEMP
$origTMP  = $env:TMP
$env:TEMP = $TmpRoot
$env:TMP  = $TmpRoot

try {
  $cpuOnly = Confirm-YesNo "Install CPU-only dependencies (no NVIDIA GPU)?" "N"
  if ($cpuOnly) {
    Write-Host "Installing CPU-only PyTorch..."
    uv pip install --python $PythonExe torch torchvision --index-url https://download.pytorch.org/whl/cpu
  } else {
    $cuda130 = Confirm-YesNo "Install CUDA 13 (cu130) *nightly* PyTorch build? (Check with nvidia-smi)" "N"
    if ($cuda130) {
      uv pip install --python $PythonExe --pre --upgrade --index-url https://download.pytorch.org/whl/nightly/cu130 `
        --extra-index-url https://pypi.org/simple `
        torch torchvision
    }
  }

  $reqPath = Join-Path $InstallDir "requirements.txt"
  if (-not (Test-Path $reqPath)) {
    Write-Err "requirements.txt not found at: $reqPath"
    exit 1
  }

  uv pip install --python $PythonExe -r $reqPath

  # Optional torch version info (non-fatal)
  try {
    $torchver = & $PythonExe -c "import torch; print(torch.__version__)"
    Write-Host "Installed torch: $torchver"
  } catch { }

} finally {
  $env:TEMP = $origTEMP
  $env:TMP  = $origTMP
}

# ---------------------------------------------------------
# 5) Hugging Face login (required for gated model)
# ---------------------------------------------------------
Write-Host ""
Write-Warn "SAM3 checkpoints are gated on Hugging Face."
Write-Host "Request access (wait for approval): $HFModelPage"
Write-Host "Create token (Read):               $HFTokenPage"
Write-Host ""

if (Confirm-YesNo "Log in to Hugging Face now (recommended)?" "Y") {
  Write-Host "Starting Hugging Face login..."
  try {
    # uv run will use the project's .venv automatically when run from repo root
    uv run hf auth login
  } catch {
    Write-Warn "HF login failed (or 'hf' not found). You can try later with: uv run hf auth login"
  }
} else {
  Write-Warn "Skipping Hugging Face login. Model download will fail until you run: uv run hf auth login"
}

# ---------------------------------------------------------
# 6) Optional: trigger first model download
# ---------------------------------------------------------
Write-Host ""
if (Confirm-YesNo "Download the SAM3 model now (~3.5GB+)?" "Y") {
  Write-Host "Triggering model download..."
  $tmpPy = Join-Path $TmpRoot "sam3_download_test.py"
  $py = @'
from transformers import Sam3Model, Sam3Processor

m = Sam3Model.from_pretrained("facebook/sam3")
p = Sam3Processor.from_pretrained("facebook/sam3")
print("Downloaded model + processor OK.")
'@
  Set-Content -Path $tmpPy -Value $py -Encoding UTF8
  try {
    uv run python $tmpPy
  } catch {
    Write-Warn "Model download failed. Common causes: HF access not approved yet, not logged in, or network issues."
  } finally {
    Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
  }
} else {
  Write-Warn "Skipping model download."
}

# ---------------------------------------------------------
# 7) Optional: Darktable plugin installation
# ---------------------------------------------------------
Write-Host ""
if (Confirm-YesNo "Install Darktable plugin?" "Y") {
  if (Test-Path (Join-Path $PluginDir ".git")) {
    Write-Host "Updating Darktable plugin..."
    git -C $PluginDir pull
  } else {
    Write-Host "Installing Darktable plugin..."
    Remove-Item $PluginDir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path (Split-Path $PluginDir) | Out-Null
    git clone $LuaRepoURL $PluginDir
  }
} else {
  Write-Host "Skipping Darktable plugin installation."
}

# ---------------------------------------------------------
# 8) Optional: launcher (.bat)
# ---------------------------------------------------------
Write-Host ""
if (Confirm-YesNo "Generate launcher (launcher\sam3-tools.bat)?" "Y") {
  New-Item -ItemType Directory -Force -Path $LauncherDir | Out-Null

  $bat = @"
@echo off
set APP_DIR=%~dp0..
cd /d "%APP_DIR%" || exit /b 1
"%APP_DIR%\.venv\Scripts\python.exe" main.py %*
"@
  Set-Content -Path $LauncherBat -Value $bat -Encoding ASCII
  Write-Ok "Launcher created: $LauncherBat"
} else {
  Write-Host "Skipping launcher creation."
}

# ---------------------------------------------------------
# 9) Summary
# ---------------------------------------------------------
Write-Host ""
Write-Ok "=== Installation complete ==="
Write-Host "Installed to:   $InstallDir"
Write-Host "Virtual env:    $VenvDir"
Write-Host "Plugin dir:     $PluginDir"
Write-Host ""
Write-Host "Run with:"
Write-Host "  $LauncherBat"
Write-Host "or:"
Write-Host "  cd $InstallDir"
Write-Host "  uv run python main.py"
Write-Host ""
