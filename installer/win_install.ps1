# SAM2-Tools Windows Installer (uv, .venv only)
Write-Host "=== SAM2-Tools Installer (Windows) ==="

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
$RepoURL       = "https://github.com/AyedaOk/sam2-tools.git"
$InstallDir    = "$env:USERPROFILE\sam2-tools"

$VenvDir       = Join-Path $InstallDir ".venv"
$PythonExe     = Join-Path $VenvDir "Scripts\python.exe"

$ConfigDir     = Join-Path $env:APPDATA "sam2"
$CheckpointDir = Join-Path $ConfigDir "checkpoints"

$ModelURLs = @(
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)

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

function Download-File {
  param(
    [Parameter(Mandatory=$true)][string]$Url,
    [Parameter(Mandatory=$true)][string]$Dest
  )

  try {
    if (Has-Cmd "Start-BitsTransfer") {
      Start-BitsTransfer -Source $Url -Destination $Dest
    } else {
      Invoke-WebRequest -Uri $Url -OutFile $Dest
    }
  } catch {
    Invoke-WebRequest -Uri $Url -OutFile $Dest
  }
}

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

if ($MissingUv)  { Write-Host "[Missing] uv" -ForegroundColor Yellow }
if ($MissingGit) { Write-Host "[Missing] Git" -ForegroundColor Yellow }
if ($MissingVC)  { Write-Host "[Missing] Microsoft Visual C++ Redistributable (x64)" -ForegroundColor Yellow }

if ($MissingUv -or $MissingGit -or $MissingVC) {
  if (-not (Has-Cmd "winget")) {
    Write-Host "winget not available. Install missing dependencies manually:" -ForegroundColor Red
    if ($MissingUv)  { Write-Host "  - uv (Astral)" -ForegroundColor Red }
    if ($MissingGit) { Write-Host "  - Git" -ForegroundColor Red }
    if ($MissingVC)  { Write-Host "  - Microsoft VC++ 2015-2022 Redistributable (x64)" -ForegroundColor Red }
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
  Write-Host "Please close this terminal, reopen PowerShell, and re-run the installer." -ForegroundColor Yellow
  exit 0
}

# ---------------------------------------------------------
# 2) Clone or update repo
# ---------------------------------------------------------
if (Test-Path (Join-Path $InstallDir ".git")) {
  Write-Host "Repository exists - updating..."
  git -C $InstallDir pull
} elseif (Test-Path $InstallDir) {
  Write-Host "Install path exists but is not a git repo:" -ForegroundColor Red
  Write-Host "  $InstallDir" -ForegroundColor Red
  Write-Host "Move it aside or delete it, then re-run." -ForegroundColor Yellow
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
  Write-Host "uv environment python not found at: $PythonExe" -ForegroundColor Red
  Write-Host "Try deleting .venv and re-running the installer." -ForegroundColor Yellow
  exit 1
}

# ---------------------------------------------------------
# 4) Install Python dependencies (uv pip)
# ---------------------------------------------------------
Write-Host "Installing Python dependencies..."

$cpuReply = Read-Host "Install CPU-only dependencies (no NVIDIA GPU)? [y/N]"
if ($cpuReply -match "^[Yy]") {
  Write-Host "Installing CPU-only PyTorch..."
  uv pip install --python $PythonExe torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
  $cudaReply = Read-Host "Do you want to install the CUDA 13 (cu130) PyTorch build? [y/N]"
  if ($cudaReply -match "^[Yy]") {
    uv pip install --python $PythonExe --pre --upgrade --index-url https://download.pytorch.org/whl/nightly/cu130 `
      --extra-index-url https://pypi.org/simple `
      torch torchvision
  }
}

uv pip install --python $PythonExe -r (Join-Path $InstallDir "requirements.txt")

# ---------------------------------------------------------
# 5) Create config
# ---------------------------------------------------------
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
$configPath = Join-Path $ConfigDir "config.yaml"

if (-not (Test-Path $configPath)) {
  Write-Host "Generating config..."
  & $PythonExe (Join-Path $InstallDir "main.py") --config
} else {
  Write-Host "Config already exists: $configPath"
}

# ---------------------------------------------------------
# 6) Model download (optional)
# ---------------------------------------------------------
$reply = Read-Host "Download SAM2 model checkpoints now? [Y/n]"
if ($reply -notmatch "^[Nn]") {
  New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null

  foreach ($url in $ModelURLs) {
    $name = Split-Path $url -Leaf
    $dest = Join-Path $CheckpointDir $name

    if (Test-Path $dest) {
      Write-Host "Already exists: $name"
      continue
    }

    Write-Host "Downloading $name ..."
    Download-File -Url $url -Dest $dest
  }
}

# ---------------------------------------------------------
# 7) Darktable plugin installation (optional)
# ---------------------------------------------------------
$PluginDir = Join-Path $env:LOCALAPPDATA "darktable\lua\Custom"

$reply = Read-Host "Install Darktable plugin? [Y/n]"
if ($reply -notmatch "^[Nn]") {

  if (Test-Path (Join-Path $PluginDir ".git")) {
    Write-Host "Updating Darktable plugin..."
    git -C $PluginDir pull
  } else {
    Write-Host "Installing Darktable plugin..."
    Remove-Item $PluginDir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path (Split-Path $PluginDir) | Out-Null
    git clone https://github.com/AyedaOk/DT_custom_script.git $PluginDir
  }
} else {
  Write-Host "Skipping Darktable plugin installation."
}

# ---------------------------------------------------------
# 8) Summary
# ---------------------------------------------------------
Write-Host ""
Write-Host "=== Installation complete ===" -ForegroundColor Green
Write-Host "Installed to:   $InstallDir"
Write-Host "Virtual env:    $VenvDir"
Write-Host "Config dir:     $ConfigDir"
Write-Host "Checkpoints:    $CheckpointDir"
Write-Host ""
Write-Host "Run with:"
Write-Host "  $InstallDir\launcher\sam2-tools.bat"
Write-Host "  $InstallDir\launcher\sam2-tools.exe"
Write-Host ""
