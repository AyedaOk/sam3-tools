#!/usr/bin/env bash
set -e

echo "=== SAM3-Tools Installer ==="

# ---------------------------------------------------------
# Paths / URLs
# ---------------------------------------------------------
REPO_URL="https://github.com/AyedaOk/sam3-tools.git"
LUA_REPO_URL="https://github.com/AyedaOk/DT_custom_script.git"
INSTALL_DIR="$HOME/.local/opt/sam3-tools"
VENV_DIR="$INSTALL_DIR/.venv"                # uv default
LAUNCHER_PATH="/usr/local/bin/sam3-tools"
TMPDIR_PATH="$HOME/.cache/sam3-tools/tmp"
PLUGIN_DIR="$HOME/.config/darktable/lua/Custom"

HF_MODEL_PAGE="https://huggingface.co/facebook/sam3"
HF_TOKEN_PAGE="https://huggingface.co/settings/tokens"

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
ok()   { printf "\033[1;32m%s\033[0m\n" "$1"; }
warn() { printf "\033[1;33m%s\033[0m\n" "$1"; }
err()  { printf "\033[1;31m%s\033[0m\n" "$1"; }

has_cmd() { command -v "$1" >/dev/null 2>&1; }

download_to_stdout() {
  if has_cmd curl; then
    curl -fsSL "$1"
  elif has_cmd wget; then
    wget -qO- "$1"
  else
    return 1
  fi
}

# ---------------------------------------------------------
# 1) Detect distro family
# ---------------------------------------------------------
if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  source /etc/os-release
else
  err "Cannot read /etc/os-release. Unsupported system."
  exit 1
fi

detect_family() {
  case "${ID:-}" in
    arch) echo "arch"; return ;;
    ubuntu|debian|linuxmint|pop|zorin|elementary|neon) echo "debian"; return ;;
    fedora|rhel|rocky|almalinux|centos) echo "fedora"; return ;;
  esac

  if [[ "${ID_LIKE:-}" == *"debian"* ]] || [[ "${ID_LIKE:-}" == *"ubuntu"* ]]; then
    echo "debian"; return
  fi
  if [[ "${ID_LIKE:-}" == *"fedora"* ]] || [[ "${ID_LIKE:-}" == *"rhel"* ]]; then
    echo "fedora"; return
  fi
  if [[ "${ID_LIKE:-}" == *"arch"* ]]; then
    echo "arch"; return
  fi

  if has_cmd pacman; then echo "arch"; return; fi
  if has_cmd apt; then echo "debian"; return; fi
  if has_cmd dnf; then echo "fedora"; return; fi

  echo "unknown"
}

FAMILY="$(detect_family)"
ok "Detected distro family: $FAMILY"
if [[ "$FAMILY" == "unknown" ]]; then
  err "Unsupported Linux distribution."
  exit 1
fi

# ---------------------------------------------------------
# 2) Ensure system dependencies
# ---------------------------------------------------------
missing=false

# core tools
has_cmd git     || missing=true
has_cmd python3 || missing=true
has_cmd curl || has_cmd wget || missing=true

# Tk (GUI)
case "$FAMILY" in
  debian)
    dpkg -l 2>/dev/null | grep -q "python3-tk" || missing=true
    ;;
  arch)
    pacman -Q tk >/dev/null 2>&1 || missing=true
    ;;
  fedora)
    rpm -qa 2>/dev/null | grep -q "python3-tkinter" || missing=true
    ;;
esac

if $missing; then
  warn "Some dependencies are missing, installing them..."
  case "$FAMILY" in
    debian)
      sudo apt update
      sudo apt install -y python3 python3-tk git curl
      ;;
    arch)
      sudo pacman -Syu --noconfirm python tk git curl
      ;;
    fedora)
      sudo dnf install -y python3 git curl python3-tkinter gcc gcc-c++ make python-devel
      ;;
  esac
else
  ok "All system dependencies already installed."
fi

# ---------------------------------------------------------
# 3) Install uv if missing + ensure PATH
# ---------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"

if ! has_cmd uv; then
  ok "Installing uv..."
  if download_to_stdout "https://astral.sh/uv/install.sh" | sh; then
    ok "uv installed."
  else
    err "Failed to install uv (need curl or wget)."
    exit 1
  fi
fi

export PATH="$HOME/.local/bin:$PATH"
if ! has_cmd uv; then
  err "uv is not on PATH. Try restarting your shell or adding ~/.local/bin to PATH."
  exit 1
fi

# ---------------------------------------------------------
# 4) Clone or update repo
# ---------------------------------------------------------
mkdir -p "$(dirname "$INSTALL_DIR")"
if [ -d "$INSTALL_DIR/.git" ]; then
  ok "Repository exists — updating..."
  git -C "$INSTALL_DIR" pull
else
  ok "Cloning repository..."
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# ---------------------------------------------------------
# 5) Create uv virtual environment
# ---------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  ok "Creating virtual environment with uv..."
  uv venv
else
  ok "Virtual environment already exists: $VENV_DIR"
fi

# ---------------------------------------------------------
# 6) Install Python dependencies
# ---------------------------------------------------------
ok "Installing Python dependencies..."
rm -rf "$TMPDIR_PATH"
mkdir -p "$TMPDIR_PATH"
export TMPDIR="$TMPDIR_PATH"

# CPU-only?
read -rp "Install CPU-only dependencies (no NVIDIA GPU)? [y/N] " CPU_REPLY </dev/tty
CPU_REPLY="${CPU_REPLY:-N}"

if [[ "$CPU_REPLY" =~ ^[Yy]$ ]]; then
  ok "Installing CPU-only requirements..."
  uv pip install -r requirements-cpu.txt
else
  ok "Installing default requirements..."
  uv pip install -r requirements.txt

  # CUDA 13 detection (best-effort)
  CUDA_VER=""
  if has_cmd nvidia-smi; then
    CUDA_VER="$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9]\+" | head -n1 | awk '{print $3}' || true)"
  fi
  DEFAULT_CU130="N"
  if [[ "$CUDA_VER" == 13* ]]; then
    DEFAULT_CU130="Y"
    ok "Detected NVIDIA CUDA Version: $CUDA_VER (from nvidia-smi)"
  fi

  read -rp "Install/upgrade PyTorch nightly for CUDA 13 (cu130)? [${DEFAULT_CU130}/n] " CU130_REPLY </dev/tty
  CU130_REPLY="${CU130_REPLY:-$DEFAULT_CU130}"

  if [[ "$CU130_REPLY" =~ ^[Yy]$ ]]; then
    ok "Upgrading torch + torchvision from cu130 nightly index..."
    uv pip install --pre --upgrade \
      --index-url https://download.pytorch.org/whl/nightly/cu130 \
      --extra-index-url https://pypi.org/simple \
      torch torchvision
  else
    ok "Skipping cu130 torch upgrade."
  fi
fi

rm -rf "$TMPDIR_PATH"

# ---------------------------------------------------------
# 7) Hugging Face login (required for gated model)
# ---------------------------------------------------------
echo ""
warn "SAM3 checkpoints are gated on Hugging Face."
echo "Request access (wait for approval): $HF_MODEL_PAGE"
echo "Create token (Read):               $HF_TOKEN_PAGE"
echo ""

read -rp "Log in to Hugging Face now (recommended)? [Y/n] " HF_REPLY </dev/tty
HF_REPLY="${HF_REPLY:-Y}"

if [[ "$HF_REPLY" =~ ^[Yy]$ ]]; then
  ok "Starting Hugging Face login..."
  uv run hf auth login </dev/tty || warn "HF login returned non-zero; continuing..."
else
  warn "Skipping Hugging Face login. Model download will fail until you run: uv run hf auth login"
fi

# ---------------------------------------------------------
# 8) Optional: trigger first model download
# ---------------------------------------------------------
echo ""
read -rp "Download the SAM3 model now (~3Gb)? [Y/n] " DL_REPLY </dev/tty
DL_REPLY="${DL_REPLY:-Y}"

if [[ "$DL_REPLY" =~ ^[Yy]$ ]]; then
  ok "Triggering model download..."
  uv run python - <<'PY'
from transformers import Sam3Model, Sam3Processor
m = Sam3Model.from_pretrained("facebook/sam3")
p = Sam3Processor.from_pretrained("facebook/sam3")
print("Downloaded model + processor OK.")
PY
else
  warn "Skipping model download."
fi

# ---------------------------------------------------------
# 9) Optional: system-wide launcher
# ---------------------------------------------------------
echo ""
read -rp "Install system-wide launcher to ${LAUNCHER_PATH}? [Y/n] " LAUNCH_REPLY </dev/tty
LAUNCH_REPLY="${LAUNCH_REPLY:-Y}"

if [[ "$LAUNCH_REPLY" =~ ^[Yy]$ ]]; then
  ok "Generating launcher..."
  sudo bash -c "cat > '$LAUNCHER_PATH'" <<EOF
#!/bin/bash
APP_DIR="$INSTALL_DIR"
cd "\$APP_DIR" || exit 1
"\$APP_DIR/.venv/bin/python" main.py "\$@"
EOF
  sudo chmod +x "$LAUNCHER_PATH"
  warn "Testing launcher..."
  sam3-tools --help || warn "Launcher test failed — but installation may still be OK."
else
  warn "Skipping launcher installation."
fi

# ---------------------------------------------------------
# 10) Optional: Darktable plugin
# ---------------------------------------------------------
echo ""
read -rp "Do you want to install the Darktable plugin? [Y/n] " PLUG_REPLY </dev/tty
PLUG_REPLY="${PLUG_REPLY:-Y}"

if [[ "$PLUG_REPLY" =~ ^[Yy]$ ]]; then
  ok "Installing Darktable plugin..."
  rm -rf "$PLUGIN_DIR"
  git clone "$LUA_REPO_URL" "$PLUGIN_DIR"
else
  warn "Skipping plugin installation."
fi

# ---------------------------------------------------------
# 11) Summary
# ---------------------------------------------------------
echo ""
ok "=== Installation complete ==="
echo "Installed to:   $INSTALL_DIR"
echo "Virtual env:    $VENV_DIR"
echo "Launcher:       $LAUNCHER_PATH"
echo "Plugin:         $PLUGIN_DIR"
echo ""
echo "Run with:"
echo "  cd \"$INSTALL_DIR\" && uv run python main.py"
echo "or (if launcher installed):"
echo "  sam3-tools"
echo ""
