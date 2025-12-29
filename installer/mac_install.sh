#!/usr/bin/env bash
set -e

echo "=== SAM3-Tools Installer (macOS / Apple Silicon) ==="

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
REPO_URL="https://github.com/AyedaOk/sam3-tools.git"
LUA_REPO_URL="https://github.com/AyedaOk/DT_custom_script.git"

INSTALL_DIR="${INSTALL_DIR:-$HOME/Applications/sam3-tools}"
VENV_DIR="$INSTALL_DIR/.venv"   # uv default
PYTHON_EXE="$VENV_DIR/bin/python"

PLUGIN_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/darktable/lua/Custom"

LAUNCHER_OUT="${LAUNCHER_OUT:-$HOME/Applications/sam3-tools.command}"

HF_MODEL_PAGE="https://huggingface.co/facebook/sam3"
HF_TOKEN_PAGE="https://huggingface.co/settings/tokens"

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
ok()   { printf "\033[1;32m%s\033[0m\n" "$1"; }
warn() { printf "\033[1;33m%s\033[0m\n" "$1"; }
err()  { printf "\033[1;31m%s\033[0m\n" "$1"; }

has_cmd() { command -v "$1" >/dev/null 2>&1; }

confirm() {
  # Usage: confirm "Question" "Y"|"N"
  local prompt="$1"
  local default="${2:-Y}"
  local reply=""

  if [[ "$default" == "Y" ]]; then
    if [[ -r /dev/tty ]]; then
      read -rp "$prompt [Y/n] " reply </dev/tty || true
    fi
    reply="${reply:-Y}"
  else
    if [[ -r /dev/tty ]]; then
      read -rp "$prompt [y/N] " reply </dev/tty || true
    fi
    reply="${reply:-N}"
  fi

  [[ "$reply" =~ ^[Yy]$ ]]
}

# ---------------------------------------------------------
# 1. Platform checks
# ---------------------------------------------------------
if [[ "$(uname -s)" != "Darwin" ]]; then
  err "This installer is for macOS only."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  err "Apple Silicon (arm64) only for now (matches the SAM2 macOS installer)."
  err "Intel Macs may require a different PyTorch setup."
  exit 1
fi

ok "Detected macOS (arm64)."

# ---------------------------------------------------------
# 2. Homebrew (required)
# ---------------------------------------------------------
BREW="$(command -v brew 2>/dev/null || true)"
if [[ -z "$BREW" && -x "/opt/homebrew/bin/brew" ]]; then
  BREW="/opt/homebrew/bin/brew"
fi

if [[ -z "$BREW" ]]; then
  warn "Homebrew not found (required)."
  if confirm "Install Homebrew now?" "Y"; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    BREW="/opt/homebrew/bin/brew"
  else
    err "Homebrew is required. Aborting."
    exit 1
  fi
fi

# Make sure brew is in PATH for this script run
eval "$("$BREW" shellenv)"
ok "Using Homebrew: $BREW"

# ---------------------------------------------------------
# 3. Install dependencies (git, uv)
# ---------------------------------------------------------
ok "Installing dependencies (git, uv)..."
brew update
brew install git uv

if ! has_cmd git; then
  err "git not found after brew install."
  exit 1
fi
if ! has_cmd uv; then
  err "uv not found after brew install."
  err "Try: brew doctor && brew reinstall uv"
  exit 1
fi
ok "Using uv: $(command -v uv)"

# ---------------------------------------------------------
# 4. Clone or update repo
# ---------------------------------------------------------
mkdir -p "$(dirname "$INSTALL_DIR")"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  ok "Repository exists — updating..."
  git -C "$INSTALL_DIR" pull
elif [[ -e "$INSTALL_DIR" ]]; then
  err "Install path exists but is not a git repo:"
  err "  $INSTALL_DIR"
  err "Move it aside or delete it, then re-run."
  exit 1
else
  ok "Cloning repository..."
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# ---------------------------------------------------------
# 5. Create virtual environment with uv
# ---------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  ok "Creating virtual environment with uv..."
  uv venv
else
  ok "Virtual environment already exists: $VENV_DIR"
fi

if [[ ! -x "$PYTHON_EXE" ]]; then
  err "uv environment python not found at: $PYTHON_EXE"
  err "Try deleting .venv and re-running the installer."
  exit 1
fi

# ---------------------------------------------------------
# 6. Install PyTorch then requirements
# ---------------------------------------------------------
ok "Installing PyTorch..."
uv pip install --python "$PYTHON_EXE" torch torchvision

ok "Installing Python dependencies from requirements.txt..."
uv pip install --python "$PYTHON_EXE" -r requirements.txt

# ---------------------------------------------------------
# 7. Hugging Face login (required for gated model)
# ---------------------------------------------------------
echo ""
warn "SAM3 checkpoints are gated on Hugging Face."
echo "Request access (wait for approval): $HF_MODEL_PAGE"
echo "Create token (Read):               $HF_TOKEN_PAGE"
echo ""

if confirm "Log in to Hugging Face now (recommended)?" "Y"; then
  ok "Starting Hugging Face login..."
  if uv run hf auth login </dev/tty; then
    ok "Hugging Face login complete."
  elif uv run huggingface-cli login </dev/tty; then
    ok "Hugging Face login complete (huggingface-cli)."
  else
    warn "HF login failed or 'hf' CLI not available; continuing."
    warn "You can retry later with: uv run hf auth login"
  fi
else
  warn "Skipping Hugging Face login. Model download will fail until you run: uv run hf auth login"
fi

# ---------------------------------------------------------
# 8. Optional: trigger first model download
# ---------------------------------------------------------
echo ""
if confirm "Download the SAM3 model now (~3.5GB)?" "Y"; then
  ok "Triggering model download (into your Hugging Face cache)..."
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
# 9. Generate Finder/Darktable launcher (.command)
# ---------------------------------------------------------
ok "Generating launcher: $LAUNCHER_OUT"
mkdir -p "$HOME/Applications"

if [[ -f "$LAUNCHER_OUT" ]]; then
  warn "Existing launcher found — backing up to: $LAUNCHER_OUT.bak"
  cp -f "$LAUNCHER_OUT" "$LAUNCHER_OUT.bak"
fi

cat > "$LAUNCHER_OUT" <<EOF
#!/bin/bash
set -e
APP_DIR="$INSTALL_DIR"
VENV_PY="\$APP_DIR/.venv/bin/python"

cd "\$APP_DIR"

if [ ! -x "\$VENV_PY" ]; then
  echo "Could not find venv python at: \$VENV_PY"
  echo "Re-run the installer to recreate the venv."
  exit 1
fi

exec "\$VENV_PY" main.py "\$@"
EOF

chmod +x "$LAUNCHER_OUT"

# ---------------------------------------------------------
# 10. Installing Darktable plugin (optional)
# ---------------------------------------------------------
echo ""
if confirm "Do you want to install Darktable plugin?" "Y"; then
  mkdir -p "$(dirname "$PLUGIN_DIR")"
  if [[ -d "$PLUGIN_DIR/.git" ]]; then
    ok "Plugin repo exists — updating..."
    git -C "$PLUGIN_DIR" pull
  else
    rm -rf "$PLUGIN_DIR"
    ok "Cloning plugin repo..."
    git clone "$LUA_REPO_URL" "$PLUGIN_DIR"
  fi
else
  warn "Skipping plugin installation."
fi

# ---------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------
echo ""
ok "=== Installation complete ==="
echo "Installed to:   $INSTALL_DIR"
echo "Virtual env:    $VENV_DIR"
echo "Finder launcher:$LAUNCHER_OUT"
echo "Plugin:         $PLUGIN_DIR"
echo ""
echo "Run with:"
echo "  \"$LAUNCHER_OUT\""
echo "or:"
echo "  cd \"$INSTALL_DIR\" && uv run python main.py"
echo "or (if system launcher installed):"
echo "  sam3-tools"
echo ""
warn "If builds fail (missing compilers/headers), install Xcode CLT with: xcode-select --install (then re-run)."
