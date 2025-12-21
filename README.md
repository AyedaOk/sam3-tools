# SAM3‑Tools
SAM3‑Tools is a lightweight Python application offering both a simple GUI and command‑line interface for running Meta AI’s Segment Anything 3 (SAM3) model. It supports box selection, auto segmentation, point‑based segmentation and text prompt segmentation.

---

## Features
- CLI interface for integration with Darktable  
- Basic GUI interface  
- Works with Darktable via PFM output  
- Works with the Darktable SAM3 plugin – [GitHub repo](https://github.com/AyedaOk/DT_custom_script)
- Segmentation modes: Prompt, Box, Auto, Points  
- Cross‑platform: Linux, macOS, Windows  

---

## Install

### Linux Installation Steps:

Install the following first:

* Arch: `sudo pacman -S tk git`

* Debian/Ubuntu: `sudo apt install python3-tk git`

* Fedora: `sudo dnf install -y git python3-tkinter gcc gcc-c++ make python-devel` (À tester)

Install UV:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To add $HOME/.local/bin to your PATH, either restart your shell or run:

```
source $HOME/.local/bin/env 
```

Or this for fish:

```
source $HOME/.local/bin/env.fish 
```

Create the installation directory:

```
mkdir -p $HOME/.local/opt/
```

Clone the repo, create the virtual environment and install the Python App:

```
cd $HOME/.local/opt/
git clone https://github.com/AyedaOk/sam3-tools.git
cd sam3-tools
uv venv
```

Install application:

If your GPU is running CUDA 13, run this command first. You can check your version by running `nvidia-smi`:

```
uv pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu130 torch
```

If you don't have a GPU and want to install it for CPU only:

```
uv pip install -r requirements-cpu.txt
```

Otherwise, run this command:

```
uv pip install -r requirements.txt
```

SAM3 checkpoints are gated on Hugging Face, so you must request access and log in before first run.

Request access here (wait for approval):
[https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)

Create a Hugging Face access token (token type should be Read):
[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Log in from your terminal using the access token:

```
uv run hf auth login
```

First run to download the model files into the HF cache (`~/.cache/huggingface/`).

```
ls -lah ./example
uv run main.py -i ./example/1.jpg -o ./example --text "sky" 
ls -lah ./example
```

#### Optional: System-wide launcher (required for Darktable integration)

To install like a system-wide “app”:

Install the launcher

```
sudo cp ./launcher/sam3-tools /usr/local/bin/sam3-tools
sudo chmod +x /usr/local/bin/sam3-tools
```

#### Optional: Darktable integration

Install the darktable plugin:
```
rm -rf $HOME/.config/darktable/lua/Custom
git clone https://github.com/AyedaOk/DT_custom_script.git $HOME/.config/darktable/lua/Custom
```

