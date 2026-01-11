import os
import platform
from pathlib import Path
import numpy as np

# import yaml
import cv2
import rawpy
from PIL import Image


RAW_EXTENSIONS = {
    ".3fr",
    ".ari",
    ".arw",
    ".bay",
    ".cap",
    ".cr2",
    ".cr3",
    ".crw",
    ".dcr",
    ".dcs",
    ".dng",
    ".eip",
    ".erf",
    ".iiq",
    ".kdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".ptx",
    ".raf",
    ".r3d",
    ".rw2",
    ".rwl",
    ".rwz",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
}


# ============================================================
# Unique filename generator
# ============================================================
def get_unique_path(path):
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = path
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_path


# ============================================================
# Save PFM files
# ============================================================
def save_pfm(path, image, scale=1.0):
    image = np.flipud(image)

    if image.dtype != np.float32:
        image = image.astype(np.float32)

    color = image.ndim == 3 and image.shape[2] == 3

    with open(path, "wb") as f:
        f.write(b"PF\n" if color else b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())

        endian = -scale if image.dtype.byteorder in ("=", "little") else scale
        f.write(f"{endian}\n".encode())

        image.tofile(f)


# ============================================================
# Image loading
# ============================================================
def load_image_rgb(path):
    if not os.path.isfile(path):
        print("Input not found:", path)
        return None, None

    ext = Path(path).suffix.lower()
    try:
        if ext in RAW_EXTENSIONS:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess()
        else:
            rgb = np.array(Image.open(path).convert("RGB"))
    except Exception as exc:
        print("Failed to load image:", exc)
        return None, None

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb, bgr


# ============================================================
# Config handling
# ============================================================

# def get_config_path():
#     if platform.system().lower() == "windows":
#         base = Path(
#             os.getenv("APPDATA", Path.home() / "AppData" / "Roaming")
#         )
#         cfg_dir = base / "sam2"
#     else:
#         cfg_dir = Path.home() / ".config" / "sam2"
#
#     cfg_dir.mkdir(parents=True, exist_ok=True)
#     return cfg_dir / "config.yaml"
#
#
# def load_or_create_config():
#     cfg_path = get_config_path()
#
#     if not cfg_path.exists():
#         if platform.system().lower() == "windows":
#             # Correct Windows path
#             base = Path(os.getenv("APPDATA")) / "sam2" / "checkpoints"
#         else:
#             # Correct Linux/macOS path
#             base = Path.home() / ".config" / "sam2" / "checkpoints"
#
#         default = {
#             "checkpoints": {
#                 "1": str(base / "sam2.1_hiera_large.pt"),
#                 "2": str(base / "sam2.1_hiera_base_plus.pt"),
#                 "3": str(base / "sam2.1_hiera_small.pt"),
#                 "4": str(base / "sam2.1_hiera_tiny.pt"),
#             }
#         }
#
#         base.mkdir(parents=True, exist_ok=True)
#
#         with open(cfg_path, "w") as f:
#             yaml.dump(default, f)
#
#         print(f"Created default config: {cfg_path}")
#         return default
#
#     with open(cfg_path, "r") as f:
#         return yaml.safe_load(f)


# ============================================================
# Box Selector (OpenCV drawing)
# ============================================================
class BoxSelector:
    def __init__(self, img):
        self.image_bgr = img
        self.clone = img.copy()
        self.start = None
        self.end = None
        self.drawing = False

    def reset(self):
        self.image_bgr = self.clone.copy()
        self.start = None
        self.end = None
        self.drawing = False

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
            temp = self.clone.copy()
            cv2.rectangle(temp, self.start, self.end, (0, 255, 0), 2)
            self.image_bgr = temp

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end = (x, y)
            temp = self.clone.copy()
            cv2.rectangle(temp, self.start, self.end, (0, 255, 0), 2)
            self.image_bgr = temp

    def get_box(self):
        if not self.start or not self.end:
            return None

        (x1, y1), (x2, y2) = self.start, self.end
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
