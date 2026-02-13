import os
import platform
from pathlib import Path
import numpy as np

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
# Box Selector (OpenCV drawing)
# ============================================================
class BoxSelector:
    def __init__(self, img, win_name=None):
        self.image_bgr = img
        self.clone = img.copy()
        self.start = None
        self.end = None
        self.drawing = False
        self.win_name = win_name

    def _line_thickness(self):
        if not self.win_name or not hasattr(cv2, "getWindowImageRect"):
            return 2

        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(self.win_name)
        except cv2.error:
            return 2

        if win_w <= 0 or win_h <= 0:
            return 2

        img_h, img_w = self.image_bgr.shape[:2]
        scale_x = img_w / win_w
        scale_y = img_h / win_h
        return max(2, int(np.ceil(max(scale_x, scale_y))))

    def reset(self):
        self.image_bgr[:] = self.clone
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
            self.image_bgr[:] = self.clone
            if self.start and self.end:
                cv2.rectangle(
                    self.image_bgr,
                    self.start,
                    self.end,
                    (0, 255, 0),
                    self._line_thickness(),
                )

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end = (x, y)
            self.image_bgr[:] = self.clone
            if self.start and self.end:
                cv2.rectangle(
                    self.image_bgr,
                    self.start,
                    self.end,
                    (0, 255, 0),
                    self._line_thickness(),
                )

    def get_box(self):
        if not self.start or not self.end:
            return None

        (x1, y1), (x2, y2) = self.start, self.end
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
