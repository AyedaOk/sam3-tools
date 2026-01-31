import os
import numpy as np
import cv2
import torch
from PIL import Image
from datetime import datetime, timezone

from transformers import Sam3TrackerProcessor, Sam3TrackerModel
from accelerate import Accelerator
from .shared_utils import (
    get_unique_path,
    save_pfm,
    load_image_rgb,
)


# ============================================================
# Point Selector (interactive point mode)
# ============================================================
class PointSelector:
    def __init__(self, img_bgr, model, processor, raw_image):
        self.clone = img_bgr.copy()
        self.image_bgr = img_bgr.copy()

        self.model = model
        self.processor = processor
        self.raw_image = raw_image
        self.points_pos = []  # left-click = foreground
        self.points_neg = []  # right-click = background

        self.current_mask = None
        self.rgb_for_predictor = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def reset(self):
        self.image_bgr = self.clone.copy()
        self.points_pos.clear()
        self.points_neg.clear()
        self.current_mask = None

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Foreground click
            self.points_pos.append((x, y))
            self.update_mask()

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.points_neg.append((x, y))
            self.update_mask()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points_neg.append((x, y))
            self.update_mask()

    # ------------------------------------------------------------------

    def update_mask(self):
        # No points → no mask
        if not self.points_pos and not self.points_neg:
            self.current_mask = None
            self.render_preview()
            return

        all_pts = [list(p) for p in (self.points_pos + self.points_neg)]
        labels = [1] * len(self.points_pos) + [0] * len(self.points_neg)

        input_points = [[all_pts]]
        input_labels = [[labels]]

        inputs = self.processor(
            images=self.raw_image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        # masks: [num_objects, num_masks, H, W] for the first (and only) image
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]

        # Pick best mask by IOU score if available; otherwise take mask 0
        best_idx = 0
        iou = getattr(outputs, "iou_scores", None)
        if iou is not None:
            iou = iou.detach().cpu()
            # typically [batch, objects, num_masks]
            if iou.ndim >= 3:
                iou_vec = iou[0, 0]
            elif iou.ndim == 2:
                iou_vec = iou[0]
            else:
                iou_vec = iou
            best_idx = int(torch.argmax(iou_vec).item())

        best_mask = masks[0, best_idx]  # object 0, best candidate
        if torch.is_tensor(best_mask):
            best_mask = best_mask.cpu().numpy()

        self.current_mask = best_mask
        self.render_preview()

    # ------------------------------------------------------------------
    def render_preview(self):
        img = self.clone.copy()

        # Overlay mask
        if self.current_mask is not None:
            mask = (self.current_mask > 0).astype(np.uint8)
            # Red overlay for mask preview
            img[mask > 0] = (0, 0, 255)

        # Draw points
        for x, y in self.points_pos:
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # green = FG

        for x, y in self.points_neg:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # red = BG

        self.image_bgr = img


# ============================================================
# RUN POINT SEGMENTATION
# ============================================================
def run_point_segmentation(
    input_path,
    output_path,
    num_masks=1,
    pfm=False,
):
    # Prepare output directories
    if not os.path.exists(input_path):
        print("Input not found:", input_path)
        return

    os.makedirs(output_path, exist_ok=True)

    save_dir = output_path
    base = os.path.splitext(os.path.basename(input_path))[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    device = Accelerator().device
    model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

    # Load image
    rgb, bgr_img = load_image_rgb(input_path)
    if bgr_img is None:
        return
    raw_image = Image.fromarray(rgb)

    # Create selector interface
    win = "Left Click=Positive, Right/Middle Click=Negative, Enter=Confirm, R=Reset, Esc=Cancel"
    selector = PointSelector(bgr_img, model, processor, raw_image)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, selector.mouse_cb)

    final_mask = None

    while True:
        cv2.imshow(win, selector.image_bgr)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # ENTER
            final_mask = selector.current_mask
            break

        elif key in (ord("r"), ord("R")):
            selector.reset()

        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    if final_mask is None:
        print("No mask generated.")
        return

    # Save final mask
    mask = final_mask.squeeze().astype(np.uint8) * 255

    if pfm:
        out = get_unique_path(f"{save_dir}/{base}_{ts}_mask.pfm")
        save_pfm(out, final_mask.squeeze())  # PFM uses float mask, not 0–255
    else:
        out = get_unique_path(f"{save_dir}/{base}_{ts}_mask.png")
        Image.fromarray(mask).save(out)

    print("Saved:", out)
