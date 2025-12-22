import os
import numpy as np
import cv2
import torch
from PIL import Image

from transformers import Sam3TrackerProcessor, Sam3TrackerModel

from .shared_utils import get_unique_path, save_pfm, BoxSelector


def run_box_segmentation(input_path, output_path, num_masks=1, box=None, pfm=False):

    if not input_path or not os.path.exists(input_path):
        print("Input not found:", input_path)
        return
    if not output_path:
        print("Output path is required.")
        return

    os.makedirs(output_path, exist_ok=True)
    save_dir = output_path
    base = os.path.splitext(os.path.basename(input_path))[0]

    # Load image for box selection
    bgr_img = cv2.imread(input_path)
    if bgr_img is None:
        print("Failed to load image:", input_path)
        return
    H, W = bgr_img.shape[:2]

    # Get user box if not provided
    if box is None:
        print("Draw selection box...")
        win = "Box Selection (Enter=OK, R=reset, Esc=cancel)"
        selector = BoxSelector(bgr_img.copy())

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, selector.mouse_cb)

        while True:
            cv2.imshow(win, selector.image_bgr)
            key = cv2.waitKey(20) & 0xFF
            if key == 13:  # Enter
                b = selector.get_box()
                if b:
                    box = b
                    break
            elif key in (ord("r"), ord("R")):
                selector.reset()
            elif key == 27:  # Esc
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

    # Normalize + clip box
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))

    if x2 <= x1 or y2 <= y1:
        print("Invalid box:", (x1, y1, x2, y2))
        return

    # Build a PIL image from the same pixels used for box selection
    raw_image = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))

    # Load SAM3 tracker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_name = "facebook/sam3"
    model = Sam3TrackerModel.from_pretrained(model_name).to(device)
    processor = Sam3TrackerProcessor.from_pretrained(model_name)

    input_boxes = [[[x1, y1, x2, y2]]]

    inputs = processor(
        images=raw_image,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    # Post-process to original image size
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )[0]  # typically [num_objects, num_masks, H, W] :contentReference[oaicite:1]{index=1}

    if masks is None or masks.numel() == 0:
        print("No masks returned.")
        return

    # Rank by iou_scores if available
    iou = getattr(outputs, "iou_scores", None)
    if iou is not None:
        iou = iou.detach().cpu()
        if iou.ndim >= 3:
            iou_vec = iou[0, 0]
        elif iou.ndim == 2:
            iou_vec = iou[0]
        else:
            iou_vec = iou
        order = torch.argsort(iou_vec, descending=True).tolist()
    else:
        # Fallback: keep default order
        order = list(range(masks.shape[1])) if masks.ndim >= 4 else list(range(masks.shape[0]))

    # Save up to num_masks
    count = min(int(num_masks), len(order))
    for rank, idx in enumerate(order[:count]):
        if masks.ndim == 4:
            m = masks[0, idx]  # object 0, candidate idx
        else:
            m = masks[idx]

        if torch.is_tensor(m):
            m = m.cpu().numpy()

        seg = np.squeeze(m).astype(np.uint8)  # 0/1

        if pfm:
            out = get_unique_path(f"{save_dir}/{base}_mask_{rank}.pfm")
            save_pfm(out, seg.astype(np.float32))
        else:
            out = get_unique_path(f"{save_dir}/{base}_mask_{rank}.png")
            Image.fromarray(seg * 255).save(out)

        print("Saved:", out)
