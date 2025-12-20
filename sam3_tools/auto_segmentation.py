import os
import numpy as np
import torch
from PIL import Image

from transformers import pipeline
from .shared_utils import (
    get_unique_path,
    save_pfm,
)

def run_auto_segmentation(input_path, output_path, num_masks, pfm=False):
    save_dir = output_path
    base = os.path.splitext(os.path.basename(input_path))[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load input
    img = Image.open(input_path).convert("RGB")
    image_np = np.array(img)

    device_id = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
    generator = pipeline("mask-generation", model="facebook/sam3", device=device_id)

    outputs = generator(input_path, points_per_batch=64)  # OR pass PIL image
    masks = outputs["masks"]
    scores = outputs.get("scores")

    print("Generated masks:", len(masks))

    # Save masks
    for i, m in enumerate(masks[:num_masks]):
        if torch.is_tensor(m):
            m = m.detach().cpu().numpy()
        seg = np.squeeze(m).astype(np.uint8)

        if pfm:
            out = get_unique_path(f"{save_dir}/{base}_mask_{i}.pfm")
            save_pfm(out, seg.astype(np.float32))
        else:
            out = get_unique_path(f"{save_dir}/{base}_mask_{i}.png")
            Image.fromarray(seg * 255).save(out)







