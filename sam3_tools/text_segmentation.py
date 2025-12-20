from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import numpy as np
import os

from .shared_utils import (
    save_pfm,
)

def run_text_segmentation(input_path, output_path, prompt, num_masks, pfm=False):

    output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)

    # Load the image from path (not URL)
    image = Image.open(input_path).convert("RGB")

    # Device + models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # Prepare inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks = results["masks"]
    scores = results["scores"]

    print(f"Found {len(masks)} objects")

    if len(masks) == 0:
        print("No masks found.")
        return

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    count = min(num_masks, len(masks))

    # Save masks
    for i in range(count):
        if not pfm:
            # Save to PNG
            mask = masks[i].cpu().numpy().astype(np.uint8) * 255
            out_path = f"{output_dir}/{base_name}_mask_{i}.png"
            Image.fromarray(mask).save(out_path)
            print(f"Saved mask {i} (score={scores[i]:.4f}) → {out_path}")
        else:
            # Save to PFM
            m = masks[i].cpu().numpy()
            seg = np.squeeze(m).astype(np.float32)  # float32 mask for PFM
            out_path = f"{output_dir}/{base_name}_mask_{i}.pfm"
            save_pfm(out_path, seg)
            print(f"Saved mask {i} (score={scores[i]:.4f}) → {out_path}")

if __name__ == "__main__":
    main()
