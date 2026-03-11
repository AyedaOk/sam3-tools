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
    def __init__(self, img_bgr, model, processor, raw_image, box=None):
        self.clone = img_bgr.copy()
        self.image_bgr = img_bgr.copy()

        self.model = model
        self.processor = processor
        self.raw_image = raw_image
        self.points_pos = []  
        self.points_neg = []  
        
        self.box = box 

        self.current_mask = None
        self.rgb_for_predictor = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        self.mask_idx = -1
        
        self.render_preview()

    def reset(self):
        self.image_bgr = self.clone.copy()
        self.points_pos.clear()
        self.points_neg.clear()
        self.current_mask = None
        self.mask_idx = -1
        self.render_preview()

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
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
        try:
            if not self.points_pos and not self.points_neg:
                self.current_mask = None
                self.render_preview()
                return

            inputs_kwargs = {
                "images": self.raw_image,
                "return_tensors": "pt"
            }

            all_pts = [list(p) for p in (self.points_pos + self.points_neg)]
            labels = [1] * len(self.points_pos) + [0] * len(self.points_neg)
            inputs_kwargs["input_points"] = [[all_pts]]
            inputs_kwargs["input_labels"] = [[labels]]
                
            if self.box is not None:
                inputs_kwargs["input_boxes"] = [[self.box]]

            inputs = self.processor(**inputs_kwargs).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model(**inputs)

            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
                binarize=False 
            )[0]

            best_idx = 0
            iou = getattr(outputs, "iou_scores", None)
            num_candidates = masks.shape[1]
            
            if iou is not None:
                iou = iou.detach().cpu()
                if iou.ndim >= 3:
                    iou_vec = iou[0, 0]
                elif iou.ndim == 2:
                    iou_vec = iou[0]
                else:
                    iou_vec = iou
                    
                if self.mask_idx == -1:
                    best_idx = int(torch.argmax(iou_vec).item())
                else:
                    best_idx = self.mask_idx % num_candidates
            else:
                if self.mask_idx != -1:
                    best_idx = self.mask_idx % num_candidates

            best_mask = masks[0, best_idx]  
            if torch.is_tensor(best_mask):
                best_mask = best_mask.cpu().numpy()

            macro_threshold = 1.5
            best_mask = (best_mask > macro_threshold).astype(np.uint8)

            if self.box is not None:
                xmin, ymin, xmax, ymax = self.box
                exclusion_mask = np.zeros_like(best_mask)
                exclusion_mask[ymin:ymax, xmin:xmax] = 1
                best_mask = best_mask * exclusion_mask

            self.current_mask = best_mask
            self.render_preview()
            
        except Exception as e:
            print(f"\n[ERREUR IA] Le calcul a échoué : {e}\n")

    # ------------------------------------------------------------------
    def render_preview(self):
        img = self.clone.copy()

        if self.box is not None:
            xmin, ymin, xmax, ymax = self.box
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 100, 0), 2)

        if self.current_mask is not None:
            mask = (self.current_mask > 0).astype(np.uint8)
            img[mask > 0] = (0, 0, 255)

        for x, y in self.points_pos:
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1) 

        for x, y in self.points_neg:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1) 

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
    if not os.path.exists(input_path):
        print("Input not found:", input_path)
        return

    os.makedirs(output_path, exist_ok=True)
    save_dir = output_path
    base = os.path.splitext(os.path.basename(input_path))[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = Accelerator().device
    model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

    rgb, bgr_img = load_image_rgb(input_path)
    if bgr_img is None:
        return
    raw_image = Image.fromarray(rgb)

    roi_win = "ETAPE 1 : Dessinez un cadre (Espace = Valider, Echap = Ignorer)"
    cv2.namedWindow(roi_win, cv2.WINDOW_NORMAL)
    
    h, w = bgr_img.shape[:2]
    scale = 800.0 / max(h, w)
    if scale < 1.0:
        cv2.resizeWindow(roi_win, int(w * scale), int(h * scale))

    print("Etape 1 : Dessinez le cadre d'exclusion (ou appuyez sur Espace sans dessiner pour passer)")
    roi = cv2.selectROI(roi_win, bgr_img, showCrosshair=True, fromCenter=False)
    
    cv2.destroyWindow(roi_win)
    cv2.waitKey(10) 

    box = None
    if roi[2] > 0 and roi[3] > 0:
        box = [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]]
        print(f"Cadre d'exclusion verrouillé : {box}")

    win = "ETAPE 2 : Clics = Points, TAB = Variations, R = Reset, Enter = Confirmer"
    selector = PointSelector(bgr_img, model, processor, raw_image, box)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    
    if scale < 1.0:
        cv2.resizeWindow(win, int(w * scale), int(h * scale))
        
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

        elif key == 9:  # TAB
            selector.mask_idx += 1
            print(f"Cycle mask: Variation {selector.mask_idx % 3}")
            selector.update_mask()

        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Hack Windows pour vider la mémoire vidéo proprement
    
    if final_mask is None:
        print("No mask generated.")
        return

    # ============================================================
    # NOUVEAU : POP-UP VISUEL POUR LE NOM
    # ============================================================
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw() # Cache la fenêtre principale
    root.attributes("-topmost", True) # Force le pop-up au premier plan
    
    custom_name = simpledialog.askstring("Nom du masque", "Entrez un TAG (ex: arbre, peau)\nou laissez vide et cliquez sur OK :", parent=root)
    root.destroy()

    custom_name = custom_name.strip() if custom_name else ""

    # ============================================================
    # NOUVELLE LOGIQUE DE NOMMAGE INTELLIGENT
    # ============================================================
    if custom_name:
        # Si tu as mis un Tag : on garde juste HeureMinuteSeconde
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        tag_str = f"_{custom_name}"
    else:
        # Si tu n'as rien mis : on met la Date et l'Heure complète
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag_str = ""

    mask = final_mask.squeeze().astype(np.uint8) * 255

    if pfm:
        out = get_unique_path(f"{save_dir}/{base}{tag_str}_{ts}_mask.pfm")
        save_pfm(out, final_mask.squeeze())  
    else:
        out = get_unique_path(f"{save_dir}/{base}{tag_str}_{ts}_mask.png")
        Image.fromarray(mask).save(out)

    print(f"Sauvegardé : {out}")