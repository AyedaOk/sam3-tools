import os
import numpy as np
import cv2  
import torch
from PIL import Image
from datetime import datetime, timezone
from transformers import Sam3Processor, Sam3Model

from .shared_utils import (
    save_pfm,
    load_image_rgb,
)


def run_text_segmentation(input_path, output_path, prompt, num_masks, pfm=False):
    output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)

    rgb, bgr = load_image_rgb(input_path)
    if rgb is None or bgr is None:
        print("Erreur de chargement de l'image.")
        return
    image = Image.fromarray(rgb)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # ============================================================
    # NOUVEAU : GESTION DES VIRGULES (MULTI-PROMPTS)
    # ============================================================
    # On découpe la phrase à chaque virgule et on enlève les espaces
    sub_prompts = [p.strip() for p in prompt.split(",") if p.strip()]
    
    # On crée une toile noire vide de la taille de ton image
    combined_mask = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=bool)
    
    print(f"\nRecherche en cours pour les mots : {sub_prompts} ...")

    # On boucle sur chaque mot séparément
    with torch.no_grad():
        for sp in sub_prompts:
            inputs = processor(images=image, text=sp, return_tensors="pt").to(device)
            outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.25,
                mask_threshold=0.45,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks = results["masks"]
            
            if len(masks) > 0:
                print(f" -> {len(masks)} objets trouvés pour '{sp}'")
                # On additionne mathématiquement tous les masques trouvés sur notre toile
                for i in range(len(masks)):
                    m = masks[i].cpu().numpy().squeeze()
                    combined_mask = np.logical_or(combined_mask, m > 0.5)
            else:
                print(f" -> Rien trouvé pour '{sp}'")

    # Si la toile est toujours 100% noire, on arrête
    if not np.any(combined_mask):
        print("Aucun masque généré au total.")
        return

    # ============================================================
    # L'APERÇU VISUEL GLOBAL
    # ============================================================
    preview_img = bgr.copy()
    preview_img[combined_mask] = (0, 0, 255) 

    win_name = f"Multi-Texte: {sub_prompts} | Entree = Valider, Echap = Annuler"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    h, w = bgr.shape[:2]
    scale = 800.0 / max(h, w)
    if scale < 1.0:
        cv2.resizeWindow(win_name, int(w * scale), int(h * scale))

    cv2.imshow(win_name, preview_img)

    save_approved = False
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Touche Entrée
            save_approved = True
            break
        elif key == 27:  # Touche Échap
            save_approved = False
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)  

    # ============================================================
    # SAUVEGARDE DU MASQUE FUSIONNÉ UNIQUE
    # ============================================================
    if not save_approved:
        print(f"Action annulée. Le masque combiné n'a pas été sauvegardé.")
        return

    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    custom_name = simpledialog.askstring("Nom du masque", f"Entrez un TAG pour le masque global\n(ex: sujet_complet) :", parent=root)
    root.destroy()

    custom_name = custom_name.strip() if custom_name else ""

    if custom_name:
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        tag_str = f"_{custom_name}"
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag_str = "_merged"

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # On exporte l'image fusionnée en UN SEUL fichier
    final_mask_export = combined_mask.astype(np.uint8) * 255

    if not pfm:
        out_path = f"{output_dir}/{base_name}{tag_str}_{ts}_mask.png"
        Image.fromarray(final_mask_export).save(out_path)
        print(f"Sauvegardé (Fusionné) : {out_path}")
    else:
        seg = final_mask_export.astype(np.float32) 
        out_path = f"{output_dir}/{base_name}{tag_str}_{ts}_mask.pfm"
        save_pfm(out_path, seg)
        print(f"Sauvegardé (Fusionné) : {out_path}")