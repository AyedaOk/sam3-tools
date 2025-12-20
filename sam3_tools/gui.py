import tkinter as tk
import os
from tkinter import filedialog, ttk
import threading
from .auto_segmentation import run_auto_segmentation
from .box_segmentation import run_box_segmentation
from .point_segmentation import run_point_segmentation
from .shared_utils import BoxSelector

def start_gui():
    root = tk.Tk()
    root.title("SAM2 Segmentation Tool")

    tk.Label(root, text="Input image:").grid(row=0, column=0)
    input_var = tk.StringVar(value=os.path.expanduser("~"))
    tk.Entry(root, textvariable=input_var, width=40).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=lambda: input_var.set(
        filedialog.askopenfilename(initialdir=os.path.expanduser("~"))
    )).grid(row=0, column=2)

    tk.Label(root, text="Output folder:").grid(row=1, column=0)
    output_var = tk.StringVar(value=os.path.expanduser("~"))
    tk.Entry(root, textvariable=output_var, width=40).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=lambda: output_var.set(
        filedialog.askdirectory(initialdir=os.path.expanduser("~"))
    )).grid(row=1, column=2)

    tk.Label(root, text="Model:").grid(row=2, column=0)
    model_var = tk.StringVar(value="1")
    ttk.Combobox(root, textvariable=model_var, values=["1", "2", "3", "4"]).grid(row=2, column=1)

    # Mode moved here (between model and num masks)
    tk.Label(root, text="Mode:").grid(row=3, column=0)
    mode_var = tk.StringVar(value="Box")
    ttk.Combobox(root, textvariable=mode_var, values=["Box", "Auto", "Points"]).grid(row=3, column=1)

    tk.Label(root, text="Num Masks:").grid(row=4, column=0)
    num_masks_var = tk.IntVar(value=1)
    tk.Spinbox(root, from_=1, to=10, textvariable=num_masks_var).grid(row=4, column=1)

    pfm_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Save as PFM", variable=pfm_var).grid(row=5, column=0)

    overlay_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Overlay", variable=overlay_var).grid(row=5, column=1)

    def run_clicked():
        inp = input_var.get()
        out = output_var.get()
        model = int(model_var.get())
        n = num_masks_var.get()
        mode = mode_var.get()
        save_pfm = pfm_var.get()
        overlay = overlay_var.get()

        if mode == "Points":
            threading.Thread(target=lambda: run_point_segmentation(inp, out, n, model, save_pfm)).start()
        elif mode == "Auto":
            threading.Thread(target=lambda: run_auto_segmentation(inp, out, n, model, save_pfm)).start()
        else:
            threading.Thread(target=lambda: run_box_segmentation(inp, out, n, model, None, save_pfm, overlay)).start()

    tk.Button(root, text="Run", command=run_clicked).grid(row=6, column=1)



    root.mainloop()
