import tkinter as tk
import os
import threading
import inspect
from tkinter import filedialog, ttk, messagebox

from .auto_segmentation import run_auto_segmentation
from .box_segmentation import run_box_segmentation
from .point_segmentation import run_point_segmentation
from .text_segmentation import run_text_segmentation  # NEW


def start_gui():
    root = tk.Tk()
    root.title("SAM3 Segmentation Tool")

    # --- Inputs ---
    tk.Label(root, text="Input image:").grid(row=0, column=0, sticky="w")
    input_var = tk.StringVar(value=os.path.expanduser("~"))
    tk.Entry(root, textvariable=input_var, width=40).grid(
        row=0, column=1, padx=4, pady=2
    )
    tk.Button(
        root,
        text="Browse",
        command=lambda: input_var.set(
            filedialog.askopenfilename(initialdir=os.path.expanduser("~"))
        ),
    ).grid(row=0, column=2, padx=4, pady=2)

    tk.Label(root, text="Output folder:").grid(row=1, column=0, sticky="w")
    output_var = tk.StringVar(value=os.path.expanduser("~"))
    tk.Entry(root, textvariable=output_var, width=40).grid(
        row=1, column=1, padx=4, pady=2
    )
    tk.Button(
        root,
        text="Browse",
        command=lambda: output_var.set(
            filedialog.askdirectory(initialdir=os.path.expanduser("~"))
        ),
    ).grid(row=1, column=2, padx=4, pady=2)

    # --- Mode (now includes Text) ---
    tk.Label(root, text="Mode:").grid(row=2, column=0, sticky="w")
    mode_var = tk.StringVar(value="Box")
    mode_cb = ttk.Combobox(
        root, textvariable=mode_var, values=["Box", "Auto", "Points", "Text"]
    )

    mode_cb.grid(row=2, column=1, sticky="w", padx=4, pady=2)

    # --- Text prompt (enabled only for Text mode) ---
    tk.Label(root, text="Text prompt:").grid(row=3, column=0, sticky="w")
    prompt_var = tk.StringVar(value="")
    prompt_entry = tk.Entry(root, textvariable=prompt_var, width=40, state="disabled")
    prompt_entry.grid(row=3, column=1, padx=4, pady=2, sticky="w")

    # --- Options ---
    num_masks_label = tk.Label(root, text="Num Masks:")
    num_masks_label.grid(row=4, column=0, sticky="w")
    num_masks_var = tk.IntVar(value=1)
    num_masks_spin = tk.Spinbox(
        root, from_=1, to=10, textvariable=num_masks_var, width=6
    )
    num_masks_spin.grid(row=4, column=1, sticky="w", padx=4, pady=2)

    pfm_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Save as PFM", variable=pfm_var).grid(
        row=5, column=0, sticky="w"
    )

    # --- Status + Run button ---
    status_var = tk.StringVar(value="Ready.")
    status_lbl = tk.Label(root, textvariable=status_var, anchor="w")
    status_lbl.grid(row=6, column=0, columnspan=3, sticky="we", padx=4, pady=(6, 2))

    run_btn = tk.Button(root, text="Run")
    run_btn.grid(row=7, column=1, pady=(2, 8))

    def _set_running(is_running: bool, msg: str):
        # Must run on Tk main thread
        status_var.set(msg)
        run_btn.config(state=("disabled" if is_running else "normal"))

    def _toggle_prompt(*_):
        # Enable prompt only when Mode == Text
        if mode_var.get() == "Text":
            prompt_entry.config(state="normal")
            prompt_entry.focus_set()
        else:
            prompt_entry.config(state="disabled")

    def _toggle_num_masks(*_):
        if mode_var.get() == "Points":
            num_masks_label.grid_remove()
            num_masks_spin.grid_remove()
        else:
            num_masks_label.grid()
            num_masks_spin.grid()

    # trace_add triggers when variable changes :contentReference[oaicite:2]{index=2}
    mode_var.trace_add("write", _toggle_prompt)
    mode_var.trace_add("write", _toggle_num_masks)
    _toggle_prompt()
    _toggle_num_masks()

    def _call_with_supported_kwargs(func, **kwargs):
        """
        Small compatibility helper: passes only kwargs that the function accepts.
        This avoids crashes if your SAM3 wrappers differ slightly across modes.
        """
        sig = inspect.signature(func)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(**filtered)

    def _validate_paths(inp: str, out: str) -> bool:
        if not inp or not os.path.isfile(inp):
            messagebox.showwarning(
                "Missing input", "Please select a valid input image file."
            )
            return False
        if not out:
            messagebox.showwarning("Missing output", "Please select an output folder.")
            return False
        os.makedirs(out, exist_ok=True)
        return True

    def run_clicked():
        inp = input_var.get().strip()
        out = output_var.get().strip()
        n = int(num_masks_var.get())
        mode = mode_var.get()
        save_pfm = bool(pfm_var.get())
        prompt = prompt_var.get().strip()

        if not _validate_paths(inp, out):
            return

        if mode == "Text" and not prompt:
            messagebox.showwarning(
                "Missing prompt", "Please enter a text prompt for Text mode."
            )
            return

        # Disable Run while working; keep GUI responsive by using a worker thread :contentReference[oaicite:3]{index=3}
        _set_running(True, f"Running {mode}…")

    def run_clicked():
        inp = input_var.get().strip()
        out = output_var.get().strip()
        n = int(num_masks_var.get())
        mode = mode_var.get()
        save_pfm = bool(pfm_var.get())
        prompt = prompt_var.get().strip()

        if not _validate_paths(inp, out):
            return
        if mode == "Text" and not prompt:
            messagebox.showwarning(
                "Missing prompt", "Please enter a text prompt for Text mode."
            )
            return

        _set_running(True, f"Running {mode}…")
        root.update_idletasks()  # ensure label/button update before blocking work

        def do_work():
            try:
                if mode == "Text":
                    run_text_segmentation(inp, out, prompt, n, pfm=save_pfm)
                elif mode == "Points":
                    _call_with_supported_kwargs(
                        run_point_segmentation,
                        input_path=inp,
                        output_path=out,
                        num_masks=n,
                        pfm=save_pfm,
                    )
                elif mode == "Auto":
                    _call_with_supported_kwargs(
                        run_auto_segmentation,
                        input_path=inp,
                        output_path=out,
                        num_masks=n,
                        pfm=save_pfm,
                    )
                else:  # Box
                    _call_with_supported_kwargs(
                        run_box_segmentation,
                        input_path=inp,
                        output_path=out,
                        num_masks=n,
                        box=None,
                        pfm=save_pfm,
                    )

                _set_running(False, "Done.")
            except Exception as e:
                _set_running(False, "Failed.")
                messagebox.showerror("Error", str(e))

        root.after(0, do_work)  # run on Tk main thread

    run_btn.config(command=run_clicked)

    root.mainloop()
