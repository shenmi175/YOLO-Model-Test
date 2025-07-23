#!/usr/bin/env python3
"""GUI tool to split a dataset into images and label folders.

The script lets you choose a root directory and select subfolders to copy.
Selected folders will have their images and matching ``.txt`` annotation files
copied to the given destination directories. Duplicate names are automatically
renamed while keeping image and annotation names consistent.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox


def list_subdirs(root: Path) -> list[Path]:
    """Return all subdirectories under ``root`` (recursively), sorted."""
    return sorted([p for p in root.rglob("*") if p.is_dir()])


def gather_files(dirs: list[Path]) -> list[tuple[Path, Path]]:
    """Gather image and txt file pairs inside ``dirs``."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs: list[tuple[Path, Path]] = []
    for d in dirs:
        for img in d.rglob("*"):
            if img.suffix.lower() in exts:
                txt = img.with_suffix(".txt")
                if txt.exists():
                    pairs.append((img, txt))
    return pairs


def unique_paths(img_path: Path, txt_path: Path, img_dst: Path, lbl_dst: Path) -> tuple[Path, Path]:
    """Return destination paths ensuring no name conflicts."""
    stem = img_path.stem
    suffix = img_path.suffix
    new_img = img_dst / f"{stem}{suffix}"
    new_txt = lbl_dst / f"{stem}.txt"
    idx = 1
    while new_img.exists() or new_txt.exists():
        new_img = img_dst / f"{stem}_{idx}{suffix}"
        new_txt = lbl_dst / f"{stem}_{idx}.txt"
        idx += 1
    return new_img, new_txt


def copy_pairs(pairs: list[tuple[Path, Path]], img_dst: Path, lbl_dst: Path, progress: ttk.Progressbar | None = None) -> None:
    total = len(pairs)
    if progress is not None:
        progress["maximum"] = total
    for idx, (img, txt) in enumerate(pairs, 1):
        dst_img, dst_txt = unique_paths(img, txt, img_dst, lbl_dst)
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_img)
        shutil.copy2(txt, dst_txt)
        if progress is not None:
            progress["value"] = idx
            progress.update_idletasks()


def launch() -> None:
    """Launch the Tkinter GUI."""
    root = tk.Tk()
    root.title("Dataset Splitter")

    tk.Label(root, text="Dataset root:").grid(row=0, column=0, sticky="e")
    root_var = tk.StringVar()
    tk.Entry(root, textvariable=root_var, width=60).grid(row=0, column=1)
    tk.Button(root, text="Browse", command=lambda: root_var.set(
        filedialog.askdirectory() or root_var.get())
    ).grid(row=0, column=2)

    tk.Label(root, text="Images output:").grid(row=1, column=0, sticky="e")
    img_var = tk.StringVar()
    tk.Entry(root, textvariable=img_var, width=60).grid(row=1, column=1)
    tk.Button(root, text="Browse", command=lambda: img_var.set(
        filedialog.askdirectory() or img_var.get())
    ).grid(row=1, column=2)

    tk.Label(root, text="Labels output:").grid(row=2, column=0, sticky="e")
    lbl_var = tk.StringVar()
    tk.Entry(root, textvariable=lbl_var, width=60).grid(row=2, column=1)
    tk.Button(root, text="Browse", command=lambda: lbl_var.set(
        filedialog.askdirectory() or lbl_var.get())
    ).grid(row=2, column=2)

    list_frame = tk.Frame(root)
    list_frame.grid(row=3, columnspan=3, pady=10)
    canvas = tk.Canvas(list_frame)
    scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)
    scroll_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set, width=500, height=200)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    dir_vars: list[tuple[tk.BooleanVar, Path]] = []

    def load_dirs() -> None:
        for child in scroll_frame.winfo_children():
            child.destroy()
        dir_vars.clear()
        path = Path(root_var.get())
        if not path.exists():
            return
        for sub in list_subdirs(path):
            var = tk.BooleanVar()
            cb = tk.Checkbutton(scroll_frame, text=sub.relative_to(path).as_posix(), variable=var)
            cb.pack(anchor="w")
            dir_vars.append((var, sub))

    # 新增：全选/取消全选功能
    select_all_var = tk.BooleanVar(value=False)  # 默认不全选

    def select_all_folders():
        # 切换选中/取消选中
        new_value = not select_all_var.get()
        select_all_var.set(new_value)
        for var, _ in dir_vars:
            var.set(new_value)
        # 更新全选按钮文本
        if new_value:
            select_all_btn.config(text="取消全选")
        else:
            select_all_btn.config(text="全选")

    tk.Button(root, text="Load Folders", command=load_dirs).grid(row=0, column=3, padx=5)

    # 新增：全选/取消全选按钮
    select_all_btn = tk.Button(root, text="全选", command=select_all_folders)
    select_all_btn.grid(row=0, column=4, padx=5)

    tk.Button(root, text="Load Folders", command=load_dirs).grid(row=0, column=3, padx=5)

    progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
    progress.grid(row=4, columnspan=4, pady=5)

    def start() -> None:
        img_dst = Path(img_var.get())
        lbl_dst = Path(lbl_var.get())
        if not img_dst or not lbl_dst or not Path(root_var.get()).exists():
            messagebox.showerror("Error", "Please select valid directories")
            return
        selected = [p for var, p in dir_vars if var.get()]
        if not selected:
            messagebox.showerror("Error", "No folders selected")
            return
        pairs = gather_files(selected)
        if not pairs:
            messagebox.showinfo("Dataset Splitter", "No files found")
            return
        copy_pairs(pairs, img_dst, lbl_dst, progress)
        messagebox.showinfo("Dataset Splitter", "Done")

    tk.Button(root, text="Start", command=start).grid(row=5, columnspan=4, pady=5)
    root.mainloop()


if __name__ == "__main__":
    launch()