"""Simple Tkinter GUI for launching model evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox


def launch() -> None:
    """Launch the parameter selection window."""
    root = tk.Tk()
    root.title("YOLO Model Test")

    tk.Label(root, text="Model path:").grid(row=0, column=0, sticky="e")
    model_var = tk.StringVar(value="models/best.pt")
    tk.Entry(root, textvariable=model_var, width=40).grid(row=0, column=1)

    tk.Label(root, text="Data directory:").grid(row=1, column=0, sticky="e")
    data_var = tk.StringVar(value="test_data")
    tk.Entry(root, textvariable=data_var, width=40).grid(row=1, column=1)

    tk.Label(root, text="Output directory:").grid(row=2, column=0, sticky="e")
    out_var = tk.StringVar(value="output")
    tk.Entry(root, textvariable=out_var, width=40).grid(row=2, column=1)

    save_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Save predictions", variable=save_var).grid(row=3, columnspan=2)

    def run_cli() -> None:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "cli.py"),
            "--model",
            model_var.get(),
            "--data",
            data_var.get(),
            "--output",
            out_var.get(),
        ]
        if not save_var.get():
            cmd.append("--no-save")
        subprocess.Popen(cmd)
        messagebox.showinfo("YOLO Model Test", "Started: {}".format(" ".join(cmd)))

    tk.Button(root, text="Run", command=run_cli).grid(row=4, columnspan=2, pady=5)
    root.mainloop()


if __name__ == "__main__":
    launch()