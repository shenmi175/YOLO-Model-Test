#!/usr/bin/env python3
"""Convert Pascal VOC XML annotations to simple TXT format.

Each line in the output contains: ``label xmin ymin xmax ymax``.
Use ``--normalize`` to divide coordinates by the image width/height.

Example
-------
```
python tools/xml_to_txt.py test_data/test1 --normalize
python tools/xml_to_txt.py --gui  # launch GUI
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog, messagebox


def parse_xml(xml_path: str) -> tuple[int, int, list[tuple[str, int, int, int, int]]]:
    """Parse a single XML file and return width, height and boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.findtext("size/width", "1"))
    height = int(root.findtext("size/height", "1"))

    boxes: list[tuple[str, int, int, int, int]] = []
    for obj in root.findall("object"):
        label = obj.findtext("name") or "unknown"
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = int(float(bnd.findtext("xmin", "0")))
            ymin = int(float(bnd.findtext("ymin", "0")))
            xmax = int(float(bnd.findtext("xmax", "0")))
            ymax = int(float(bnd.findtext("ymax", "0")))
        except ValueError:
            continue
        boxes.append((label, xmin, ymin, xmax, ymax))
    return width, height, boxes


def convert(xml_file: str, normalize: bool) -> None:
    width, height, boxes = parse_xml(xml_file)
    txt_path = Path(xml_file).with_suffix(".txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        for label, xmin, ymin, xmax, ymax in boxes:
            if normalize:
                x1 = xmin / width
                y1 = ymin / height
                x2 = xmax / width
                y2 = ymax / height
                f.write(f"{label} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")
            else:
                f.write(f"{label} {xmin} {ymin} {xmax} {ymax}\n")



def process_path(path: str, normalize: bool) -> None:
    p = Path(path)
    if p.is_dir():
        for xml_path in p.rglob("*.xml"):
            convert(str(xml_path), normalize)
    else:
        convert(str(p), normalize)


def run_gui() -> None:
    root = tk.Tk()
    root.title("XML to TXT Converter")

    path_var = tk.StringVar()
    norm_var = tk.BooleanVar()

    def select_path() -> None:
        path = filedialog.askdirectory(title="Select folder with XML files")
        if not path:
            files = filedialog.askopenfilename(title="Select XML file", filetypes=[("XML", "*.xml")])
            path = files
        path_var.set(path)

    def start() -> None:
        path = path_var.get()
        if not path:
            messagebox.showerror("Error", "Please select a file or directory")
            return
        process_path(path, norm_var.get())
        messagebox.showinfo("Done", "Conversion complete")

    tk.Label(root, text="XML file or directory:").pack(padx=10, pady=5)
    entry = tk.Entry(root, textvariable=path_var, width=50)
    entry.pack(padx=10, pady=5)
    tk.Button(root, text="Browse", command=select_path).pack(padx=10, pady=5)
    tk.Checkbutton(root, text="Normalize coordinates", variable=norm_var).pack(padx=10, pady=5)
    tk.Button(root, text="Convert", command=start).pack(padx=10, pady=10)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert XML annotations to TXT")
    parser.add_argument("paths", nargs="*", help="XML files or directories containing XMLs")
    parser.add_argument("--normalize", action="store_true", help="Normalize coordinates")
    parser.add_argument("--gui", action="store_true", help="Launch GUI for parameters")
    args = parser.parse_args()

    if args.gui or not args.paths:
        run_gui()
        return

    for path in args.paths:
        process_path(path, args.normalize)


if __name__ == "__main__":
    main()