import json
from pathlib import Path
from tkinter import Tk, Label, Entry, Button, StringVar, filedialog, messagebox

import cv2
import numpy as np
from tqdm import tqdm

from src.body import Body
from src.hand import Hand
from src import util


def save_json(candidate, subset, json_path):
    pose_data = {
        "candidate": candidate.tolist(),
        "subset": subset.tolist(),
    }
    with open(json_path, "w") as f:
        json.dump(pose_data, f)


def annotate_image(image_path, body_estimation, hand_estimation, json_path):
    ori_img = cv2.imread(str(image_path))
    if ori_img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    candidate, subset = body_estimation(ori_img)
    hands_list = util.handDetect(candidate, subset, ori_img)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(ori_img[y : y + w, x : x + w, :])
        peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
        peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
        all_hand_peaks.append(peaks)

    save_json(candidate, subset, json_path)


def annotate_dataset(input_dir, output_dir, body_model, hand_model):
    body_estimation = Body(body_model)
    hand_estimation = Hand(hand_model)

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in Path(input_dir).rglob("*") if p.suffix.lower() in image_suffixes]

    for img_path in tqdm(image_paths, desc="Annotating", unit="image"):
        relative = img_path.relative_to(input_dir)
        json_path = Path(output_dir) / relative.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        annotate_image(img_path, body_estimation, hand_estimation, json_path)


def start_gui():
    root = Tk()
    root.title("OpenPose Auto Annotation")

    input_dir_var = StringVar()
    output_dir_var = StringVar()
    body_model_var = StringVar()
    hand_model_var = StringVar()

    def browse_input():
        path = filedialog.askdirectory()
        if path:
            input_dir_var.set(path)

    def browse_output():
        path = filedialog.askdirectory()
        if path:
            output_dir_var.set(path)

    def browse_body_model():
        path = filedialog.askopenfilename()
        if path:
            body_model_var.set(path)

    def browse_hand_model():
        path = filedialog.askopenfilename()
        if path:
            hand_model_var.set(path)

    Label(root, text="Input Directory").grid(row=0, column=0, sticky="e")
    Entry(root, textvariable=input_dir_var, width=40).grid(row=0, column=1)
    Button(root, text="Browse", command=browse_input).grid(row=0, column=2)

    Label(root, text="Output Directory").grid(row=1, column=0, sticky="e")
    Entry(root, textvariable=output_dir_var, width=40).grid(row=1, column=1)
    Button(root, text="Browse", command=browse_output).grid(row=1, column=2)

    Label(root, text="Body Model Path").grid(row=2, column=0, sticky="e")
    Entry(root, textvariable=body_model_var, width=40).grid(row=2, column=1)
    Button(root, text="Browse", command=browse_body_model).grid(row=2, column=2)

    Label(root, text="Hand Model Path").grid(row=3, column=0, sticky="e")
    Entry(root, textvariable=hand_model_var, width=40).grid(row=3, column=1)
    Button(root, text="Browse", command=browse_hand_model).grid(row=3, column=2)

    def start():
        in_dir = input_dir_var.get()
        out_dir = output_dir_var.get()
        body_model = body_model_var.get()
        hand_model = hand_model_var.get()
        if not (in_dir and out_dir and body_model and hand_model):
            messagebox.showerror("Error", "Please select all paths before starting")
            return
        root.destroy()
        annotate_dataset(in_dir, out_dir, body_model, hand_model)
        print("Annotation completed.")

    Button(root, text="Start", command=start).grid(row=4, column=1)
    root.mainloop()


if __name__ == "__main__":
    start_gui()