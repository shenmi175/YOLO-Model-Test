"""
使用已经训练好的yolo模型进行预测，实现半自动标注
"""


import os
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.messagebox import showerror
import threading

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

class AutoAnnotateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 自动标注工具")

        # 初始化变量
        self.model_path = tk.StringVar()
        self.root_dir = tk.StringVar()
        self.confidence = tk.DoubleVar(value=0.5)
        self.iou = tk.DoubleVar(value=0.45)
        self.class_names = []

        # 构建界面
        self.build_ui()

    def build_ui(self):
        # 模型路径
        tk.Label(self.root, text="YOLOv8 模型:").grid(row=0, column=0, sticky='w')
        tk.Entry(self.root, textvariable=self.model_path, width=50).grid(row=0, column=1)
        tk.Button(self.root, text="浏览", command=self.select_model).grid(row=0, column=2)

        # 类别选择（多选 Listbox）
        tk.Label(self.root, text="选择类别:").grid(row=1, column=0, sticky='w')
        self.class_listbox = tk.Listbox(self.root, selectmode='multiple', exportselection=False, height=6)
        self.class_listbox.grid(row=1, column=1, sticky='ew', columnspan=2)

        # 置信度
        tk.Label(self.root, text="置信度阈值:").grid(row=2, column=0, sticky='w')
        tk.Entry(self.root, textvariable=self.confidence, width=10).grid(row=2, column=1, sticky='w')

        # IOU
        tk.Label(self.root, text="IOU 阈值:").grid(row=3, column=0, sticky='w')
        tk.Entry(self.root, textvariable=self.iou, width=10).grid(row=3, column=1, sticky='w')

        # 根目录
        tk.Label(self.root, text="根目录:").grid(row=4, column=0, sticky='w')
        tk.Entry(self.root, textvariable=self.root_dir, width=50).grid(row=4, column=1)
        tk.Button(self.root, text="浏览", command=self.select_root_dir).grid(row=4, column=2)

        # 开始按钮
        tk.Button(self.root, text="开始处理", command=self.start_processing).grid(row=5, column=1, pady=10)

        # 进度条
        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress.grid(row=6, column=0, columnspan=3, pady=10)

        # 状态栏
        self.status = tk.Label(self.root, text="等待操作...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.grid(row=7, column=0, columnspan=3, sticky='we')

    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLOv8 模型", "*.pt")])
        if path:
            self.model_path.set(path)
            self.load_model_classes(path)

    def load_model_classes(self, model_path):
        """加载模型并读取类别"""
        try:
            self.status.config(text="正在加载模型...")
            self.root.update_idletasks()
            self.model = YOLO(model_path)
            self.class_names = list(self.model.names.values())
            self.class_listbox.delete(0, tk.END)
            for cls in self.class_names:
                self.class_listbox.insert(tk.END, cls)
            self.class_listbox.config(state='normal')
            self.status.config(text="模型加载完成")
        except Exception as e:
            showerror("错误", f"加载模型失败: {e}")
            self.status.config(text="加载模型失败")

    def select_root_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.root_dir.set(path)

    def get_image_files(self, root_dir):
        """递归获取所有支持的图像文件"""
        image_files = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(root, f))
        return image_files

    def create_object_element(self, class_name, xmin, ymin, xmax, ymax):
        """创建一个 object 元素"""
        obj = ET.Element("object")
        ET.SubElement(obj, "name").text = class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)
        return obj

    def get_selected_classes(self):
        indices = self.class_listbox.curselection()
        return [self.class_names[i] for i in indices]

    def process_single_image(self, image_path, selected_classes, confidence, iou):
        try:
            img_width, img_height = self.get_image_size(image_path)
            results = self.model.predict(image_path, conf=confidence, iou=iou)
            boxes = results[0].boxes

            multi_class_boxes = []
            for box in boxes:
                class_name = self.model.names[int(box.cls)]
                if class_name in selected_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    xmin, ymin, xmax, ymax = int(x1), int(y1), int(x2), int(y2)
                    if xmin >= xmax or ymin >= ymax:
                        continue
                    if xmax > img_width or ymax > img_height:
                        continue
                    multi_class_boxes.append((class_name, xmin, ymin, xmax, ymax))

            xml_path = os.path.splitext(image_path)[0] + ".xml"
            xml_dir = os.path.dirname(xml_path)
            if not os.path.exists(xml_dir):
                os.makedirs(xml_dir)

            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
            else:
                root = ET.Element("annotation")
                folder = os.path.basename(os.path.dirname(image_path))
                ET.SubElement(root, "folder").text = folder
                ET.SubElement(root, "filename").text = os.path.basename(image_path)
                ET.SubElement(root, "path").text = image_path
                source = ET.SubElement(root, "source")
                ET.SubElement(source, "database").text = "Unknown"
                size = ET.SubElement(root, "size")
                ET.SubElement(size, "width").text = str(img_width)
                ET.SubElement(size, "height").text = str(img_height)
                ET.SubElement(size, "depth").text = "3"
                ET.SubElement(root, "segmented").text = "0"
                tree = ET.ElementTree(root)

            for class_name, xmin, ymin, xmax, ymax in multi_class_boxes:
                root.append(self.create_object_element(class_name, xmin, ymin, xmax, ymax))

            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            return len(multi_class_boxes)
        except Exception as e:
            self.status.config(text=f"写入失败: {e}")
            return 0

    def get_image_size(self, image_path):
        """获取图像的宽度和高度"""
        img = cv2.imread(image_path)
        if img is not None:
            return img.shape[1], img.shape[0]
        else:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

    def start_processing(self):
        model_path = self.model_path.get()
        root_dir = self.root_dir.get()
        selected_classes = self.get_selected_classes()
        confidence = self.confidence.get()
        iou = self.iou.get()

        if not os.path.isfile(model_path):
            showerror("错误", "请选择有效的 YOLOv8 模型文件")
            return

        if not self.class_names:
            showerror("错误", "请先加载模型以获取类别")
            return

        if not selected_classes:
            showerror("错误", "请选择要标注的类别")
            return

        if not os.path.isdir(root_dir):
            showerror("错误", "请选择有效的根目录")
            return

        if confidence < 0 or confidence > 1:
            showerror("错误", "置信度必须在 0~1 之间")
            return

        if iou < 0 or iou > 1:
            showerror("错误", "IOU 必须在 0~1 之间")
            return

        image_files = self.get_image_files(root_dir)
        total_images = len(image_files)
        if total_images == 0:
            showerror("错误", "未找到任何图像文件")
            return

        self.processing_thread = threading.Thread(
            target=self.run_processing,
            args=(image_files, selected_classes, confidence, iou, total_images)
        )
        self.processing_thread.start()

    def run_processing(self, image_files, selected_classes, confidence, iou, total_images):
        self.progress["maximum"] = total_images
        self.progress["value"] = 0
        for i, image_path in enumerate(image_files):
            try:
                self.status.config(text=f"处理中: {os.path.basename(image_path)}")
                self.process_single_image(image_path, selected_classes, confidence, iou)
                self.progress["value"] = i + 1
                self.root.update_idletasks()
            except Exception as e:
                self.status.config(text=f"处理失败: {e}")
        self.status.config(text="处理完成！")

if __name__ == '__main__':
    root = tk.Tk()
    app = AutoAnnotateGUI(root)
    root.mainloop()
