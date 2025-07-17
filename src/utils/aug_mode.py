import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, _, cy = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0,0,0,0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
            font=("tahoma", "10", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()



class AugModeHelper:
    def __init__(self, gui):
        """
        gui: 主界面AugmentationGUI类的实例
        """
        self.gui = gui

    def parse_voc_xml(self, xml_path):
        """
        读取VOC格式xml文件，返回bboxes和class_labels
        bboxes: [(x1, y1, x2, y2), ...]
        class_labels: [name1, name2, ...]
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        class_labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            x1 = int(float(bbox.find("xmin").text))
            y1 = int(float(bbox.find("ymin").text))
            x2 = int(float(bbox.find("xmax").text))
            y2 = int(float(bbox.find("ymax").text))
            bboxes.append([x1, y1, x2, y2])
            class_labels.append(name)
        return bboxes, class_labels

    # ========== 绘制检测框 ==========
    def draw_boxes(self, img, bboxes, labels=None, color=(0, 255, 0)):
        img = img.copy()
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if labels is not None:
                label = str(labels[i])
                cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img

    # =======================
    # 区间滑块联动
    # =======================
    def slider_pair_link(self, min_slider, max_slider):
        """
        联动两个滑块：min_slider, max_slider 保证 min <= max
        只需要将事件分别绑定到两个滑块即可
        """
        min_val = min_slider.get()
        max_val = max_slider.get()
        changed = False

        if min_val > max_val:
            max_slider.set(min_val)
            changed = True
        if max_val < min_val:
            min_slider.set(max_val)
            changed = True

        if changed or True:
            self.update_preview()

    # 普朗克抖动滑块联动
    def get_safe_temperature_limit(self):
        tmin = self.gui.temp_min.get()
        tmax = self.gui.temp_max.get()
        # 确保tmin < tmax
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        # 保证6500在区间内
        if tmin > 6500:
            tmin = 6500
        if tmax < 6500:
            tmax = 6500
        return (tmin, tmax)

    # =======================
    # 图片相关
    # =======================
    def load_image(self, file_path=None):
        if file_path is None:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
            if not file_path:
                return
        self.gui.image_path = file_path
        self.gui.orig_img = cv2.imread(file_path)
        if self.gui.orig_img is None:
            messagebox.showerror("错误", f"无法读取图片: {file_path}")
            return
        self.show_image(self.gui.orig_img, self.gui.orig_canvas)
        self.update_preview()

    def show_image(self, img, canvas):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        im_pil = im_pil.resize((960, 540))
        imgtk = ImageTk.PhotoImage(im_pil)
        canvas.imgtk = imgtk
        canvas.config(image=imgtk)

    def save_aug_image(self):
        if getattr(self.gui, 'aug_img', None) is None:
            messagebox.showerror("错误", "还未生成增强图像！")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            cv2.imwrite(save_path, self.gui.aug_img)
            messagebox.showinfo("提示", f"已保存到 {save_path}")

    # =======================
    # 预览更新
    # =======================
    def update_preview(self):
        if getattr(self.gui, 'orig_img', None) is None:
            return
        aug_img = self.gui.orig_img.copy()

        # 运动模糊
        if getattr(self.gui, 'use_motion_blur', None) and self.gui.use_motion_blur.get():
            aug_img = self.gui.apply_motion_blur(
                aug_img,
                blur_limit=self.gui.blur_limit.get(),
                angle=self.gui.angle.get(),
                direction=self.gui.direction.get(),
                p=self.gui.blur_probability.get()
            )

        # 加性噪声
        if getattr(self.gui, 'use_add_noise', None) and self.gui.use_add_noise.get():
            aug_img = self.gui.apply_AdditiveNoise(
                aug_img,
                noise_type=self.gui.noise_type_var.get(),
                spatial_mode=self.gui.spatial_mode_var.get(),
                mean_range=(self.gui.mean_min.get(), self.gui.mean_max.get()),
                std_range=(self.gui.std_min.get(), self.gui.std_max.get()),
                approximation=self.gui.approximation.get(),
                p=self.gui.add_noise_probability.get()
            )

        # 灰度变换
        if getattr(self.gui, 'use_To_Gray', None) and self.gui.use_To_Gray.get():
            aug_img = self.gui.apply_ToGray(
                aug_img,
                method=self.gui.To_Gray_var.get(),
                p=self.gui.To_Gray_probability.get()
            )

        # 普朗克抖动
        if getattr(self.gui, 'use_Planckian_Jitter', None) and self.gui.use_Planckian_Jitter.get():
            aug_img = self.gui.apply_PlanckianJitter(
                aug_img,
                mode=self.gui.mode_var.get(),
                sampling_method=self.gui.sampling_method_var.get(),
                # 不传 temperature_limit
                p=self.gui.Planckian_Jitter_probability.get()
            )

        # 浮雕效果
        if getattr(self.gui, 'use_Emboss', None) and self.gui.use_Emboss.get():
            aug_img = self.gui.apply_Emboss(
                aug_img,
                alpha=(self.gui.alpha_min.get(), self.gui.alpha_max.get()),
                strength=(self.gui.strength_min.get(), self.gui.strength_max.get()),
                p=self.gui.Emboss_probability.get()
            )

        # 应用散粒噪声
        if getattr(self.gui, 'use_ShotNoise', None) and self.gui.use_ShotNoise.get():
            aug_img = self.gui.apply_ShotNoise(
                aug_img,
                scale_range=(self.gui.scale_min.get(), self.gui.scale_max.get()),
                p=self.gui.ShotNoise_probability.get()
            )

        # 应用相机传感器噪声
        if getattr(self.gui, 'use_ISONoise', None) and self.gui.use_ISONoise.get():
            aug_img = self.gui.apply_ISONoise(
                aug_img,
                color_shift=(self.gui.acolor_shift_min.get(), self.gui.color_shift_max.get()),
                intensity=(self.gui.intensity_min.get(), self.gui.intensity_max.get()),
                p=self.gui.Emboss_probability.get()
            )

        # 应用改变色调、饱和度和明度
        if getattr(self.gui, 'use_HueSaturationValue', None) and self.gui.use_HueSaturationValue.get():
            aug_img = self.gui.apply_HueSaturationValue(
                aug_img,
                hue_shift_limit=(self.gui.hue_shift_limit_min.get(), self.gui.hue_shift_limit_max.get()),
                sat_shift_limit=(self.gui.sat_shift_limit_min.get(), self.gui.sat_shift_limit_max.get()),
                val_shift_limit=(self.gui.val_shift_limit_min.get(), self.gui.val_shift_limit_max.get()),
                p=self.gui.HueSaturationValue_probability.get()
            )

        # 应用光照效果
        if getattr(self.gui, 'use_Illumination', None) and self.gui.use_Illumination.get():
            aug_img = self.gui.apply_Illumination(
                aug_img,
                Illumination_mode=self.gui.Illumination_mode_var.get(),
                effect_type=self.gui.effect_type_var.get(),
                intensity_range=(self.gui.intensity_range_min.get(), self.gui.intensity_range_max.get()),
                angle_range=(self.gui.angle_range_min.get(), self.gui.angle_range_max.get()),
                center_range=(self.gui.center_range_min.get(), self.gui.center_range_max.get()),
                sigma_range=(self.gui.sigma_range_min.get(), self.gui.sigma_range_max.get()),
                p=self.gui.Illumination_probability.get()
            )

        # 应用失焦模糊
        if getattr(self.gui, 'use_Defocus', None) and self.gui.use_Defocus.get():
            aug_img = self.gui.apply_Defocus(
                aug_img,
                radius=(self.gui.radius_min.get(), self.gui.radius_max.get()),
                alias_blur=(self.gui.alias_blur_min.get(), self.gui.alias_blur_max.get()),
                p=self.gui.Defocus_probability.get()
            )

        # 应用缩放模糊
        if getattr(self.gui, 'use_ZoomBlur', None) and self.gui.use_ZoomBlur.get():
            aug_img = self.gui.apply_ZoomBlur(
                aug_img,
                max_factor=(self.gui.max_factor_min.get(), self.gui.max_factor_max.get()),
                astep_factor=(self.gui.astep_factor_min.get(), self.gui.astep_factor_max.get()),
                p=self.gui.ZoomBlur_probability.get()
            )

        # 应用光学扭曲
        if getattr(self.gui, 'use_OpticalDistortion', None) and self.gui.use_OpticalDistortion.get():
            aug_img = self.gui.apply_OpticalDistortion(
                aug_img,
                OpticalDistortion_mode=self.gui.OpticalDistortion_mode_var.get(),
                effect_type=self.gui.effect_type_var.get(),
                distort_limit=(self.gui.distort_limit_min.get(), self.gui.distort_limit_max.get()),
                p=self.gui.OpticalDistortion_probability.get()
            )

        self.gui.aug_img = aug_img
        self.show_image(aug_img, self.gui.aug_canvas)
