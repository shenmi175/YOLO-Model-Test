import os
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
import threading
import cv2
import numpy as np
from lxml import etree
from glob import glob
import random
from PIL import Image, ImageTk, ImageDraw

ALL_CLASSES = ['person', 'cat', 'dog', 'catface', 'dogface', 'hand', 'face']

def read_voc_xml(xml_path):
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            bbox = [int(float(bndbox.find('xmin').text)),
                    int(float(bndbox.find('ymin').text)),
                    int(float(bndbox.find('xmax').text)),
                    int(float(bndbox.find('ymax').text))]
            objects.append({'name': name, 'bbox': bbox})
        return objects
    except:
        return []

def copy_and_sync_xml(src_xml, dst_xml, bboxes):
    tree = etree.parse(src_xml)
    root = tree.getroot()
    objs = root.findall('object')
    for i, obj in enumerate(objs):
        if i < len(bboxes):
            bb = bboxes[i]['bbox']
            obj.find('bndbox/xmin').text = str(int(bb[0]))
            obj.find('bndbox/ymin').text = str(int(bb[1]))
            obj.find('bndbox/xmax').text = str(int(bb[2]))
            obj.find('bndbox/ymax').text = str(int(bb[3]))
            obj.find('name').text = bboxes[i]['name']
    tree.write(dst_xml)

def draw_bboxes_on_img_pil(img, bboxes, color=(255,0,0)):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    for obj in bboxes:
        x1, y1, x2, y2 = obj['bbox']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1+2, y1+2), obj['name'], fill=color)
    return im

# ----------------- 各增强方法 ------------------
def rgb_to_bgr(img):  return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
def to_gray(img): gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
def add_noise(img, region=None, sigma=20):
    if region is not None:
        x1, y1, x2, y2 = region
        roi = img[y1:y2, x1:x2].copy()
        noise = np.random.normal(0, sigma, roi.shape).astype(np.int16)
        roi = np.clip(roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img[y1:y2, x1:x2] = roi
    else:
        noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img
def gaussian_blur(img, region=None, ksize=5, sigma=3):
    ksize = max(3, int(ksize)//2*2+1)
    if region is not None:
        x1, y1, x2, y2 = region
        roi = img[y1:y2, x1:x2].copy()
        roi = cv2.GaussianBlur(roi, (ksize, ksize), sigma)
        img[y1:y2, x1:x2] = roi
    else:
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img
def jpeg_compress(img, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(encimg, 1)
    return img
def shrink_and_resize(img, scale=0.5):
    h, w = img.shape[:2]
    new_h, new_w = max(1,int(h*scale)), max(1,int(w*scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img
def add_stripe_occlusion_v2(img, bbox, orientation='horizontal', max_area_ratio=0.2, n_stripes=3, min_gap_ratio=0.05, color=None):
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2].copy()
    h, w = roi.shape[:2]
    area = h * w
    color = color if color is not None else [random.randint(0,255) for _ in range(3)]
    n_stripes = max(1, n_stripes)
    min_gap = int(min_gap_ratio * (h if orientation=='horizontal' else w))
    if orientation == 'horizontal':
        total_stripe_area = area * max_area_ratio
        stripe_h = max(1, int((total_stripe_area / n_stripes) / w))
        y_positions = []
        y = 0
        for i in range(n_stripes):
            if y+stripe_h > h:
                break
            y_positions.append(y)
            y += stripe_h + min_gap
        for y in y_positions:
            cv2.rectangle(roi, (0, y), (w, min(y+stripe_h, h-1)), color, -1)
    else:
        total_stripe_area = area * max_area_ratio
        stripe_w = max(1, int((total_stripe_area / n_stripes) / h))
        x_positions = []
        x = 0
        for i in range(n_stripes):
            if x+stripe_w > w:
                break
            x_positions.append(x)
            x += stripe_w + min_gap
        for x in x_positions:
            cv2.rectangle(roi, (x, 0), (min(x+stripe_w, w-1), h), color, -1)
    img[y1:y2, x1:x2] = roi
    return img
def add_area_occlusion(img, bbox, area_ratio=0.1):
    x1, y1, x2, y2 = bbox
    h, w = y2-y1, x2-x1
    occ_area = int(h*w*area_ratio)
    for _ in range(2):  # 可以加多块
        rh, rw = random.randint(int(0.1*h), int(0.5*h)), random.randint(int(0.1*w), int(0.5*w))
        ry = random.randint(0, h-rh)
        rx = random.randint(0, w-rw)
        color = [random.randint(0,255) for _ in range(3)]
        img[y1+ry:y1+ry+rh, x1+rx:x1+rx+rw] = color
    return img
def add_sun_flare(img, num_circles=8, src_radius=70):
    h, w = img.shape[:2]
    overlay = img.copy()
    for i in range(num_circles):
        alpha = random.uniform(0.12, 0.22)
        radius = random.randint(int(src_radius*0.3), int(src_radius*1.3))
        x = int(w * random.uniform(0.2, 0.8))
        y = int(h * random.uniform(0.1, 0.4))
        color = [random.randint(180,255) for _ in range(3)]
        cv2.circle(overlay, (x, y), radius, color, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img
def add_shadow(img, shadow_dim=40, transparency=0.4):
    h, w = img.shape[:2]
    top_x, bot_x = random.randint(0, w//2), random.randint(w//2, w)
    poly = np.array([[top_x,0],[bot_x,h],[w,h],[w,0]], np.int32)
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, [poly], (0,0,0))
    img = cv2.addWeighted(img, 1-transparency, mask, transparency, 0)
    return img
def lens_distortion(img, distortion=0.15):
    h, w = img.shape[:2]
    K = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float32)
    D = np.array([distortion,0,0,0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (w,h), cv2.CV_32FC1)
    dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    return dst

# ================ GUI主类 ==================
class AugmentationGUI:
    def __init__(self, master):
        self.master = master
        master.title("图像增强批量工具-全滑条美观预览")
        self.param_vars = {}
        self.class_vars = {}
        self.use_method = {}  # 新增
        self.img_list = []
        self.xml_list = []
        self.preview_img_idx = 0

        self.augment_defs = [
            ('通道变换', 'channel', [
                ('p_bgr', '', 0.0, 1.0, 0.2, 0.01, False, 'p_bgr', "RGB转BGR概率"),
                ('p_gray','', 0.0, 1.0, 0.1, 0.01, False, 'p_gray', "灰度概率"),
            ], None),
            ('遮挡', 'occlusion', [
                ('p_stripe', '', 0.0, 1.0, 0.12, 0.01, False, 'p_stripe', "条状遮挡概率"),
                ('n_stripes','', 1, 6, 3, 1, True, 'n_stripes', "条数"),
                ('max_area','', 0.05, 0.3, 0.15, 0.01, False, 'max_area', "面积比例"),
                ('gap','', 0.03, 0.15, 0.07, 0.01, False, 'gap', "间隔比例"),
                ('p_area','', 0.0, 1.0, 0.12, 0.01, False, 'p_area', "区域遮挡概率"),
                ('area_ratio','', 0.05, 0.2, 0.11, 0.01, False, 'area_ratio', "区域占比"),
            ], ALL_CLASSES),
            ('噪声', 'noise', [
                ('p','',0.0,1.0,0.12,0.01,False,'p',"概率"),
                ('sigma','',3,60,16,1,True,'sigma',"噪声强度"),
            ], ALL_CLASSES+['__all__']),
            ('模糊', 'blur', [
                ('p','',0.0,1.0,0.18,0.01,False,'p',"概率"),
                ('ksize','',3,19,5,2,True,'ksize',"核大小"),
                ('sigma','',0,14,2,1,True,'sigma',"高斯σ"),
            ], ALL_CLASSES+['__all__']),
            ('压缩/降质','compression', [
                ('p','',0.0,1.0,0.11,0.01,False,'p',"概率"),
                ('jpeg','',20,100,62,1,True,'jpeg',"JPEG质量"),
                ('scale','',0.2,1.0,0.44,0.01,False,'scale',"缩放比例"),
            ], None),
            ('耀斑', 'sunflare', [
                ('p','',0.0,1.0,0.09,0.01,False,'p',"概率"),
                ('circles','',2,13,8,1,True,'circles',"圆圈数"),
                ('radius','',20,180,60,1,True,'radius',"主光源半径"),
            ], None),
            ('阴影', 'shadow', [
                ('p','',0.0,1.0,0.10,0.01,False,'p',"概率"),
                ('dim','',8,80,36,2,True,'dim',"尺寸"),
                ('trans','',0.1,0.8,0.33,0.01,False,'trans',"透明度"),
            ], None),
            ('畸变', 'distort', [
                ('p','',0.0,1.0,0.12,0.01,False,'p',"概率"),
                ('k','',-0.35,0.35,0.16,0.01,False,'k',"畸变强度"),
            ], None),
        ]
        # ====== 顶部文件/预览控件 ======
        tk.Label(master, text="图片目录:").grid(row=0, column=0, sticky='e')
        self.dir_var = tk.StringVar()
        tk.Entry(master, textvariable=self.dir_var, width=34).grid(row=0, column=1, columnspan=2)
        tk.Button(master, text="选择", command=self.select_dir).grid(row=0, column=3)
        self.run_btn = tk.Button(master, text="开始增强", command=self.run)
        self.run_btn.grid(row=0, column=5, padx=10)
        # ===== 预览区 =====
        self.tk_orig = None
        self.tk_aug = None
        self.orig_label = tk.Label(master, text="原图")
        self.orig_label.grid(row=1, column=0, rowspan=12)
        self.aug_label = tk.Label(master, text="增强后")
        self.aug_label.grid(row=13, column=0, rowspan=12)

        # ===== 参数控件生成 =====
        row = 1
        for label, key, controls, class_opt in self.augment_defs:
            ttk.Separator(master, orient='horizontal').grid(row=row, column=2, columnspan=14, sticky='ew', pady=2)
            row += 1
            # 新增: 每组前加启用勾选框
            self.use_method[key] = tk.IntVar(value=1)
            tk.Checkbutton(master, variable=self.use_method[key], command=self.update_preview).grid(row=row, column=2, sticky='e')
            tk.Label(master, text=label, font=('Arial',11,'bold')).grid(row=row, column=3, sticky='w')
            row += 1
            self.param_vars[key] = {}
            col = 4
            for (pkey, unit, vmin, vmax, vdef, step, is_int, pkey_id, desc) in controls:
                var = tk.DoubleVar(value=vdef)
                scale = tk.Scale(master, from_=vmin, to=vmax, resolution=step, orient=tk.HORIZONTAL,
                                 variable=var, length=120, label=desc+(f"({unit})" if unit else ""),
                                 command=lambda e: self.update_preview())
                scale.set(vdef)
                scale.grid(row=row, column=col, columnspan=2, sticky='w')
                self.param_vars[key][pkey_id] = (var, is_int)
                col += 2
            if class_opt is not None:
                row += 1
                col = 4
                tk.Label(master, text="作用类别:").grid(row=row, column=col)
                col += 1
                classbox = {}
                for ci, cname in enumerate(ALL_CLASSES):
                    var = tk.IntVar(value=1)
                    classbox[cname] = var
                    tk.Checkbutton(master, text=cname, variable=var, command=self.update_preview).grid(row=row, column=col+ci)
                if '__all__' in class_opt:
                    var = tk.IntVar(value=0)
                    classbox['__all__'] = var
                    tk.Checkbutton(master, text="全图", variable=var, command=self.update_preview).grid(row=row, column=col+len(ALL_CLASSES))
                self.class_vars[key] = classbox
            row += 1

        # ===== 日志区 =====
        row += 1
        ttk.Separator(master, orient='horizontal').grid(row=row, column=0, columnspan=16, sticky='ew', pady=2)
        row += 1
        self.log_text = tk.Text(master, width=108, height=8)
        self.log_text.grid(row=row, column=0, columnspan=16)

    # ==== 仅加载第一张图片和xml进行预览 ====
    def select_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.dir_var.set(d)
            img_list = []
            xml_list = []
            for ext in ('*.jpg','*.jpeg','*.png'):
                img_list.extend(sorted(glob(os.path.join(d, ext))))
            if not img_list:
                messagebox.showerror("提示", "无图片文件")
                return
            img = img_list[0]
            base = os.path.splitext(img)[0]
            xml = base + '.xml'
            self.img_list = [img]
            self.xml_list = [xml]
            self.update_preview()

    def get_params(self):
        out = {}
        for label, key, controls, class_opt in self.augment_defs:
            d = {}
            for (pkey, unit, vmin, vmax, vdef, step, is_int, pkey_id, desc) in controls:
                var, is_int = self.param_vars[key][pkey_id]
                d[pkey_id] = int(var.get()) if is_int else float(var.get())
            if class_opt:
                d['class'] = [c for c in self.class_vars[key] if self.class_vars[key][c].get()]
            out[key] = d
        return out

    def update_preview(self, *args):
        if not self.img_list: return
        img = cv2.imread(self.img_list[0])
        xml_path = self.xml_list[0]
        bboxes = read_voc_xml(xml_path) if os.path.exists(xml_path) else []
        params = self.get_params()
        img_aug = img.copy()
        # ---------- 增强全流程，只有勾选才执行 ----------
        if self.use_method['channel'].get():
            if random.random() < params['channel']['p_bgr']:
                img_aug = rgb_to_bgr(img_aug)
            if random.random() < params['channel']['p_gray']:
                img_aug = to_gray(img_aug)
        if self.use_method['occlusion'].get():
            if random.random() < params['occlusion']['p_stripe']:
                classes = params['occlusion']['class']
                for obj in bboxes:
                    if obj['name'] in classes:
                        img_aug = add_stripe_occlusion_v2(
                            img_aug, obj['bbox'],
                            orientation=random.choice(['horizontal','vertical']),
                            max_area_ratio=params['occlusion']['max_area'],
                            n_stripes=params['occlusion']['n_stripes'],
                            min_gap_ratio=params['occlusion']['gap']
                        )
            if random.random() < params['occlusion']['p_area']:
                classes = params['occlusion']['class']
                for obj in bboxes:
                    if obj['name'] in classes:
                        img_aug = add_area_occlusion(img_aug, obj['bbox'], params['occlusion']['area_ratio'])
        if self.use_method['noise'].get():
            if random.random() < params['noise']['p']:
                sigma = params['noise']['sigma']
                img_aug = add_noise(img_aug, None, sigma)
        if self.use_method['blur'].get():
            if random.random() < params['blur']['p']:
                img_aug = gaussian_blur(img_aug, None, params['blur']['ksize'], params['blur']['sigma'])
        if self.use_method['compression'].get():
            if random.random() < params['compression']['p']:
                img_aug = jpeg_compress(img_aug, params['compression']['jpeg'])
                img_aug = shrink_and_resize(img_aug, params['compression']['scale'])
        if self.use_method['sunflare'].get():
            if random.random() < params['sunflare']['p']:
                img_aug = add_sun_flare(img_aug, params['sunflare']['circles'], params['sunflare']['radius'])
        if self.use_method['shadow'].get():
            if random.random() < params['shadow']['p']:
                img_aug = add_shadow(img_aug, params['shadow']['dim'], params['shadow']['trans'])
        if self.use_method['distort'].get():
            if random.random() < params['distort']['p']:
                img_aug = lens_distortion(img_aug, params['distort']['k'])
        # 绘制增强后检测框
        im_orig = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((960,540))
        im_aug = draw_bboxes_on_img_pil(img_aug, bboxes).resize((960,540))
        self.tk_orig = ImageTk.PhotoImage(im_orig)
        self.tk_aug = ImageTk.PhotoImage(im_aug)
        self.orig_label.config(image=self.tk_orig)
        self.aug_label.config(image=self.tk_aug)

    def run(self):
        img_dir = self.dir_var.get()
        if not os.path.isdir(img_dir):
            messagebox.showerror("错误", "请选择正确的图片目录")
            return
        params = self.get_params()
        out_dir = img_dir + '_aug'
        os.makedirs(out_dir, exist_ok=True)
        img_files = sorted([f for f in glob(os.path.join(img_dir, '*')) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for idx, img_path in enumerate(img_files):
            img_name = os.path.basename(img_path)
            xml_path = os.path.splitext(img_path)[0] + '.xml'
            img = cv2.imread(img_path)
            bboxes = read_voc_xml(xml_path) if os.path.exists(xml_path) else []
            img_aug = img.copy()
            # == 全增强流程与预览一致 ==
            if self.use_method['channel'].get():
                if random.random() < params['channel']['p_bgr']:
                    img_aug = rgb_to_bgr(img_aug)
                if random.random() < params['channel']['p_gray']:
                    img_aug = to_gray(img_aug)
            if self.use_method['occlusion'].get():
                if random.random() < params['occlusion']['p_stripe']:
                    classes = params['occlusion']['class']
                    for obj in bboxes:
                        if obj['name'] in classes:
                            img_aug = add_stripe_occlusion_v2(
                                img_aug, obj['bbox'],
                                orientation=random.choice(['horizontal','vertical']),
                                max_area_ratio=params['occlusion']['max_area'],
                                n_stripes=params['occlusion']['n_stripes'],
                                min_gap_ratio=params['occlusion']['gap']
                            )
                if random.random() < params['occlusion']['p_area']:
                    classes = params['occlusion']['class']
                    for obj in bboxes:
                        if obj['name'] in classes:
                            img_aug = add_area_occlusion(img_aug, obj['bbox'], params['occlusion']['area_ratio'])
            if self.use_method['noise'].get():
                if random.random() < params['noise']['p']:
                    img_aug = add_noise(img_aug, None, params['noise']['sigma'])
            if self.use_method['blur'].get():
                if random.random() < params['blur']['p']:
                    img_aug = gaussian_blur(img_aug, None, params['blur']['ksize'], params['blur']['sigma'])
            if self.use_method['compression'].get():
                if random.random() < params['compression']['p']:
                    img_aug = jpeg_compress(img_aug, params['compression']['jpeg'])
                    img_aug = shrink_and_resize(img_aug, params['compression']['scale'])
            if self.use_method['sunflare'].get():
                if random.random() < params['sunflare']['p']:
                    img_aug = add_sun_flare(img_aug, params['sunflare']['circles'], params['sunflare']['radius'])
            if self.use_method['shadow'].get():
                if random.random() < params['shadow']['p']:
                    img_aug = add_shadow(img_aug, params['shadow']['dim'], params['shadow']['trans'])
            if self.use_method['distort'].get():
                if random.random() < params['distort']['p']:
                    img_aug = lens_distortion(img_aug, params['distort']['k'])
            out_path = os.path.join(out_dir, img_name)
            cv2.imwrite(out_path, img_aug)
            if os.path.exists(xml_path):
                out_xml = os.path.join(out_dir, os.path.basename(xml_path))
                copy_and_sync_xml(xml_path, out_xml, bboxes)
            self.log(f"{idx+1}/{len(img_files)}: {img_name} 完成")
        self.log("全部增强完成，增强图片和xml保存在：" + out_dir)

    def log(self, msg):
        self.log_text.insert(tk.END, msg+'\n')
        self.log_text.see(tk.END)
        self.master.update()

if __name__ == '__main__':
    root = tk.Tk()
    gui = AugmentationGUI(root)
    root.mainloop()





