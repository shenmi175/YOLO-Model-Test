import tkinter as tk
from tkinter import ttk
from utils.aug_mode import ToolTip, AugModeHelper
from augment.aug import *

DEFAULT_IMAGE = r'G:\A_Share\YOLO-Model-Test\test_data\test1\test2\22112604_002940.jpg'

class AugmentationGUI:
    def create_probability_control(self, frame, row, col_label=2, col_scale=3, text="概率:", default=0.5, command=None):
        label = tk.Label(frame, text=text)
        label.grid(row=row, column=col_label, sticky='e')
        ToolTip(label, "应用转换的概率。默认值: 0.5")
        scale = tk.Scale(frame, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.1, length=180, command=command)
        scale.set(default)
        scale.grid(row=row, column=col_scale, sticky='w', padx=2)
        return label, scale



    def __init__(self, master):
        self.master = master
        master.title("图像增强工具")
        self.apply_motion_blur = apply_motion_blur
        self.apply_AdditiveNoise = apply_AdditiveNoise
        self.apply_ToGray = apply_ToGray
        self.apply_PlanckianJitter = apply_PlanckianJitter
        self.apply_Emboss = apply_Emboss
        self.apply_ShotNoise=apply_ShotNoise
        self.apply_ISONoise=apply_ISONoise
        self.apply_HueSaturationValue=apply_HueSaturationValue
        self.apply_Illumination=apply_Illumination
        self.apply_Defocus=apply_Defocus
        self.apply_ZoomBlur=apply_ZoomBlur
        self.apply_OpticalDistortion=apply_OpticalDistortion

        self.helper = AugModeHelper(self)

        # ===== 图片显示区域 (独占一行) =====
        img_frame = tk.Frame(master)
        img_frame.grid(row=0, column=0, columnspan=4, padx=10, pady=(10, 0), sticky='w')
        self.orig_label = tk.Label(img_frame, text="原图")
        self.orig_label.pack(side='left')
        self.orig_canvas = tk.Label(img_frame)
        self.orig_canvas.pack(side='left', padx=10)
        self.aug_label = tk.Label(img_frame, text="增强后")
        self.aug_label.pack(side='left')
        self.aug_canvas = tk.Label(img_frame)
        self.aug_canvas.pack(side='left', padx=10)

        # 选择图片按钮
        self.select_btn = tk.Button(master, text="选择图片", command=self.helper.load_image)
        self.select_btn.grid(row=1, column=0, padx=10, pady=2, sticky='w')
        tk.Label(master, text="增强根目录:").grid(row=2, column=0, sticky='e', padx=10)
        self.input_root_var = tk.StringVar()
        tk.Entry(master, textvariable=self.input_root_var, width=40).grid(row=2, column=1, sticky='w')
        tk.Button(master, text="浏览", command=self.helper.browse_input_root).grid(row=2, column=2, sticky='w')

        tk.Label(master, text="输出根目录:").grid(row=3, column=0, sticky='e', padx=10)
        self.output_root_var = tk.StringVar()
        tk.Entry(master, textvariable=self.output_root_var, width=40).grid(row=3, column=1, sticky='w')
        tk.Button(master, text="浏览", command=self.helper.browse_output_root).grid(row=3, column=2, sticky='w')

        tk.Label(master, text="增强数量:").grid(row=4, column=0, sticky='e', padx=10)
        self.augment_count_var = tk.IntVar(value=1)
        tk.Entry(master, textvariable=self.augment_count_var, width=10).grid(row=4, column=1, sticky='w')
        tk.Button(master, text="开始增强", command=self.helper.batch_augment).grid(row=4, column=2, sticky='w')

        # ==== 可滚动参数区域 ====
        scroll_container = tk.Frame(master)
        scroll_container.grid(row=5, column=0, columnspan=4, sticky='nsew')
        master.grid_rowconfigure(5, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(scroll_container, highlightthickness=0)
        self.v_scroll = tk.Scrollbar(scroll_container, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.v_scroll.pack(side='right', fill='y')

        self.scroll_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor='nw')
        self.scroll_frame.bind(
            '<Configure>',
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        )
        self.canvas.bind_all(
            '<MouseWheel>',
            lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        )

        # ==== 各分组网格参数 ====
        max_cols = 3   # 一行显示几列分组，可调整
        group_frames = []  # 保存各分组frame（用于批量布局）

        # ========== 定义所有分组及参数 ==========
        # 按需在groups里补充/调整分组
        groups = [
            {
                'title': "运动模糊",
                'build_func': self.build_motion_blur_group,
            },
            {
                'title': "加性噪声",
                'build_func': self.build_add_noise_group,
            },
            {
                'title': "灰度变换",
                'build_func': self.build_to_gray_group,
            },
            {
                'title': "普朗克抖动",
                'build_func': self.build_planckian_jitter_group,
            },
            {
                'title': "浮雕效果",
                'build_func': self.build_emboss_group,
            },
            {
                'title': "散粒噪声",
                'build_func': self.build_ShotNoise_group,
            },
            {
                'title': "相机传感器噪声",
                'build_func': self.build_ISONoise_group,
            },
            {
                'title': "改变色调、饱和度和明度",
                'build_func': self.build_HueSaturationValue_group,
            },
            {
                'title': "光照效果",
                'build_func': self.build_Illumination_group,
            },
            {
                'title': "失焦模糊",
                'build_func': self.build_Defocus_group,
            },
            {
                'title': "缩放模糊",
                'build_func': self.build_ZoomBlur_group,
            },
            {
                'title': "光学扭曲",
                'build_func': self.build_OpticalDistortion_group,
            },
        ]

        group_row = 0
        group_col = 0
        for group in groups:
            frame = tk.LabelFrame(self.scroll_frame, text=group['title'], padx=10, pady=5)
            frame.grid(row=group_row, column=group_col, padx=10, pady=6, sticky='nw')
            group['build_func'](frame)
            group_frames.append(frame)
            group_col += 1
            if group_col >= max_cols:
                group_col = 0
                group_row += 1

        # 保存按钮
        self.save_btn = tk.Button(self.scroll_frame, text="保存增强后图片", command=self.helper.save_aug_image)
        self.save_btn.grid(row=group_row+1, column=0, padx=10, pady=10, sticky='w')

        # 默认加载图片
        self.helper.load_image(DEFAULT_IMAGE)

    # ===== 分组控件封装为函数，每组独立 =====
    def build_motion_blur_group(self, frame):
        self.use_motion_blur = tk.BooleanVar(value=True)
        self.motion_blur_check = tk.Checkbutton(frame, text="启用", variable=self.use_motion_blur, command=self.helper.update_preview)
        self.motion_blur_check.grid(row=0, column=0, sticky='w')
        self.blur_label = tk.Label(frame, text="模糊核大小:")
        self.blur_label.grid(row=1, column=0, sticky='e')
        ToolTip(self.blur_label, '''
        模糊的最大核大小。
        应在范围 [3, inf) 内。
        - 如果是整数：核大小将从[3, blur_limit]中随机选择
        - 如果是元组：核大小将从[min, max]中随机选择
        更大的值会产生更强的模糊效果。
        默认值：(3, 7)
        ''')
        self.blur_limit = tk.Scale(frame, from_=3, to=31, orient=tk.HORIZONTAL, resolution=2, length=180, command=lambda v: self.helper.update_preview())
        self.blur_limit.set(13)
        self.blur_limit.grid(row=1, column=1, sticky='w', padx=2)

        self.angle_label = tk.Label(frame, text="角度:")
        self.angle_label.grid(row=1, column=2, sticky='e')
        ToolTip(self.angle_label, '''
        可能的角度范围，单位为度。
        控制运动模糊线的旋转：
        - 0°: 水平运动模糊 →
        - 45°: 对角线运动模糊 ↗
        - 90°: 垂直运动模糊 ↑
        - 135°: 对角线运动模糊 ↖
        默认: (0, 360)
        ''')
        self.angle = tk.Scale(frame, from_=0, to=180, orient=tk.HORIZONTAL, length=180, command=lambda v: self.helper.update_preview())
        self.angle.set(0)
        self.angle.grid(row=1, column=3, sticky='w', padx=2)

        self.direction_label = tk.Label(frame, text="方向:")
        self.direction_label.grid(row=2, column=0, sticky='e')
        ToolTip(self.direction_label, '''
        运动偏移的范围。
        控制模糊从中心扩展的方式：
        - -1.0：模糊仅向后扩展（←）
        -  0.0：模糊向两个方向均匀扩展（←→）
        - 1.0: 模糊效果仅向前扩展 (→)
        例如，当角度=0 时：
        - direction=-1.0: ←•
        - direction=0.0:  ←•→
        - direction=1.0:   •→
        默认值: (-1.0, 1.0)
        ''')
        self.direction = tk.Scale(frame, from_=-1, to=1, orient=tk.HORIZONTAL, resolution=0.1, length=180, command=lambda v: self.helper.update_preview())
        self.direction.set(0)
        self.direction.grid(row=2, column=1, sticky='w', padx=2)
        self.blur_probability_label, self.blur_probability = self.create_probability_control(
            frame, row=2, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())

    def build_add_noise_group(self, frame):
        self.use_add_noise = tk.BooleanVar(value=False)
        self.add_noise_check = tk.Checkbutton(frame, text="启用", variable=self.use_add_noise, command=self.helper.update_preview)
        self.add_noise_check.grid(row=0, column=0, sticky='w')
        self.noise_type_label = tk.Label(frame, text="噪声类型:")
        self.noise_type_label.grid(row=1, column=0, sticky='e')
        ToolTip(self.noise_type_label, '''
        要使用的噪声分布类型。选项：
        - "uniform": 均匀分布，适用于简单的随机扰动
        - "gaussian": 正态分布，模拟自然随机过程
        - "laplace": 类似于正态分布但尾部更重，适用于异常值
        - "beta": 灵活的有界分布，可以是对称的或偏态的
        ''')
        self.noise_type_var = tk.StringVar(value="gaussian")
        self.noise_type_menu = tk.OptionMenu(frame, self.noise_type_var, "uniform", "gaussian", "laplace", "beta", command=lambda _: self.helper.update_preview())
        self.noise_type_menu.grid(row=1, column=1, sticky='w')
        self.spatial_mode_label = tk.Label(frame, text="空间模式:")
        self.spatial_mode_label.grid(row=1, column=2, sticky='e')
        ToolTip(self.spatial_mode_label, '''
        如何生成和应用噪声。选项：
        - "constant": 每个通道一个噪声值，最快
        - "per_pixel": 每个像素和通道一个独立的噪声值，最慢
        - "shared": 所有通道共享一个噪声图，中等速度
        ''')
        self.spatial_mode_var = tk.StringVar(value="shared")
        self.spatial_mode_menu = tk.OptionMenu(frame, self.spatial_mode_var, "constant", "per_pixel", "shared", command=lambda _: self.helper.update_preview())
        self.spatial_mode_menu.grid(row=1, column=3, sticky='w')
        self.approximation_label = tk.Label(frame, text="approximation:")
        self.approximation_label.grid(row=2, column=0, sticky='e')
        ToolTip(self.approximation_label, '''
        浮点数在[0, 1]范围内，默认值为 1.0
        控制噪声生成速度与质量之间的权衡。
        - 1.0：生成全分辨率噪声（最慢，最高质量）
        - 0.5: 在半分辨率下生成噪声并上采样
        - 0.25: 在四分之一分辨率下生成噪声并上采样
        仅影响 'per_pixel' 和 'shared' 空间模式。
        ''')
        self.approximation = tk.Scale(frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=120, command=lambda _: self.helper.update_preview())
        self.approximation.set(1.0)
        self.approximation.grid(row=2, column=1, sticky='w')

        self.mean_range_label = tk.Label(frame, text="mean_range:")
        self.mean_range_label.grid(row=3, column=0, sticky='e')
        ToolTip(self.mean_range_label, '''
        高斯：
        mean_range: tuple[float, float], default (0.0, 0.0)
        采样均值值的范围，在[-1, 1]之间
        std_range: tuple[float, float], default (0.1, 0.1)
        Range for sampling standard deviation, in [0, 1]
        ''')

        self.mean_min = tk.Scale(
            frame, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.mean_min, self.mean_max)
        )
        self.mean_min.set(0.0)
        self.mean_min.grid(row=3, column=1, sticky='w')
        self.mean_max = tk.Scale(
            frame, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.mean_min, self.mean_max)
        )
        self.mean_max.set(0.0)
        self.mean_max.grid(row=3, column=2, sticky='w')
        self.std_range_label = tk.Label(frame, text="std_range:")
        self.std_range_label.grid(row=4, column=0, sticky='e')
        ToolTip(self.std_range_label, '''
        高斯：
        mean_range: tuple[float, float], default (0.0, 0.0)
        采样均值值的范围，在[-1, 1]之间
        std_range: tuple[float, float], default (0.1, 0.1)
        Range for sampling standard deviation, in [0, 1]
        ''')

        self.std_min = tk.Scale(
            frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.std_min, self.std_max)
        )
        self.std_min.set(0.05)
        self.std_min.grid(row=4, column=1, sticky='w')
        self.std_max = tk.Scale(
            frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.std_min, self.std_max)
        )
        self.std_max.set(0.15)
        self.std_max.grid(row=4, column=2, sticky='w')
        self.add_noise_probability_label, self.add_noise_probability = self.create_probability_control(
            frame, row=2, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())

    def build_to_gray_group(self, frame):
        self.use_To_Gray = tk.BooleanVar(value=True)
        self.To_Gray_check = tk.Checkbutton(frame, text="启用", variable=self.use_To_Gray, command=self.helper.update_preview)
        self.To_Gray_check.grid(row=0, column=0, sticky='w')
        self.To_Gray_label = tk.Label(frame, text="方法选择:")
        self.To_Gray_label.grid(row=1, column=0, sticky='e')
        ToolTip(self.To_Gray_label, '''
        用于灰度转换的方法：
        - "weighted_average": 使用 RGB 通道的加权总和（0.299R + 0.587G + 0.114B）。
          仅适用于 3 通道图像。根据人类感知提供逼真的结果。
        - "from_lab": 从 LAB 色彩空间中提取 L 通道。
        仅适用于 3 通道图像。提供感知上均匀的结果。
        - "desaturation": 对所有通道的最大值和最小值进行平均。
        适用于任意数量的通道。快速但可能无法很好地保留感知亮度。
        - "average": 所有通道的简单平均值。
        适用于任意数量的通道。速度快但可能无法得到逼真的结果。
        - "max": 在所有通道中取最大值。
        适用于任意数量的通道。倾向于产生更亮的结果。
        - "pca": 应用主成分分析来减少通道。
        适用于任意数量的通道。可以保留更多信息，但计算密集。
        ''')
        self.To_Gray_var = tk.StringVar(value="weighted_average")
        self.To_Gray_menu = tk.OptionMenu(frame, self.To_Gray_var, "from_lab", "desaturation", "average", "max", "pca", "weighted_average", command=lambda _: self.helper.update_preview())
        self.To_Gray_menu.grid(row=1, column=3, sticky='w')
        self.To_Gray_probability_label, self.To_Gray_probability = self.create_probability_control(
            frame, row=2, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())

    def build_planckian_jitter_group(self, frame):
        self.temperature_ranges = {
            'blackbody': (3000, 15000),
            'cied': (4000, 15000),
        }
        self.use_Planckian_Jitter = tk.BooleanVar(value=False)
        self.Planckian_Jitter_check = tk.Checkbutton(
            frame, text="启用", variable=self.use_Planckian_Jitter,
            command=self.helper.update_preview)
        self.Planckian_Jitter_check.grid(row=0, column=0, sticky='w')

        # mode选择
        self.mode_label = tk.Label(frame, text="模式:")
        self.mode_label.grid(row=1, column=0, sticky='e')
        ToolTip(self.mode_label, '''
        转换模式。
        - "blackbody"：模拟黑体辐射颜色变化。
        - "cied"：使用 CIE D 光源系列进行色温模拟。
        ''')
        self.mode_var = tk.StringVar(value='blackbody')
        self.mode_menu = tk.OptionMenu(
            frame, self.mode_var, *self.temperature_ranges.keys(),
            command=lambda _: self.on_mode_change()
        )
        self.mode_menu.grid(row=1, column=1, sticky='w')

        # 采样温度方法
        self.sampling_method_label = tk.Label(frame, text="采样温度方法:")
        self.sampling_method_label.grid(row=3, column=0, sticky='e')
        ToolTip(self.sampling_method_label, '''
        用于采样温度的方法。
        - "uniform": 在指定范围内均匀采样。
        - "gaussian": 从以 6500K 为中心的高斯分布（近似日光）采样。
        ''')
        self.sampling_method_var = tk.StringVar(value="uniform")
        self.sampling_method_menu = tk.OptionMenu(
            frame, self.sampling_method_var, "uniform", "gaussian",
            command=lambda _: self.helper.update_preview())
        self.sampling_method_menu.grid(row=3, column=1, sticky='w')

        # 概率
        self.Planckian_Jitter_probability_label, self.Planckian_Jitter_probability = self.create_probability_control(
            frame, row=4, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())

        # ========== 模式切换时自动更新温度滑块区间 ==========

    def build_emboss_group(self, frame):
        self.use_Emboss = tk.BooleanVar(value=True)
        self.emboss_check = tk.Checkbutton(frame, text="启用", variable=self.use_Emboss,
                                                command=self.helper.update_preview)
        self.emboss_check.grid(row=0, column=0, sticky='w')
        self.alpha_range_label = tk.Label(frame, text="alpha_range:")
        self.alpha_range_label.grid(row=3, column=0, sticky='e')
        ToolTip(self.alpha_range_label, '''
        选择凸起图像可见性的范围。
        在 0 时，仅显示原始图像，在 1.0 时，仅显示其凸起版本。
        值应在[0, 1]范围内。
        Alpha 将随机从该范围中选择，用于每张图像。
        默认值：(0.2, 0.5)
        ''')
        self.alpha_min = tk.Scale(frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=100,
                                 command=lambda v: self.helper.slider_pair_link(self.alpha_min, self.alpha_max))
        self.alpha_min.set(0.2)
        self.alpha_min.grid(row=3, column=1, sticky='w')
        self.alpha_max = tk.Scale(frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=100,
                                 command=lambda v: self.helper.slider_pair_link(self.alpha_min, self.alpha_max))
        self.alpha_max.set(0.5)
        self.alpha_max.grid(row=3, column=2, sticky='w')

        self.strength_range_label = tk.Label(frame, text="strength_range:")
        self.strength_range_label.grid(row=4, column=0, sticky='e')
        ToolTip(self.strength_range_label, '''
        选择浮雕效果的强度范围。
        更高的值会产生更明显的 3D 效果。
        值应为非负。
        每张图像的强度将从这个范围内随机选择。
        默认值：(0.2, 0.7)
        ''')
        self.strength_min = tk.Scale(frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=100,
                                command=lambda v: self.helper.slider_pair_link(self.strength_min, self.strength_max))
        self.strength_min.set(0.2)
        self.strength_min.grid(row=4, column=1, sticky='w')
        self.strength_max = tk.Scale(frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=100,
                                command=lambda v: self.helper.slider_pair_link(self.strength_min, self.strength_max))
        self.strength_max.set(0.7)
        self.strength_max.grid(row=4, column=2, sticky='w')
        self.Emboss_probability_label, self.Emboss_probability = self.create_probability_control(
            frame, row=2, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())


    def build_ShotNoise_group(self, frame):
        self.use_ShotNoise = tk.BooleanVar(value=True)
        self.ShotNoise_check = tk.Checkbutton(frame, text="启用", variable=self.use_ShotNoise,
                                           command=self.helper.update_preview)
        self.ShotNoise_check.grid(row=0, column=0, sticky='w')
        self.scale_range_label = tk.Label(frame, text="scale:")
        self.scale_range_label.grid(row=3, column=0, sticky='e')
        ToolTip(self.scale_range_label, '''
                噪声尺度因子的采样范围。
                表示单位强度下预期光子数的倒数。
                数值越高意味着更多噪声：
                - scale = 0.1: 每单位强度约 100 个光子（低噪声）
                - scale = 1.0: ~1 个光子每单位强度（中等噪声）
                - scale = 10.0: ~0.1 个光子每单位强度（高噪声）
                默认: (0.1, 0.3)
                ''')
        self.scale_min = tk.Scale(frame, from_=0.05, to=1, resolution=0.1, orient=tk.HORIZONTAL, length=100,
                                  command=lambda v: self.helper.slider_pair_link(self.scale_min, self.scale_max))
        self.scale_min.set(0.1)
        self.scale_min.grid(row=3, column=1, sticky='w')
        self.scale_max = tk.Scale(frame, from_=0.05, to=1, resolution=0.1, orient=tk.HORIZONTAL, length=100,
                                  command=lambda v: self.helper.slider_pair_link(self.scale_min, self.scale_max))
        self.scale_max.set(0.3)
        self.scale_max.grid(row=3, column=2, sticky='w')


        self.ShotNoise_probability_label, self.ShotNoise_probability = self.create_probability_control(
            frame, row=2, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())

    def build_ISONoise_group(self, frame):
        self.use_ISONoise = tk.BooleanVar(value=True)
        self.ISONoise_check = tk.Checkbutton(frame, text="启用", variable=self.use_ISONoise,
                                              command=self.helper.update_preview)
        self.ISONoise_check.grid(row=0, column=0, sticky='w')
        self.color_shift_label = tk.Label(frame, text="色彩色调范围:")
        self.color_shift_label.grid(row=3, column=0, sticky='e')
        ToolTip(self.color_shift_label, '''
                        color_shift:
                        改变色彩色调的范围。
                        值应在[0, 1]范围内，其中 1 表示完整的 360°色调旋转。
                        默认值：(0.01, 0.05)
                        ''')
        self.acolor_shift_min = tk.Scale(frame, from_=0.05, to=0.2, resolution=0.01, orient=tk.HORIZONTAL, length=100,
                                  command=lambda v: self.helper.slider_pair_link(self.acolor_shift_min, self.color_shift_max))
        self.acolor_shift_min.set(0.2)
        self.acolor_shift_min.grid(row=3, column=1, sticky='w')
        self.color_shift_max = tk.Scale(frame, from_=0.05, to=0.2, resolution=0.01, orient=tk.HORIZONTAL, length=100,
                                  command=lambda v: self.helper.slider_pair_link(self.acolor_shift_min, self.color_shift_max))
        self.color_shift_max.set(0.5)
        self.color_shift_max.grid(row=3, column=2, sticky='w')

        self.intensity_label = tk.Label(frame, text="噪声强度:")
        self.intensity_label.grid(row=3, column=0, sticky='e')
        ToolTip(self.intensity_label, '''
                                color_shift:
                                改变色彩色调的范围。
                                值应在[0, 1]范围内，其中 1 表示完整的 360°色调旋转。
                                默认值：(0.01, 0.05)
                                ''')
        self.intensity_min = tk.Scale(frame, from_=0.1, to=0.5, resolution=0.05, orient=tk.HORIZONTAL, length=100,
                                         command=lambda v: self.helper.slider_pair_link(self.intensity_min,
                                                                                        self.intensity_max))
        self.intensity_min.set(0.1)
        self.intensity_min.grid(row=3, column=1, sticky='w')
        self.intensity_max = tk.Scale(frame, from_=0.1, to=0.5, resolution=0.05, orient=tk.HORIZONTAL, length=100,
                                        command=lambda v: self.helper.slider_pair_link(self.intensity_min,
                                                                                       self.intensity_max))
        self.intensity_max.set(0.5)
        self.intensity_max.grid(row=3, column=2, sticky='w')



        self.ISONoise_probability_label, self.ISONoise_probability = self.create_probability_control(
            frame, row=2, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())

    def build_HueSaturationValue_group(self, frame):

        self.use_HueSaturationValue = tk.BooleanVar(value=True)
        self.HueSaturationValue_check = tk.Checkbutton(frame, text="启用", variable=self.use_HueSaturationValue,
                                              command=self.helper.update_preview)
        self.HueSaturationValue_check.grid(row=0, column=0, sticky='w')
        self.hue_shift_limit_label = tk.Label(frame, text="色调范围:")
        self.hue_shift_limit_label.grid(row=3, column=0, sticky='e')
        ToolTip(self.hue_shift_limit_label, '''
                        改变色调的范围。
                        如果提供一个单一的浮点数值，范围将是 (-hue_shift_limit, hue_shift_limit)。
                        值应在 [-180, 180] 范围内。默认：(-20, 20)。
                        ''')
        self.hue_shift_limit_min = tk.Scale(frame, from_=-180, to=180, resolution=10, orient=tk.HORIZONTAL, length=100,
                                  command=lambda v: self.helper.slider_pair_link(self.hue_shift_limit_min, self.hue_shift_limit_max))
        self.hue_shift_limit_min.set(-20)
        self.hue_shift_limit_min.grid(row=3, column=1, sticky='w')
        self.hue_shift_limit_max = tk.Scale(frame, from_=-180, to=180, resolution=10, orient=tk.HORIZONTAL, length=100,
                                  command=lambda v: self.helper.slider_pair_link(self.hue_shift_limit_min, self.hue_shift_limit_max))
        self.hue_shift_limit_max.set(20)
        self.hue_shift_limit_max.grid(row=3, column=2, sticky='w')

        self.sat_shift_limit_label = tk.Label(frame, text="饱和度范围:")
        self.sat_shift_limit_label.grid(row=3, column=0, sticky='e')
        ToolTip(self.sat_shift_limit_label, '''
                        改变饱和度的范围。
                        如果提供一个单一的浮点值，范围将是 (-sat_shift_limit, sat_shift_limit)。
                        值应在 [-255, 255] 范围内。默认：(-30, 30)。
                                ''')
        self.sat_shift_limit_min = tk.Scale(frame, from_=-250, to=250, resolution=10, orient=tk.HORIZONTAL, length=100,
                                         command=lambda v: self.helper.slider_pair_link(self.sat_shift_limit_min,
                                                                                        self.sat_shift_limit_max))
        self.sat_shift_limit_min.set(-30)
        self.sat_shift_limit_min.grid(row=3, column=1, sticky='w')
        self.sat_shift_limit_max = tk.Scale(frame, from_=-250, to=250, resolution=10, orient=tk.HORIZONTAL, length=100,
                                        command=lambda v: self.helper.slider_pair_link(self.sat_shift_limit_min,
                                                                                       self.sat_shift_limit_max))
        self.sat_shift_limit_max.set(30)
        self.sat_shift_limit_max.grid(row=3, column=2, sticky='w')

        self.val_shift_limit_label = tk.Label(frame, text="亮度范围:")
        self.val_shift_limit_label.grid(row=3, column=0, sticky='e')
        ToolTip(self.val_shift_limit_label, '''
                                改变饱和度的范围。
                                如果提供一个单一的浮点值，范围将是 (-sat_shift_limit, sat_shift_limit)。
                                值应在 [-255, 255] 范围内。默认：(-30, 30)。
                                        ''')
        self.val_shift_limit_min = tk.Scale(frame, from_=-250, to=250, resolution=10, orient=tk.HORIZONTAL, length=100,
                                            command=lambda v: self.helper.slider_pair_link(self.val_shift_limit_min,
                                                                                           self.val_shift_limit_max))
        self.val_shift_limit_min.set(-20)
        self.val_shift_limit_min.grid(row=3, column=1, sticky='w')
        self.val_shift_limit_max = tk.Scale(frame, from_=-250, to=250, resolution=10, orient=tk.HORIZONTAL, length=100,
                                            command=lambda v: self.helper.slider_pair_link(self.sat_shift_limit_min,
                                                                                           self.val_shift_limit_max))
        self.val_shift_limit_max.set(20)
        self.val_shift_limit_max.grid(row=3, column=2, sticky='w')

        self.HueSaturationValue_probability_label, self.HueSaturationValue_probability = self.create_probability_control(
            frame, row=2, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())

    def build_Illumination_group(self, frame):

        self.use_Illumination = tk.BooleanVar(value=False)
        self.Illumination_check = tk.Checkbutton(frame, text="启用", variable=self.use_Illumination,
                                              command=self.helper.update_preview)
        self.Illumination_check.grid(row=0, column=0, sticky='w')

        self.mode_type_label = tk.Label(frame, text="照明模式类型:")
        self.mode_type_label.grid(row=1, column=0, sticky='e')
        ToolTip(self.mode_type_label, '''
                照明模式类型：
                - 'linear': 在图像上创建平滑的渐变，
                模拟方向性光照，如阳光
                通过窗户
                - 'corner': 从任意角落应用渐变，
                           模拟从角落发出的光源
                - 'gaussian': 创建圆形聚光灯效果，
                             模拟局部光源
                默认值：'linear'
                ''')
        self.Illumination_mode_var = tk.StringVar(value="linear")
        self.Illumination_mode_menu = tk.OptionMenu(frame, self.noise_type_var, "linear", "corner", "gaussian",
                                             command=lambda _: self.helper.update_preview())
        self.Illumination_mode_menu.grid(row=1, column=1, sticky='w')

        self.effect_type_label = tk.Label(frame, text="光照变化的类型:")
        self.effect_type_label.grid(row=1, column=2, sticky='e')
        ToolTip(self.effect_type_label, '''
                光照变化的类型：
                - 'brighten': 仅增加光照（如同聚光灯）
                - 'darken': 仅移除光线（如同阴影）
                - 'both': 随机选择增亮或减暗
                默认: 'both'
                ''')
        self.effect_type_var = tk.StringVar(value="both")
        self.effect_type_menu = tk.OptionMenu(frame, self.effect_type_var, "both", "brighten", "darken",
                                               command=lambda _: self.helper.update_preview())
        self.effect_type_menu.grid(row=1, column=3, sticky='w')

        self.intensity_range_label = tk.Label(frame, text="效果强度:")
        self.intensity_range_label.grid(row=3, column=0, sticky='e')
        ToolTip(self.intensity_range_label, '''
                高斯：
                mean_range: tuple[float, float], default (0.0, 0.0)
                采样均值值的范围，在[-1, 1]之间
                std_range: tuple[float, float], default (0.1, 0.1)
                Range for sampling standard deviation, in [0, 1]
                ''')

        self.intensity_range_min = tk.Scale(
            frame, from_=0.01, to=0.2, resolution=0.01, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.intensity_range_min, self.mean_max)
        )
        self.intensity_range_min.set(0.01)
        self.intensity_range_min.grid(row=3, column=1, sticky='w')
        self.intensity_range_max = tk.Scale(
            frame, from_=0.01, to=0.2, resolution=0.01, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.mean_min, self.intensity_range_max)
        )
        self.intensity_range_max.set(0.2)
        self.intensity_range_max.grid(row=3, column=2, sticky='w')

        self.angle_range_label = tk.Label(frame, text="渐变角度:")
        self.angle_range_label.grid(row=4, column=0, sticky='e')
        ToolTip(self.angle_range_label, '''
                渐变角度的度数范围。
                控制线性渐变的方向：
                - 0°：从左到右
                - 90°：从上到下
                - 180°：从右到左
                - 270°: 从下到上
                仅用于'linear'模式。
                默认值: (0, 360)
                ''')

        self.angle_range_min = tk.Scale(
            frame, from_=0, to=360, resolution=10, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.angle_range_min, self.angle_range_max)
        )
        self.angle_range_min.set(0)
        self.angle_range_min.grid(row=4, column=1, sticky='w')
        self.angle_range_max = tk.Scale(
            frame, from_=0, to=360, resolution=10, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.angle_range_min, self.angle_range_max)
        )
        self.angle_range_max.set(360)
        self.angle_range_max.grid(row=4, column=2, sticky='w')

        self.center_range_label = tk.Label(frame, text="聚光灯位置:")
        self.center_range_label.grid(row=4, column=0, sticky='e')
        ToolTip(self.center_range_label, '''
                聚光灯位置的范围。
                0 到 1 之间的值表示相对位置：
                - (0, 0): 左上角
                - (1, 1): 右下角
                - (0.5, 0.5): 图像中心
                仅用于'gaussian'模式。
                默认值: (0.1, 0.9)
                ''')
        self.center_range_min = tk.Scale(
            frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.center_range_min, self.center_range_max)
        )
        self.center_range_min.set(0)
        self.center_range_min.grid(row=4, column=1, sticky='w')
        self.center_range_max = tk.Scale(
            frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.center_range_min, self.center_range_max)
        )
        self.center_range_max.set(1)
        self.center_range_max.grid(row=4, column=2, sticky='w')

        self.sigma_range_label = tk.Label(frame, text="聚光灯大小:")
        self.sigma_range_label.grid(row=4, column=0, sticky='e')
        ToolTip(self.sigma_range_label, '''
                聚光灯位置的范围。
                0 到 1 之间的值表示相对位置：
                - (0, 0): 左上角
                - (1, 1): 右下角
                - (0.5, 0.5): 图像中心
                仅用于'gaussian'模式。
                默认值: (0.1, 0.9)
                ''')
        self.sigma_range_min = tk.Scale(
            frame, from_=0.2, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.sigma_range_min, self.sigma_range_max)
        )
        self.sigma_range_min.set(0.2)
        self.sigma_range_min.grid(row=4, column=1, sticky='w')
        self.sigma_range_max = tk.Scale(
            frame, from_=0.2, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=100,
            command=lambda v: self.helper.slider_pair_link(self.sigma_range_min, self.sigma_range_max)
        )
        self.sigma_range_max.set(1)
        self.sigma_range_max.grid(row=4, column=2, sticky='w')

        self.Illumination_probability_label, self.Illumination_probability = self.create_probability_control(
            frame, row=2, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())

    def build_Defocus_group(self, frame):
        self.use_Defocus = tk.BooleanVar(value=False)
        tk.Checkbutton(frame, text="启用", variable=self.use_Defocus, command=self.helper.update_preview).grid(row=0, column=0, sticky='w')
        tk.Label(frame, text="模糊半径:").grid(row=1, column=0, sticky='e')
        self.radius_min = tk.Scale(frame, from_=1, to=15, orient=tk.HORIZONTAL, length=100,
                                   command=lambda v: self.helper.slider_pair_link(self.radius_min, self.radius_max))
        self.radius_min.set(3)
        self.radius_min.grid(row=1, column=1, sticky='w')
        self.radius_max = tk.Scale(frame, from_=1, to=15, orient=tk.HORIZONTAL, length=100,
                                   command=lambda v: self.helper.slider_pair_link(self.radius_min, self.radius_max))
        self.radius_max.set(10)
        self.radius_max.grid(row=1, column=2, sticky='w')
        tk.Label(frame, text="高斯模糊的标准偏差范围:").grid(row=2, column=0, sticky='e')
        self.alias_blur_min = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=100,
                                       command=lambda v: self.helper.slider_pair_link(self.alias_blur_min, self.alias_blur_max))
        self.alias_blur_min.set(0.1)
        self.alias_blur_min.grid(row=2, column=1, sticky='w')
        self.alias_blur_max = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=100,
                                       command=lambda v: self.helper.slider_pair_link(self.alias_blur_min, self.alias_blur_max))
        self.alias_blur_max.set(0.5)
        self.alias_blur_max.grid(row=2, column=2, sticky='w')
        self.Defocus_probability_label, self.Defocus_probability = self.create_probability_control(
            frame, row=3, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())

    def build_ZoomBlur_group(self, frame):
        self.use_ZoomBlur = tk.BooleanVar(value=False)
        tk.Checkbutton(frame, text="启用", variable=self.use_ZoomBlur, command=self.helper.update_preview).grid(row=0, column=0, sticky='w')
        tk.Label(frame, text="模糊最大因子的范围:").grid(row=1, column=0, sticky='e')
        self.max_factor_min = tk.Scale(frame, from_=1.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, length=100,
                                       command=lambda v: self.helper.slider_pair_link(self.max_factor_min, self.max_factor_max))
        self.max_factor_min.set(1.0)
        self.max_factor_min.grid(row=1, column=1, sticky='w')
        self.max_factor_max = tk.Scale(frame, from_=1.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, length=100,
                                       command=lambda v: self.helper.slider_pair_link(self.max_factor_min, self.max_factor_max))
        self.max_factor_max.set(1.31)
        self.max_factor_max.grid(row=1, column=2, sticky='w')
        tk.Label(frame, text="步长参数:").grid(row=2, column=0, sticky='e')
        self.astep_factor_min = tk.Scale(frame, from_=0.01, to=0.1, resolution=0.01, orient=tk.HORIZONTAL, length=100,
                                         command=lambda v: self.helper.slider_pair_link(self.astep_factor_min, self.astep_factor_max))
        self.astep_factor_min.set(0.01)
        self.astep_factor_min.grid(row=2, column=1, sticky='w')
        self.astep_factor_max = tk.Scale(frame, from_=0.01, to=0.1, resolution=0.01, orient=tk.HORIZONTAL, length=100,
                                         command=lambda v: self.helper.slider_pair_link(self.astep_factor_min, self.astep_factor_max))
        self.astep_factor_max.set(0.03)
        self.astep_factor_max.grid(row=2, column=2, sticky='w')
        self.ZoomBlur_probability_label, self.ZoomBlur_probability = self.create_probability_control(
            frame, row=3, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())

    def build_OpticalDistortion_group(self, frame):
        self.use_OpticalDistortion = tk.BooleanVar(value=False)
        tk.Checkbutton(frame, text="启用", variable=self.use_OpticalDistortion, command=self.helper.update_preview).grid(row=0, column=0, sticky='w')
        tk.Label(frame, text="扭曲模型:").grid(row=1, column=0, sticky='e')
        self.OpticalDistortion_mode_var = tk.StringVar(value="camera")
        tk.OptionMenu(frame, self.OpticalDistortion_mode_var, "camera", "fisheye", command=lambda _: self.helper.update_preview()).grid(row=1, column=1, sticky='w')
        tk.Label(frame, text="扭曲系数的范围:").grid(row=2, column=0, sticky='e')
        self.distort_limit_min = tk.Scale(frame, from_=-0.3, to=0.3, resolution=0.01, orient=tk.HORIZONTAL, length=100,
                                          command=lambda v: self.helper.slider_pair_link(self.distort_limit_min, self.distort_limit_max))
        self.distort_limit_min.set(-0.05)
        self.distort_limit_min.grid(row=2, column=1, sticky='w')
        self.distort_limit_max = tk.Scale(frame, from_=-0.3, to=0.3, resolution=0.01, orient=tk.HORIZONTAL, length=100,
                                          command=lambda v: self.helper.slider_pair_link(self.distort_limit_min, self.distort_limit_max))
        self.distort_limit_max.set(0.05)
        self.distort_limit_max.grid(row=2, column=2, sticky='w')
        self.Emboss_OpticalDistortion_label, self.OpticalDistortion_probability = self.create_probability_control(
            frame, row=3, col_label=2, col_scale=3, command=lambda v: self.helper.update_preview())


def run_gui():
    root = tk.Tk()
    app = AugmentationGUI(root)
    root.mainloop()







