# src/augment/aug.py
import albumentations as A
import cv2

import random

def _apply(transform, image, bboxes=None, labels=None, p=0.5):
    """Apply an albumentations transform with optional bbox support.

    Returns a tuple ``(image, bboxes, applied)`` where ``applied`` indicates
    whether the transform was actually executed based on ``p``.
    """
    if random.random() >= p:
        # Transformation skipped
        return image, bboxes, False
    if bboxes is not None:
        compose = A.Compose([transform], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
        data = compose(image=image, bboxes=bboxes, labels=labels or [0] * len(bboxes))
        return data["image"], data["bboxes"], True
    data = A.Compose([transform])(image=image)
    return data["image"], data["bboxes"], True

def apply_motion_blur(image, blur_limit=13, angle=0, direction=0, allow_shifted=True, p=0.5, bboxes=None, labels=None):
    """
    应用运动模糊增强
    """
    transform = A.MotionBlur(
        blur_limit=blur_limit,
        allow_shifted=allow_shifted,
        angle_range=(angle, angle),
        direction_range=(direction, direction),
        p=1.0
    )
    return _apply(transform, image, bboxes, labels, p)

def apply_AdditiveNoise(image, noise_type="gaussian", spatial_mode="shared", mean_range=(0,0), std_range=(0.05,0.15), approximation=1, p=0.5, bboxes=None, labels=None):
    """
    应用添加噪声
    """
    noise_params = {
        "mean_range":mean_range,
        "std_range":std_range,
    }

    transform = A.AdditiveNoise(
        noise_type=noise_type,
        spatial_mode=spatial_mode,
        noise_params=noise_params,
        approximation=approximation,
        p=1.0,

    )
    return _apply(transform, image, bboxes, labels, p)

def apply_ShotNoise(image, scale_range = (0.05,0.2) ,p=0.5, bboxes=None, labels=None):
    """
    应用散粒噪声
    """
    transform = A.ShotNoise(
        scale_range = scale_range,
        p=p,
    )
    return _apply(transform, image, bboxes, labels, p)

def apply_ToGray(image, method="weighted_average", p=0.5, bboxes=None, labels=None):
    """
    应用灰度变换
    """
    transform = A.ToGray(
        method = method,
        p = 1.0,
    )

    return _apply(transform, image, bboxes, labels, p)

def apply_PlanckianJitter(image, mode="blackbody", sampling_method="uniform", p=0.5, bboxes=None, labels=None):
    transform = A.PlanckianJitter(
        mode=mode,
        sampling_method=sampling_method,
        p=1.0,
    )
    return _apply(transform, image, bboxes, labels, p)


def apply_Emboss(image, alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5, bboxes=None, labels=None):
    "应用浮雕效果"
    transform = A.Emboss(
        alpha=alpha,
        strength=strength,
        p=1.0,
    )
    return _apply(transform, image, bboxes, labels, p)


def apply_ISONoise(image, color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5, bboxes=None, labels=None):
    "应用相机传感器噪声"
    """
    color_shift:
    改变色彩色调的范围。
    值应在[0, 1]范围内，其中 1 表示完整的 360°色调旋转。
    默认值：(0.01, 0.05)
    
    intensity:
    噪声强度的范围。
    更高的值会增加颜色和亮度噪声的强度。
    默认值：(0.1, 0.5)
    """
    transform = A.ISONoise(
        color_shift=color_shift,
        intensity=intensity,
        p=1.0,
    )
    return _apply(transform, image, bboxes, labels, p)


def apply_HueSaturationValue(image, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20), p=0.5,
                             bboxes=None, labels=None):
    "应用改变色调、饱和度和明度"
    """
    改变色调的范围。
    如果提供一个单一的浮点数值，范围将是 (-hue_shift_limit, hue_shift_limit)。
    值应在 [-180, 180] 范围内。默认：(-20, 20)。
    
    改变饱和度的范围。
    如果提供一个单一的浮点值，范围将是 (-sat_shift_limit, sat_shift_limit)。
    值应在 [-255, 255] 范围内。默认：(-30, 30)。
    
    改变值（亮度）的范围。
    如果提供一个单一的浮点值，范围将是 (-val_shift_limit, val_shift_limit)。
    值应在 [-255, 255] 范围内。默认：(-20, 20)。
    """
    transform = A.HueSaturationValue(
        hue_shift_limit=hue_shift_limit,
        sat_shift_limit=sat_shift_limit,
        val_shift_limit=val_shift_limit,
        p=1.0,
    )
    return _apply(transform, image, bboxes, labels, p)

def apply_Illumination(image, Illumination_mode="linear",intensity_range=(0.01, 0.2),effect_type="both",
                       angle_range=(0, 360), center_range=(0.1, 0.9), sigma_range=(0.2, 1), p=0.5, bboxes=None, labels=None):
    "应用光照效果"
    """
    照明模式类型：
    - 'linear': 在图像上创建平滑的渐变，
    模拟方向性光照，如阳光
    通过窗户
    - 'corner': 从任意角落应用渐变，
               模拟从角落发出的光源
    - 'gaussian': 创建圆形聚光灯效果，
                 模拟局部光源
    默认值：'linear'
    ===========================
    效果强度的范围。
    0.01 到 0.2 之间的值：
    - 0.01-0.05：轻微的光照变化
    - 0.05-0.1：适度的光照效果
    - 0.1-0.2：强烈的光照效果
    默认值：(0.01, 0.2)
    ===========================
    光照变化的类型：
    - 'brighten': 仅增加光照（如同聚光灯）
    - 'darken': 仅移除光线（如同阴影）
    - 'both': 随机选择增亮或减暗
    默认: 'both'
    ===========================
    渐变角度的度数范围。
    控制线性渐变的方向：
    - 0°：从左到右
    - 90°：从上到下
    - 180°：从右到左
    - 270°: 从下到上
    仅用于'linear'模式。
    默认值: (0, 360)
    ===========================
    聚光灯位置的范围。
    0 到 1 之间的值表示相对位置：
    - (0, 0): 左上角
    - (1, 1): 右下角
    - (0.5, 0.5): 图像中心
    仅用于'gaussian'模式。
    默认值: (0.1, 0.9)
    ===========================
    聚光灯大小的范围。
    0.2 到 1.0 之间的值：
    - 0.2：小而集中的聚光灯
    - 0.5：中等大小的光照区域
    - 1.0: 广泛柔和的照明
    仅用于 'gaussian' 模式。
    默认: (0.2, 1.0)
    """
    transform = A.Illumination(
        mode=Illumination_mode,
        intensity_range=intensity_range,
        effect_type=effect_type,
        angle_range=angle_range,
        center_range=center_range,
        sigma_range=sigma_range,
        p=1.0,
    )
    return _apply(transform, image, bboxes, labels, p)

def apply_Defocus(image, radius=(3, 10), alias_blur=(0.1, 0.5), p=0.5, bboxes=None, labels=None):
    "应用失焦模糊"
    """
    模糊半径的范围。
    如果提供一个整数，范围将是[1, 半径]。
    更大的值会产生更强的模糊效果。
    默认值：(3, 10)
    ===========================
    高斯模糊的标准偏差范围
    应用于主要散焦模糊之后。这有助于减少锯齿伪影。
    如果提供一个浮点数，范围将是(0, alias_blur)。
    更大的值会产生更平滑、更模糊的效果。
    默认值：(0.1, 0.5)
    """
    transform = A.Defocus(
        radius=radius,
        alias_blur=alias_blur,
        p=1.0,
    )
    return _apply(transform, image, bboxes, labels, p)

def apply_ZoomBlur(image, max_factor=(1, 1.31), step_factor=(0.01, 0.03), p=0.5, bboxes=None, labels=None):
    "应用缩放模糊"
    """
    模糊最大因子的范围。
    如果 max_factor 是一个浮点数，范围将是 (1, limit)。默认值：(1, 1.31)。
    所有 max_factor 值都应大于 1。
    ===========================
    如果使用单个浮点数作为 np.arange 的步长参数。
    如果使用浮点数元组作为步长参数，其范围将在 `[step_factor[0], step_factor[1})`。默认值：(0.01, 0.03)。
    所有 step_factor 值应为正数。
    """
    transform = A.ZoomBlur(
        max_factor=max_factor,
        step_factor=step_factor,
        p=1.0,
    )
    return _apply(transform, image, bboxes, labels, p)

def apply_OpticalDistortion(image, distort_limit=(0.5, 0.5), interpolation=cv2.INTER_LINEAR,
                            mode="camera",
                            border_mode=cv2.BORDER_CONSTANT,
                            p=0.5, bboxes=None, labels=None):
    "应用光学扭曲"
    """
    扭曲系数的范围。
    对于相机模型：推荐范围为(-0.05, 0.05)
    对于鱼眼模型：推荐范围为(-0.3, 0.3)
    默认值：(-0.05, 0.05)
    ===========================
    图像变换所使用的插值方法。
    应为以下之一：cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
    cv2.INTER_AREA, cv2.INTER_LANCZOS4. 默认值：cv2.INTER_LINEAR.
    ===========================
    要使用的扭曲模型：
    - 'camera': 原始相机矩阵模型
    - 'fisheye': 鱼眼镜头模型
    默认值：'camera'
   =========================== 
    """
    transform = A.OpticalDistortion(
        distort_limit=distort_limit,
        interpolation=interpolation,
        mode=mode,
        border_mode=border_mode,
        p=1.0,
    )
    return _apply(transform, image, bboxes, labels, p)