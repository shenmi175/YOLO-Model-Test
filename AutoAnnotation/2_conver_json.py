import os
import json
from tqdm import tqdm
from PIL import Image

# 支持的图片后缀
IMG_SUFFIX = ['.jpg', '.jpeg', '.png', '.bmp']

def find_json_files(root_dir):
    """递归查找所有json文件"""
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def get_image_path(json_path, candidate_data, folder_files):
    # 1. 优先找同名图片（和json文件同名，仅扩展名不同）
    json_base = os.path.splitext(os.path.basename(json_path))[0]
    for suf in IMG_SUFFIX:
        candidate_img = os.path.join(os.path.dirname(json_path), json_base + suf)
        if os.path.exists(candidate_img):
            return os.path.basename(candidate_img), candidate_img
    # 2. 如果json里有imagePath字段，用该字段
    if 'imagePath' in candidate_data:
        img_path = os.path.join(os.path.dirname(json_path), candidate_data['imagePath'])
        if os.path.exists(img_path):
            return candidate_data['imagePath'], img_path
    # 3. 最后，暴力查找目录下的所有图片
    for file in folder_files:
        if os.path.splitext(file)[1].lower() in IMG_SUFFIX:
            return file, os.path.join(os.path.dirname(json_path), file)
    return None, None

def convert_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"跳过无法解析的JSON: {json_path}，错误: {e}")
            return

    # 找到当前目录下的所有文件（后续查找图片名用）
    folder_files = os.listdir(os.path.dirname(json_path))

    # 获取图片名和路径
    image_name, image_full_path = get_image_path(json_path, data, folder_files)
    if image_name is None or image_full_path is None:
        print(f"未找到图片文件，跳过: {json_path}")
        return

    # 获取图片分辨率
    try:
        with Image.open(image_full_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"读取图片失败，跳过: {image_full_path}，错误: {e}")
        return

    # 检查 candidate 字段
    if 'candidate' not in data or not isinstance(data['candidate'], list):
        print(f"没有 candidate 字段，跳过: {json_path}")
        return

    # 转换 shapes
    shapes = []
    for kp in data['candidate']:
        if len(kp) >= 2:
            x, y = kp[0], kp[1]
            label = f"keypoint{int(kp[3])+1}" if len(kp) > 3 else "keypoint"
            shapes.append({
                "label": label,
                "points": [[x, y]],
                "group_id": None,
                "shape_type": "point",
                "flags": {}
            })

    # 构建 Labelme 格式 json
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    # 保存
    out_json = out_json = json_path

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

def main(root_dir):
    json_files = find_json_files(root_dir)
    print(f"共找到 {len(json_files)} 个JSON文件")
    for json_path in tqdm(json_files, desc="转换中"):
        convert_json_file(json_path)

if __name__ == '__main__':
    # TODO: 修改为你的根目录
    root_dir = r"G:\A_Share\datas\other_data\panda\pose\1m\dzy_20250804_1m"
    main(root_dir)
