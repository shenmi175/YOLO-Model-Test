import os
import xml.etree.ElementTree as ET
from collections import defaultdict

# 固定类别与ID映射
CATEGORY_MAP = {
    'cat': 1,
    'catface': 3,
    'dog': 2,
    'dogface': 4,
    'face': 6,
    'hand': 5,
    'person': 0,

}

# CATEGORY_MAP = {
#     "footwear" : 0,
# }

def process_directory(input_folder):
    """
    处理指定目录下的所有 XML 文件，生成 YOLO 格式的 txt 标注文件。
    """
    # 收集所有 XML 文件
    xml_files = [f for f in os.listdir(input_folder) if f.endswith('.xml')]

    # 收集当前目录中出现的所有类别
    categories = set()
    for xml_file in xml_files:
        xml_path = os.path.join(input_folder, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                categories.add(name)
        except Exception as e:
            print(f"[警告] 解析 XML 文件 {xml_path} 失败: {e}")

    # 排序后保存到 classes.txt
    classes_path = os.path.join(input_folder, 'classes.txt')
    try:
        with open(classes_path, 'w') as f:
            f.write('\n'.join(sorted(categories)))
    except Exception as e:
        print(f"[错误] 写入 classes.txt 失败: {e}")

    # 初始化类别计数器
    category_count = defaultdict(int)

    # 处理每个 XML 文件
    for xml_file in xml_files:
        base_name = os.path.splitext(xml_file)[0]
        # 支持多种图片扩展名
        img_exts = [".jpg", ".jpeg", ".png", ".bmp"]
        img_path = None
        for ext in img_exts:
            candidate = os.path.join(input_folder, f"{base_name}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        # 检查对应图片是否存在
        if not img_path:
            print(f"[警告] 找不到 {base_name} 的对应图片，跳过.")
            continue

        # 解析 XML 文件
        try:
            xml_path = os.path.join(input_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        except Exception as e:
            print(f"[错误] 解析 XML 文件 {xml_path} 失败: {e}")
            continue

        lines = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CATEGORY_MAP:
                print(f"[警告] 类别 {name} 未在 CATEGORY_MAP 中定义，跳过.")
                continue
            class_id = CATEGORY_MAP[name]
            bndbox = obj.find('bndbox')

            # 解析边界框坐标
            try:
                xmin = max(0, int(bndbox.find('xmin').text))
                ymin = max(0, int(bndbox.find('ymin').text))
                xmax = min(int(bndbox.find('xmax').text), width - 1)
                ymax = min(int(bndbox.find('ymax').text), height - 1)
            except AttributeError as e:
                print(f"[错误] XML 文件 {xml_file} 中缺少边界框信息: {e}")
                continue

            # 计算归一化坐标
            x_center = (xmin + xmax) / (2.0 * width)
            y_center = (ymin + ymax) / (2.0 * height)
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            # 修正坐标超出范围的情况
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            box_width = min(max(box_width, 0.0), 1.0)
            box_height = min(max(box_height, 0.0), 1.0)

            # 构建 YOLO 格式标注行
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
            lines.append(line)

            # 统计类别数量
            category_count[name] += 1

        # 保存标注文件
        txt_path = os.path.join(input_folder, f"{base_name}.txt")
        try:
            with open(txt_path, 'w') as f:
                f.writelines(lines)
        except Exception as e:
            print(f"[错误] 写入标注文件 {txt_path} 失败: {e}")

    # 打印类别统计信息
    print(f"\n【处理完成】{input_folder} 的类别统计:")
    for category in sorted(categories):
        print(f"{category}: {category_count[category]} 个实例")

if __name__ == "__main__":
    # 设置根目录
    root_folder = r'G:\A_Share\datas\merged_xml\feedback\feedback1'  # 修改为你的文件夹路径

    # 遍历根目录下的所有子目录并处理
    print(f"开始遍历根目录: {root_folder}")
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # 检查当前目录是否存在 XML 文件
        has_xml = any(fname.endswith('.xml') for fname in filenames)
        if not has_xml:
            print(f"跳过无 XML 文件的目录: {dirpath}")
            continue

        print(f"\n正在处理目录: {dirpath}")
        process_directory(dirpath)

    print("\n所有目录处理完成.")