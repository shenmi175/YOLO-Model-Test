# import os
# import cv2
# import pandas as pd
# from lxml import etree
# from tqdm import tqdm
#
# # 路径参数
# image_dir = r"G:\A_Share\datas\other_data\cat_dog\Cat\test"
# csv_path = r"G:\A_Share\datas\other_data\cat_dog\Cat\test.csv"
# xml_save_dir = r"G:\A_Share\datas\other_data\cat_dog\Cat\test"
#
# os.makedirs(xml_save_dir, exist_ok=True)
#
# df = pd.read_csv(csv_path, sep=',')
# print("DataFrame 列名:", df.columns)
#
# for idx, row in tqdm(df.iterrows(), total=len(df), desc='生成xml标注'):
#     filename = row['filename']
#     image_path = os.path.join(image_dir, filename)
#     if not os.path.exists(image_path):
#         print(f"图片不存在：{image_path}")
#         continue
#
#     # 提取所有关键点
#     coords = row.values[1:].astype(float)
#     xs = coords[::2]
#     ys = coords[1::2]
#     xmin = int(min(xs))
#     xmax = int(max(xs))
#     ymin = int(min(ys))
#     ymax = int(max(ys))
#
#     # 读取图片获取尺寸
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"无法读取图片：{image_path}")
#         continue
#     h, w = img.shape[:2]
#
#     w_box = xmax - xmin
#     h_box = ymax - ymin
#
#     # 左右各扩展5%，下方扩展20%，上方不变
#     xmin_new = max(0, int(xmin - w_box * 0.05))
#     xmax_new = min(w - 1, int(xmax + w_box * 0.05))
#     ymin_new = ymin
#     ymax_new = min(h - 1, int(ymax + h_box * 0.2))
#
#     # VOC XML内容
#     annotation = etree.Element("annotation")
#     etree.SubElement(annotation, "folder").text = os.path.basename(image_dir)
#     etree.SubElement(annotation, "filename").text = filename
#     size = etree.SubElement(annotation, "size")
#     etree.SubElement(size, "width").text = str(w)
#     etree.SubElement(size, "height").text = str(h)
#     etree.SubElement(size, "depth").text = "3"
#
#     obj = etree.SubElement(annotation, "object")
#     etree.SubElement(obj, "name").text = "catface"
#     etree.SubElement(obj, "pose").text = "Unspecified"
#     etree.SubElement(obj, "truncated").text = "0"
#     etree.SubElement(obj, "difficult").text = "0"
#     bndbox = etree.SubElement(obj, "bndbox")
#     etree.SubElement(bndbox, "xmin").text = str(xmin_new)
#     etree.SubElement(bndbox, "ymin").text = str(ymin_new)
#     etree.SubElement(bndbox, "xmax").text = str(xmax_new)
#     etree.SubElement(bndbox, "ymax").text = str(ymax_new)
#
#     # 保存xml
#     xml_str = etree.tostring(annotation, pretty_print=True, encoding="utf-8")
#     xml_path = os.path.join(xml_save_dir, filename.replace('.jpg', '.xml'))
#     with open(xml_path, 'wb') as f:
#         f.write(xml_str)


import os
import shutil
from tqdm import tqdm


def find_all_xmls(xml_root):
    xml_files = []
    for root, _, files in os.walk(xml_root):
        for file in files:
            if file.endswith('.jpg.xml'):
                xml_files.append(os.path.join(root, file))
    return xml_files


def find_pic_dir(base_pic_dir, sub_folder_name):
    # 遍历 base_pic_dir 下所有目录，找到子目录名和 sub_folder_name 相同的路径
    for root, dirs, _ in os.walk(base_pic_dir):
        if os.path.basename(root) == sub_folder_name:
            return root
    return None


def move_and_rename_xml(xml_file, pic_base_dir, annotation_root):
    # 解析结构
    xml_dir = os.path.dirname(xml_file)
    folder_name = os.path.basename(xml_dir)
    # 新文件名（去掉.jpg）
    old_name = os.path.basename(xml_file)
    new_name = old_name.replace('.jpg.xml', '.xml')
    # 目标图片目录
    target_dir = find_pic_dir(pic_base_dir, folder_name)
    if target_dir is None:
        print(f"[警告] 未找到图片目录：{folder_name}")
        return False
    # 移动并重命名
    src = xml_file
    dst = os.path.join(target_dir, new_name)
    shutil.move(src, dst)
    return True


def main():
    pic_base_dir = r'G:\A_Share\datas\other_data\cat_dog\dog\low-resolution'
    annotation_root = r'G:\A_Share\datas\other_data\cat_dog\dog\Low-Annotations'

    xml_files = find_all_xmls(annotation_root)
    if not xml_files:
        print("没有找到任何 .jpg.xml 文件")
        return

    success, fail = 0, 0
    with tqdm(total=len(xml_files), desc="处理进度") as pbar:
        for xml_file in xml_files:
            try:
                res = move_and_rename_xml(xml_file, pic_base_dir, annotation_root)
                if res:
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                print(f"处理 {xml_file} 时出错: {e}")
                fail += 1
            pbar.update(1)
    print(f"完成！成功处理 {success} 个文件，失败 {fail} 个文件。")


if __name__ == '__main__':
    main()



