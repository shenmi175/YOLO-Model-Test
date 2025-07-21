import os
import cv2
import pandas as pd
from lxml import etree
from tqdm import tqdm

# 路径参数
image_dir = r"G:\A_Share\datas\other_data\cat_dog\Cat\test"
csv_path = r"G:\A_Share\datas\other_data\cat_dog\Cat\test.csv"
xml_save_dir = r"G:\A_Share\datas\other_data\cat_dog\Cat\test"

os.makedirs(xml_save_dir, exist_ok=True)

df = pd.read_csv(csv_path, sep=',')
print("DataFrame 列名:", df.columns)

for idx, row in tqdm(df.iterrows(), total=len(df), desc='生成xml标注'):
    filename = row['filename']
    image_path = os.path.join(image_dir, filename)
    if not os.path.exists(image_path):
        print(f"图片不存在：{image_path}")
        continue

    # 提取所有关键点
    coords = row.values[1:].astype(float)
    xs = coords[::2]
    ys = coords[1::2]
    xmin = int(min(xs))
    xmax = int(max(xs))
    ymin = int(min(ys))
    ymax = int(max(ys))

    # 读取图片获取尺寸
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片：{image_path}")
        continue
    h, w = img.shape[:2]

    w_box = xmax - xmin
    h_box = ymax - ymin

    # 左右各扩展5%，下方扩展20%，上方不变
    xmin_new = max(0, int(xmin - w_box * 0.05))
    xmax_new = min(w - 1, int(xmax + w_box * 0.05))
    ymin_new = ymin
    ymax_new = min(h - 1, int(ymax + h_box * 0.2))

    # VOC XML内容
    annotation = etree.Element("annotation")
    etree.SubElement(annotation, "folder").text = os.path.basename(image_dir)
    etree.SubElement(annotation, "filename").text = filename
    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = str(w)
    etree.SubElement(size, "height").text = str(h)
    etree.SubElement(size, "depth").text = "3"

    obj = etree.SubElement(annotation, "object")
    etree.SubElement(obj, "name").text = "catface"
    etree.SubElement(obj, "pose").text = "Unspecified"
    etree.SubElement(obj, "truncated").text = "0"
    etree.SubElement(obj, "difficult").text = "0"
    bndbox = etree.SubElement(obj, "bndbox")
    etree.SubElement(bndbox, "xmin").text = str(xmin_new)
    etree.SubElement(bndbox, "ymin").text = str(ymin_new)
    etree.SubElement(bndbox, "xmax").text = str(xmax_new)
    etree.SubElement(bndbox, "ymax").text = str(ymax_new)

    # 保存xml
    xml_str = etree.tostring(annotation, pretty_print=True, encoding="utf-8")
    xml_path = os.path.join(xml_save_dir, filename.replace('.jpg', '.xml'))
    with open(xml_path, 'wb') as f:
        f.write(xml_str)


# import os
# import xml.etree.ElementTree as ET
# import glob
#
# def convert_to_voc(src_xml_path, save_dir):
#     # 读取原始xml
#     tree = ET.parse(src_xml_path)
#     root = tree.getroot()
#
#     # 获取图片名和folder等信息
#     folder = root.find('folder').text
#     filename = root.find('filename').text
#     size = root.find('size')
#     width = size.find('width').text
#     height = size.find('height').text
#     depth = size.find('depth').text
#
#     # 找到headbndbox和bodybndbox
#     object_node = root.find('object')
#     head_box = object_node.find('headbndbox')
#     body_box = object_node.find('bodybndbox')
#
#     # 新建VOC xml
#     annotation = ET.Element('annotation')
#     ET.SubElement(annotation, 'folder').text = folder
#     ET.SubElement(annotation, 'filename').text = filename
#     source = ET.SubElement(annotation, 'source')
#     ET.SubElement(source, 'database').text = "THU-Dogs database"
#     size_ele = ET.SubElement(annotation, 'size')
#     ET.SubElement(size_ele, 'width').text = width
#     ET.SubElement(size_ele, 'height').text = height
#     ET.SubElement(size_ele, 'depth').text = depth
#     ET.SubElement(annotation, 'segment').text = '0'
#
#     # 添加dogface（headbndbox）
#     if head_box is not None:
#         obj = ET.SubElement(annotation, 'object')
#         ET.SubElement(obj, 'name').text = 'dogface'
#         ET.SubElement(obj, 'pose').text = 'Unspecified'
#         ET.SubElement(obj, 'truncated').text = '0'
#         ET.SubElement(obj, 'difficult').text = '0'
#         bndbox = ET.SubElement(obj, 'bndbox')
#         for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
#             ET.SubElement(bndbox, tag).text = head_box.find(tag).text
#
#     # 添加dog（bodybndbox）
#     if body_box is not None:
#         obj = ET.SubElement(annotation, 'object')
#         ET.SubElement(obj, 'name').text = 'dog'
#         ET.SubElement(obj, 'pose').text = 'Unspecified'
#         ET.SubElement(obj, 'truncated').text = '0'
#         ET.SubElement(obj, 'difficult').text = '0'
#         bndbox = ET.SubElement(obj, 'bndbox')
#         for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
#             ET.SubElement(bndbox, tag).text = body_box.find(tag).text
#
#     # 保存为VOC格式xml，名字为xxx.xml
#     base_name = os.path.splitext(filename)[0]
#     save_path = os.path.join(save_dir, base_name + ".xml")
#     ET.ElementTree(annotation).write(save_path, encoding='utf-8', xml_declaration=True)
#     print("转换完成：", save_path)
#
# # -----------------------
# # 批量处理范例
# src_dir = r"G:\A_Share\datas\other_data\cat_dog\dog\high-resolution\200-n000008-Airedale"
# save_dir = r"G:\A_Share\datas\other_data\cat_dog\dog\high-resolution\200-n000008-Airedale"
# os.makedirs(save_dir, exist_ok=True)
# for xml_file in glob.glob(os.path.join(src_dir, "*.xml")):
#     convert_to_voc(xml_file, save_dir)


