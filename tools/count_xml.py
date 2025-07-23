import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('TkAgg')  # 推荐加上
import matplotlib.pyplot as plt

# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from tqdm import tqdm
import pandas as pd  # 新增

def parse_xml(xml_path):
    """读取单个xml文件，返回所有类别列表"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        names = [obj.find('name').text for obj in root.findall('object') if obj.find('name') is not None]
        return names
    except Exception as e:
        print(f"解析文件失败: {xml_path}，错误: {e}")
        return []

def collect_stats(root_dir):
    """遍历目录，统计每个类别数量"""
    overall_counter = Counter()
    folder_counter = defaultdict(Counter)
    xml_file_info = []
    for dirpath, _, filenames in os.walk(root_dir):
        xml_files = [f for f in filenames if f.lower().endswith('.xml')]
        folder_name = os.path.relpath(dirpath, root_dir)
        for xml_file in xml_files:
            xml_path = os.path.join(dirpath, xml_file)
            xml_file_info.append((folder_name, xml_path))
    for folder_name, xml_path in tqdm(xml_file_info, desc="处理XML文件"):
        categories = parse_xml(xml_path)
        overall_counter.update(categories)
        folder_counter[folder_name].update(categories)
    return overall_counter, folder_counter

def plot_hist(counter, title):
    if not counter:
        print(f"没有数据可以绘制：{title}")
        return
    items = list(counter.items())
    items.sort(key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_counters_to_csv(overall_counter, folder_counter, output_dir="统计结果"):
    os.makedirs(output_dir, exist_ok=True)
    # 保存总体统计
    df_total = pd.DataFrame(list(overall_counter.items()), columns=["类别", "数量"])
    df_total.sort_values(by="数量", ascending=False, inplace=True)
    total_csv_path = os.path.join(output_dir, "总体类别统计.csv")
    df_total.to_csv(total_csv_path, index=False, encoding="utf-8-sig")
    print(f"总体类别统计已保存到: {total_csv_path}")

    # 保存每个文件夹的统计
    for folder, counter in folder_counter.items():
        df_folder = pd.DataFrame(list(counter.items()), columns=["类别", "数量"])
        df_folder.sort_values(by="数量", ascending=False, inplace=True)
        safe_folder = folder.replace(os.sep, "_").replace(".", "_")
        folder_csv_path = os.path.join(output_dir, f"类别统计_{safe_folder}.csv")
        df_folder.to_csv(folder_csv_path, index=False, encoding="utf-8-sig")
        print(f"文件夹 {folder} 的类别统计已保存到: {folder_csv_path}")

if __name__ == "__main__":
    root_dir = r"G:\A_Share\datas\ebo\ebo_test"  # 修改为你的xml数据集根目录路径
    overall_counter, folder_counter = collect_stats(root_dir)

    print("总体类别分布：")
    for k, v in overall_counter.items():
        print(f"{k}: {v}")
    plot_hist(overall_counter, "总体类别分布")

    print("\n每个文件夹类别分布：")
    for folder, counter in folder_counter.items():
        print(f"\n文件夹: {folder}")
        for k, v in counter.items():
            print(f"{k}: {v}")

    # 新增：保存csv
    save_counters_to_csv(overall_counter, folder_counter)

