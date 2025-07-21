import os
import hashlib
from tqdm import tqdm
from collections import defaultdict

def get_image_files(directory, extensions={'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                image_files.append(os.path.join(root, file))
    return image_files

def calculate_md5(filepath, chunk_size=8192):
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()

def delete_file_with_log(filepath):
    try:
        os.remove(filepath)
        print(f"    已删除: {filepath}")
        return True
    except Exception as e:
        print(f"    删除失败: {filepath}, 错误: {e}")
        return False

def check_duplicates(directory, remove_duplicates=False):
    image_files = get_image_files(directory)
    md5_dict = defaultdict(list)
    for filepath in tqdm(image_files, desc=f"扫描 {directory}"):
        try:
            md5 = calculate_md5(filepath)
            md5_dict[md5].append(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    duplicates = {md5: files for md5, files in md5_dict.items() if len(files) > 1}
    print(f"\n总共有 {len(duplicates)} 组图片重复。")
    total_deleted = 0
    total_xml_deleted = 0
    if duplicates:
        for i, (md5, files) in enumerate(duplicates.items(), 1):
            print(f"\n第 {i} 组（{len(files)} 张图片）：")
            for idx, f in enumerate(files):
                print(f"  {f}")
                if remove_duplicates and idx > 0:
                    # 删除图片
                    if delete_file_with_log(f):
                        total_deleted += 1
                    # 删除对应的 xml 文件（同目录、同名，后缀 .xml）
                    xml_file = os.path.splitext(f)[0] + ".xml"
                    if os.path.exists(xml_file):
                        if delete_file_with_log(xml_file):
                            total_xml_deleted += 1
    else:
        print("没有找到重复图片。")
    if remove_duplicates:
        print(f"\n共删除了 {total_deleted} 张重复图片，以及 {total_xml_deleted} 个对应 xml 文件（每组仅保留一张图片及xml）。")
    return duplicates

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="检查一个文件夹下是否有重复图片，或比较两个文件夹下相同图片，可选删除重复图片和配套xml")
    parser.add_argument("dir1", help="第一个文件夹路径")
    parser.add_argument("dir2", nargs="?", default=None, help="（可选）第二个文件夹路径")
    parser.add_argument("--remove_duplicates", type=lambda x: str(x).lower()=='true', default=False, help="是否删除重复图片（只保留一张），True/False")
    args = parser.parse_args()

    if args.dir2:
        # 比较两个目录
        def get_md5_set(directory):
            image_files = get_image_files(directory)
            md5_set = set()
            for filepath in tqdm(image_files, desc=f"扫描 {directory}"):
                try:
                    md5 = calculate_md5(filepath)
                    md5_set.add(md5)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
            return md5_set

        md5_set1 = get_md5_set(args.dir1)
        md5_set2 = get_md5_set(args.dir2)
        same_images = md5_set1.intersection(md5_set2)
        print(f"\n一样的图片总数: {len(same_images)}")
    else:
        # 检查一个目录下的重复图片，必要时删除图片和xml
        check_duplicates(args.dir1, remove_duplicates=args.remove_duplicates)





# 使用方式
# python check_images.py /path/to/folder1 /path/to/folder2
# 传入“-remove_duplicates True”时删除一样的图片（仅保留一张）


#     dir1 = r"G:\A_Share\datas\other_data\cat_dog\Cat"
#     dir2 = r"G:\A_Share\datas\merged_xml\val\coco\The_Oxford-IIIT_Pet_Dataset_cat"

 # python .\demo2.py G:\A_Share\datas\other_data\cat_dog\dog G:\A_Share\datas\merged_xml\The_Oxford-IIIT_Pet_Dataset_dog



