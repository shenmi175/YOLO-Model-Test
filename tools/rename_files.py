import os
from tqdm import tqdm


def is_image(filename):
    IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp', 'jfif')
    return filename.lower().endswith(IMAGE_EXTS)


def find_label_file(image_path, label_types):
    """在同目录下查找与图片同名的标注文件，返回所有存在的标注文件路径列表。"""
    basename, _ = os.path.splitext(image_path)
    label_files = []
    for ext in label_types:
        label_path = basename + ext
        if os.path.exists(label_path):
            label_files.append(label_path)
    return label_files


def rename_images_and_labels_in_folder(folder_path, prefix, label_types=('.txt', '.xml')):
    files = sorted([f for f in os.listdir(folder_path) if is_image(f)])
    with tqdm(total=len(files), desc=f"Processing {os.path.basename(folder_path)}", leave=False) as pbar:
        for idx, filename in enumerate(files, start=1):
            # 新文件名
            new_name = f"{prefix}{idx}.jpg"
            src = os.path.join(folder_path, filename)
            dst = os.path.join(folder_path, new_name)

            # 先处理图片
            if src != dst and not os.path.exists(dst):
                os.rename(src, dst)
            elif src != dst:
                dst = os.path.join(folder_path, f"{prefix}{idx}_dup.jpg")
                os.rename(src, dst)

            # 查找和重命名标注文件
            label_files = find_label_file(src, label_types)
            for label_file in label_files:
                old_ext = os.path.splitext(label_file)[1]
                new_label_name = os.path.splitext(dst)[0] + old_ext
                if label_file != new_label_name and not os.path.exists(new_label_name):
                    os.rename(label_file, new_label_name)
                elif label_file != new_label_name:
                    dup_label_name = os.path.splitext(dst)[0] + "_dup" + old_ext
                    os.rename(label_file, dup_label_name)
            pbar.update(1)


def walk_and_rename_images_labels(root_dir, mode='custom', custom_prefix=None, label_types=('.txt', '.xml')):
    all_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(is_image(f) for f in filenames):
            all_dirs.append(dirpath)
    with tqdm(total=len(all_dirs), desc="Folders", position=0) as folder_bar:
        for folder in all_dirs:
            if mode == 'custom':
                prefix = custom_prefix
            elif mode == 'folder':
                prefix = os.path.basename(folder)
            else:
                raise ValueError('Invalid mode')
            rename_images_and_labels_in_folder(folder, prefix, label_types=label_types)
            folder_bar.update(1)


if __name__ == '__main__':
    root_dir = r'G:\A_Share\datas\ebo\ebo_test\cat_block'  # 修改为你的目录路径

    print("请选择命名方式：")
    print("1. 所有图片和标注用同一个自定义前缀（如 cat_1.jpg, cat_1.txt ...）")
    print("2. 每个文件夹用自己名字做前缀（如 文件夹名1.jpg, 文件夹名1.txt ...）")
    choice = input("输入 1 或 2 进行选择: ").strip()

    if choice == '1':
        custom_prefix = input("请输入你想用的前缀（比如 cat_ 或 img_ ）: ").strip()
        walk_and_rename_images_labels(root_dir, mode='custom', custom_prefix=custom_prefix)
    elif choice == '2':
        walk_and_rename_images_labels(root_dir, mode='folder')
    else:
        print("输入错误，请重新运行并输入 1 或 2")

    print("Done.")



