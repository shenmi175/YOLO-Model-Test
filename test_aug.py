import os
from tqdm import tqdm

# 图片和标签的文件夹路径
image_dir = r'G:\A_Share\datas\ebo\ebo_train\images\train'
label_dir = r'G:\A_Share\datas\ebo\ebo_train\labels\train'

# 找到所有首字母为n的图片文件
image_files = [f for f in os.listdir(image_dir) if f.lower().startswith('n')]

for filename in tqdm(image_files, desc='正在删除文件'):
    # 构造完整图片路径
    img_path = os.path.join(image_dir, filename)
    # 构造同名txt路径
    txt_name = os.path.splitext(filename)[0] + '.txt'
    txt_path = os.path.join(label_dir, txt_name)

    # 删除图片
    try:
        os.remove(img_path)
        # tqdm.write可以保证进度条不乱
        tqdm.write(f"删除图片: {img_path}")
    except FileNotFoundError:
        tqdm.write(f"图片未找到: {img_path}")

    # 删除txt
    try:
        os.remove(txt_path)
        tqdm.write(f"删除txt: {txt_path}")
    except FileNotFoundError:
        tqdm.write(f"txt未找到: {txt_path}")

