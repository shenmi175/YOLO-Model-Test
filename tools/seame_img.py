import os
import hashlib
from PIL import Image
from tqdm import tqdm
from tkinter import Tk, Label, Button, filedialog, messagebox, Listbox, Scrollbar, END, StringVar, Entry, Frame, BooleanVar, Checkbutton
import threading

def is_image(filename):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp')
    return filename.lower().endswith(exts)

def file_md5(path):
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def collect_image_hashes(folder):
    img_hash_map = {}
    all_img_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if is_image(f):
                fp = os.path.join(root, f)
                all_img_files.append(fp)
    for fp in tqdm(all_img_files, desc=f"扫描 {folder} 图片", ncols=120):
        try:
            h = file_md5(fp)
            img_hash_map.setdefault(h, []).append(fp)
        except Exception as e:
            print(f"Cannot hash {fp}: {e}")
    return img_hash_map

def find_common_hashes(map1, map2):
    common = set(map1.keys()) & set(map2.keys())
    files1, files2 = [], []
    for h in common:
        files1.extend(map1[h])
        files2.extend(map2[h])
    return common, files1, files2

def delete_files(file_list, label_dir=None, xml_dir=None):
    count, failed = 0, 0
    for f in tqdm(file_list, desc="删除图片及标注", ncols=120):
        base = os.path.splitext(os.path.basename(f))[0]
        txt_path = os.path.join(label_dir, base + '.txt') if label_dir else os.path.splitext(f)[0] + '.txt'
        xml_path = os.path.join(xml_dir, base + '.xml') if xml_dir else os.path.splitext(f)[0] + '.xml'
        try:
            os.remove(f)
            count += 1
        except Exception:
            failed += 1
        for ann_path in [txt_path, xml_path]:
            if os.path.exists(ann_path):
                try:
                    os.remove(ann_path)
                except:
                    pass
    return count, failed

class PairCleanFrame(Frame):
    def __init__(self, master):
        super().__init__(master, bd=2, relief='groove')
        Label(self, text="配对清理（清理孤立图片/txt/xml）", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=4, pady=3, sticky='w')
        Label(self, text="图片目录:").grid(row=1, column=0, sticky='w')
        self.img_dir_var = StringVar()
        Entry(self, textvariable=self.img_dir_var, width=40).grid(row=1, column=1, sticky='w')
        Button(self, text="选择", command=self.select_img_dir).grid(row=1, column=2, sticky='w')
        Label(self, text="txt标注目录:").grid(row=2, column=0, sticky='w')
        self.txt_dir_var = StringVar()
        Entry(self, textvariable=self.txt_dir_var, width=40).grid(row=2, column=1, sticky='w')
        Button(self, text="选择", command=self.select_txt_dir).grid(row=2, column=2, sticky='w')
        Label(self, text="xml标注目录:").grid(row=3, column=0, sticky='w')
        self.xml_dir_var = StringVar()
        Entry(self, textvariable=self.xml_dir_var, width=40).grid(row=3, column=1, sticky='w')
        Button(self, text="选择", command=self.select_xml_dir).grid(row=3, column=2, sticky='w')
        self.clean_btn = Button(self, text="执行配对清理", command=self.thread_pair_clean)
        self.clean_btn.grid(row=4, column=1, pady=7, sticky='w')
        self.status = Label(self, text="", fg='blue')
        self.status.grid(row=5, column=0, columnspan=4, sticky='w')

    def select_img_dir(self):
        folder = filedialog.askdirectory(title="选择图片目录")
        if folder:
            self.img_dir_var.set(folder)

    def select_txt_dir(self):
        folder = filedialog.askdirectory(title="选择txt标注目录")
        if folder:
            self.txt_dir_var.set(folder)

    def select_xml_dir(self):
        folder = filedialog.askdirectory(title="选择xml标注目录")
        if folder:
            self.xml_dir_var.set(folder)

    def thread_pair_clean(self):
        self.status.config(text="正在执行配对清理，请稍候...")
        self.clean_btn.config(state='disabled')
        threading.Thread(target=self.pair_clean_action, daemon=True).start()

    def pair_clean_action(self):
        img_dir = self.img_dir_var.get().strip()
        txt_dir = self.txt_dir_var.get().strip() or None
        xml_dir = self.xml_dir_var.get().strip() or None
        img_set = set()
        txt_set = set()
        xml_set = set()
        # 图片
        for root, _, files in os.walk(img_dir):
            for f in files:
                if is_image(f):
                    img_set.add(os.path.splitext(f)[0])
        # txt
        if txt_dir:
            for root, _, files in os.walk(txt_dir):
                for f in files:
                    if f.endswith('.txt'):
                        txt_set.add(os.path.splitext(f)[0])
        # xml
        if xml_dir:
            for root, _, files in os.walk(xml_dir):
                for f in files:
                    if f.endswith('.xml'):
                        xml_set.add(os.path.splitext(f)[0])
        # 孤立图片
        to_del_img = img_set - (txt_set | xml_set)
        # 孤立txt
        to_del_txt = txt_set - img_set
        # 孤立xml
        to_del_xml = xml_set - img_set
        nimg, ntxt, nxml = 0, 0, 0
        # 删除图片
        for name in to_del_img:
            for root, _, files in os.walk(img_dir):
                for f in files:
                    if os.path.splitext(f)[0] == name and is_image(f):
                        try:
                            os.remove(os.path.join(root, f))
                            nimg += 1
                        except Exception as e:
                            print(f"无法删除图片{name}: {e}")
        # 删除txt
        if txt_dir:
            for name in to_del_txt:
                for root, _, files in os.walk(txt_dir):
                    for f in files:
                        if os.path.splitext(f)[0] == name and f.endswith('.txt'):
                            try:
                                os.remove(os.path.join(root, f))
                                ntxt += 1
                            except Exception as e:
                                print(f"无法删除txt{name}: {e}")
        # 删除xml
        if xml_dir:
            for name in to_del_xml:
                for root, _, files in os.walk(xml_dir):
                    for f in files:
                        if os.path.splitext(f)[0] == name and f.endswith('.xml'):
                            try:
                                os.remove(os.path.join(root, f))
                                nxml += 1
                            except Exception as e:
                                print(f"无法删除xml{name}: {e}")
        self.status.config(
            text=f"清理完成！删除图片 {nimg} 张，txt {ntxt} 个，xml {nxml} 个。")
        self.clean_btn.config(state='normal')
        messagebox.showinfo("配对清理完成", f"共删除图片 {nimg} 张，txt {ntxt} 个，xml {nxml} 个。")

class HashCompareFrame(Frame):
    def __init__(self, master):
        super().__init__(master, bd=2, relief='groove')
        Label(self, text="哈希比对（查找重复图片并可一键删除）", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=4, pady=3, sticky='w')
        Label(self, text="文件夹1图片目录:").grid(row=1, column=0, sticky='w')
        self.img1_var = StringVar()
        Entry(self, textvariable=self.img1_var, width=35).grid(row=1, column=1, sticky='w')
        Button(self, text="选择", command=self.select_img1).grid(row=1, column=2, sticky='w')
        Label(self, text="txt目录:").grid(row=2, column=0, sticky='w')
        self.txt1_var = StringVar()
        Entry(self, textvariable=self.txt1_var, width=35).grid(row=2, column=1, sticky='w')
        Button(self, text="选择", command=self.select_txt1).grid(row=2, column=2, sticky='w')
        Label(self, text="xml目录:").grid(row=3, column=0, sticky='w')
        self.xml1_var = StringVar()
        Entry(self, textvariable=self.xml1_var, width=35).grid(row=3, column=1, sticky='w')
        Button(self, text="选择", command=self.select_xml1).grid(row=3, column=2, sticky='w')
        Label(self, text="文件夹2图片目录:").grid(row=4, column=0, sticky='w')
        self.img2_var = StringVar()
        Entry(self, textvariable=self.img2_var, width=35).grid(row=4, column=1, sticky='w')
        Button(self, text="选择", command=self.select_img2).grid(row=4, column=2, sticky='w')
        Label(self, text="txt目录:").grid(row=5, column=0, sticky='w')
        self.txt2_var = StringVar()
        Entry(self, textvariable=self.txt2_var, width=35).grid(row=5, column=1, sticky='w')
        Button(self, text="选择", command=self.select_txt2).grid(row=5, column=2, sticky='w')
        Label(self, text="xml目录:").grid(row=6, column=0, sticky='w')
        self.xml2_var = StringVar()
        Entry(self, textvariable=self.xml2_var, width=35).grid(row=6, column=1, sticky='w')
        Button(self, text="选择", command=self.select_xml2).grid(row=6, column=2, sticky='w')
        self.compare_btn = Button(self, text="开始比对", command=self.thread_compare)
        self.compare_btn.grid(row=7, column=1, pady=7, sticky='w')
        self.status = Label(self, text="", fg='blue')
        self.status.grid(row=8, column=0, columnspan=4, sticky='w')
        # 重复文件列表
        self.listbox = Listbox(self, selectmode='single', width=80, height=8)
        self.listbox.grid(row=9, column=0, columnspan=4)
        self.common_files1 = []
        self.common_files2 = []
        Button(self, text="删除文件夹1中重复图片及标注", command=lambda: self.thread_delete(1)).grid(row=10, column=1, pady=4, sticky='e')
        Button(self, text="删除文件夹2中重复图片及标注", command=lambda: self.thread_delete(2)).grid(row=10, column=2, pady=4, sticky='w')

    def select_img1(self): self._select_dir(self.img1_var)
    def select_txt1(self): self._select_dir(self.txt1_var)
    def select_xml1(self): self._select_dir(self.xml1_var)
    def select_img2(self): self._select_dir(self.img2_var)
    def select_txt2(self): self._select_dir(self.txt2_var)
    def select_xml2(self): self._select_dir(self.xml2_var)
    def _select_dir(self, var):
        folder = filedialog.askdirectory()
        if folder: var.set(folder)

    def thread_compare(self):
        self.status.config(text="正在哈希比对，请稍候...")
        self.compare_btn.config(state='disabled')
        threading.Thread(target=self.compare_action, daemon=True).start()

    def compare_action(self):
        img1 = self.img1_var.get().strip()
        img2 = self.img2_var.get().strip()
        if not img1 or not img2:
            self.status.config(text="请先填写或选择两个图片目录")
            self.compare_btn.config(state='normal')
            return
        map1 = collect_image_hashes(img1)
        map2 = collect_image_hashes(img2)
        common, files1, files2 = find_common_hashes(map1, map2)
        self.common_files1 = files1
        self.common_files2 = files2
        self.listbox.delete(0, END)
        show_num = min(100, len(files1))
        for i in range(show_num):
            self.listbox.insert(
                END,
                os.path.relpath(files1[i], img1) + "  <==>  " +
                os.path.relpath(files2[i], img2)
            )
        info = f"共有 {len(common)} 组相同图片，文件夹1中{len(files1)}张，文件夹2中{len(files2)}张（只显示前{show_num}项）"
        self.status.config(text=info)
        if not common:
            messagebox.showinfo("结果", "没有找到内容相同的图片！")
        self.compare_btn.config(state='normal')

    def thread_delete(self, which):
        threading.Thread(target=lambda: self.delete_action(which), daemon=True).start()

    def delete_action(self, which):
        if which == 1:
            files = self.common_files1
            tip = "你确定要删除文件夹1中所有重复图片及标注文件吗？"
            label_dir = self.txt1_var.get().strip() or None
            xml_dir = self.xml1_var.get().strip() or None
        else:
            files = self.common_files2
            tip = "你确定要删除文件夹2中所有重复图片及标注文件吗？"
            label_dir = self.txt2_var.get().strip() or None
            xml_dir = self.xml2_var.get().strip() or None
        if not files:
            messagebox.showinfo("提示", "没有可删除的重复图片。")
            return
        if messagebox.askyesno("确认删除", tip):
            num, failed = delete_files(files, label_dir, xml_dir)
            messagebox.showinfo("删除结果", f"成功删除{num}张图片及其标注，失败{failed}张。")
            self.compare_action()  # 刷新列表

if __name__ == '__main__':
    try:
        from tqdm import tqdm
    except ImportError:
        print("请先 pip install tqdm")
        exit(1)
    try:
        from PIL import Image
    except ImportError:
        print("请先 pip install pillow")
        exit(1)
    root = Tk()
    root.title("图片数据清理&重复比对工具")
    # 上分区：配对清理，下分区：哈希比对
    pair_frame = PairCleanFrame(root)
    pair_frame.pack(padx=6, pady=6, fill='x')
    compare_frame = HashCompareFrame(root)
    compare_frame.pack(padx=6, pady=6, fill='x')
    root.mainloop()


