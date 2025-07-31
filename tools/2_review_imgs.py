import os
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk

class ImageBrowser:
    def __init__(self, root):
        self.root = root
        self.root.title('图片浏览/删除工具')
        self.image_label = Label(root)
        self.image_label.pack(expand=True)

        self.btn_select = Button(root, text="选择图片文件夹", command=self.select_folder)
        self.btn_select.pack()

        self.img_paths = []
        self.idx = 0
        self.cur_img = None
        self.folder = ""

        self.root.bind("<a>", lambda e: self.show_prev())
        self.root.bind("<d>", lambda e: self.show_next())
        self.root.bind("<w>", lambda e: self.delete_current())
        self.root.bind("<Left>", lambda e: self.show_prev())
        self.root.bind("<Right>", lambda e: self.show_next())

    def select_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.folder = folder
        self.img_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'))]
        self.img_paths.sort()
        self.idx = 0
        if not self.img_paths:
            messagebox.showinfo("提示", "该文件夹下没有图片文件")
        else:
            self.show_image()

    def show_image(self):
        if not self.img_paths:
            self.image_label.config(image='', text="没有图片")
            return
        img_path = self.img_paths[self.idx]
        pil_img = Image.open(img_path)
        # Resize image to fit window
        maxsize = (1920, 1080)
        pil_img.thumbnail(maxsize)
        self.cur_img = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.cur_img, text="")
        self.root.title(f"{os.path.basename(img_path)}  ({self.idx+1}/{len(self.img_paths)})")

    def show_prev(self):
        if not self.img_paths:
            return
        self.idx = (self.idx - 1) % len(self.img_paths)
        self.show_image()

    def show_next(self):
        if not self.img_paths:
            return
        self.idx = (self.idx + 1) % len(self.img_paths)
        self.show_image()

    def delete_current(self):
        if not self.img_paths:
            return
        img_path = self.img_paths.pop(self.idx)
        try:
            os.remove(img_path)
        except Exception as e:
            messagebox.showerror("错误", f"删除失败: {e}")
            return
        if not self.img_paths:
            self.image_label.config(image='', text="没有图片")
            self.root.title("图片浏览/删除工具")
            return
        if self.idx >= len(self.img_paths):  # 最后一张被删，idx自动回到0
            self.idx = 0
        self.show_image()

if __name__ == '__main__':
    root = Tk()
    app = ImageBrowser(root)
    root.mainloop()