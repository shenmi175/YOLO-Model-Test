import os
import shutil
import cv2
import threading
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from os.path import join, exists, splitext, basename


def yolov8_style_resize(frame, target_size=(640, 640), color=(114, 114, 114)):
    """YOLOv8风格的图像缩放和填充"""
    h, w = frame.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    if scale != 1:
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = target_w - new_w
    dh = target_h - new_h
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    frame = cv2.copyMakeBorder(frame, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=color)
    return frame


def process_video(video_path, output_base, frame_interval, resize, target_size):
    """处理单个视频文件提取帧"""
    video_name = splitext(basename(video_path))[0]
    output_folder = join(output_base, video_name)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            filename = f"frame_{frame_count:04d}_{video_name}.jpg"
            output_path = join(output_folder, filename)

            if resize and target_size:
                frame = yolov8_style_resize(frame, target_size=target_size)

            cv2.imwrite(output_path, frame)

        frame_count += 1

    cap.release()


def move_and_rename_images(src_folder):
    """将子文件夹中的图片移动到根目录并重命名"""
    parent_folder = src_folder
    subfolders = [f.path for f in os.scandir(src_folder) if f.is_dir()]
    counter = 1

    for subfolder in subfolders:
        for file_name in os.listdir(subfolder):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                src_file = os.path.join(subfolder, file_name)
                ext = os.path.splitext(file_name)[1]
                target_file = os.path.join(parent_folder, f"image_{counter}{ext}")

                while os.path.exists(target_file):
                    counter += 1
                    target_file = os.path.join(parent_folder, f"image_{counter}{ext}")

                shutil.move(src_file, target_file)
                print(f"Moved and renamed: {src_file} to {target_file}")
                counter += 1


def remove_empty_folders(path):
    """递归删除空的文件夹"""
    if not os.path.isdir(path):
        return
    for root_dir, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            folder_path = os.path.join(root_dir, dir)
            try:
                if not os.listdir(folder_path):  # 如果文件夹为空
                    os.rmdir(folder_path)
            except OSError as ex:
                print(f"无法删除文件夹 {folder_path}: {ex}")


class VideoFrameExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("视频帧提取工具 - YOLOv8风格")
        self.root.geometry("600x350")

        # 初始化变量
        self.input_dir_var = StringVar()
        self.output_dir_var = StringVar(value="")  # 初始为空
        self.interval_var = StringVar(value="10")
        self.resize_var = IntVar(value=1)
        self.width_var = StringVar(value="640")
        self.height_var = StringVar(value="640")

        self.create_widgets()

    def create_widgets(self):
        """创建GUI组件"""
        # 输入目录选择
        Label(self.root, text="输入目录:").grid(row=0, column=0, sticky=W, padx=5, pady=5)
        Entry(self.root, textvariable=self.input_dir_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        Button(self.root, text="浏览", command=self.select_input_dir).grid(row=0, column=2, padx=5, pady=5)

        # 输出目录选择
        Label(self.root, text="输出目录:").grid(row=1, column=0, sticky=W, padx=5, pady=5)
        Entry(self.root, textvariable=self.output_dir_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        Button(self.root, text="浏览", command=self.select_output_dir).grid(row=1, column=2, padx=5, pady=5)  # 新增按钮

        # 帧间隔设置
        Label(self.root, text="帧间隔:").grid(row=2, column=0, sticky=W, padx=5, pady=5)
        Entry(self.root, textvariable=self.interval_var, width=10).grid(row=2, column=1, sticky=W, padx=5, pady=5)
        Label(self.root, text="（每隔N帧提取一张）").grid(row=2, column=2, sticky=W, padx=5, pady=5)

        # 尺寸调整选项
        Checkbutton(self.root, text="调整尺寸", variable=self.resize_var,
                   command=self.toggle_size_entries).grid(row=3, column=0, sticky=W, padx=5, pady=5)

        # 目标尺寸输入
        Label(self.root, text="目标尺寸 (宽 高):").grid(row=4, column=0, sticky=W, padx=5, pady=5)
        self.width_entry = Entry(self.root, textvariable=self.width_var, width=10)
        self.width_entry.grid(row=4, column=1, sticky=W, padx=5, pady=5)
        self.height_entry = Entry(self.root, textvariable=self.height_var, width=10)
        self.height_entry.grid(row=4, column=2, sticky=W, padx=5, pady=5)

        # 初始化尺寸输入框状态
        self.toggle_size_entries()

        # 进度条
        self.progress = ttk.Progressbar(self.root, orient=HORIZONTAL, length=400, mode='determinate')
        self.progress.grid(row=5, column=0, columnspan=3, padx=5, pady=10)

        # 开始按钮
        Button(self.root, text="开始处理", command=self.start_processing,
              bg="green", fg="white").grid(row=6, column=1, pady=10)

    def select_input_dir(self):
        """选择输入目录"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.input_dir_var.set(dir_path)
            # 如果输出目录为空，则自动设置为 input_dir/out
            if not self.output_dir_var.get():
                out_dir = os.path.join(dir_path, 'out')
                self.output_dir_var.set(out_dir)

    def select_output_dir(self):
        """选择输出目录"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir_var.set(dir_path)

    def toggle_size_entries(self):
        """切换尺寸输入框状态"""
        state = NORMAL if self.resize_var.get() else DISABLED
        self.width_entry.config(state=state)
        self.height_entry.config(state=state)

    def start_processing(self):
        """开始处理视频"""
        # 获取参数
        input_dir = self.input_dir_var.get()
        output_dir = self.output_dir_var.get()
        interval = self.interval_var.get()
        resize = self.resize_var.get()

        # 参数验证
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("错误", "请输入有效的输入目录")
            return
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("错误", "输出目录必须存在")
            return

        try:
            interval = int(interval)
            if interval < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "帧间隔必须是大于0的整数")
            return

        target_size = None
        if resize:
            try:
                width = int(self.width_var.get())
                height = int(self.height_var.get())
                if width <= 0 or height <= 0:
                    raise ValueError
                target_size = (width, height)
            except ValueError:
                messagebox.showerror("错误", "目标尺寸必须是正整数")
                return

        # 启动处理线程
        threading.Thread(target=self.process_videos,
                        args=(input_dir, output_dir, interval, resize, target_size),
                        daemon=True).start()

        # 禁用按钮防止重复点击
        self.root.children['!button3'].config(state=DISABLED)

    def update_progress(self, new_value):
        self.progress["value"] = new_value


    def process_videos(self, input_dir, output_dir, frame_interval, resize_flag, target_size):
        """处理视频的后台线程函数"""
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("错误", f"无法创建输出目录: {e}")
            return

        video_exts = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')

        # 获取所有视频文件
        video_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(video_exts):
                    video_files.append(join(root, file))

        total_videos = len(video_files)
        if total_videos == 0:
            self.root.after(0, lambda: messagebox.showwarning("提示", "未找到任何视频文件"))
            return

        progress_increment = 70.0 / total_videos  # 视频处理占70%进度

        # 处理每个视频
        for idx, video_path in enumerate(video_files):
            process_video(video_path, output_dir, frame_interval, resize_flag, target_size)
            self.root.after(0, lambda: self.update_progress(self.progress["value"] + progress_increment))

        # 移动文件后更新到 90%
        self.root.after(0, lambda: self.progress.config(value=90))

        # 删除空文件夹后更新到 100%
        self.root.after(0, lambda: self.progress.config(value=100))

        # 恢复按钮状态
        self.root.after(0, lambda: self.root.children['!button3'].config(state=NORMAL))
        self.root.after(0, lambda: messagebox.showinfo("完成", "处理已完成"))


if __name__ == "__main__":
    root = Tk()
    app = VideoFrameExtractorApp(root)
    root.mainloop()



    # 使用示例
    # 输入输出路径、帧率、是否调整图片尺寸(默认不调整)、目标尺寸（与resize一起使用）
    # python 1_video2imgs.py I:\allShare\1zmb\datas\datasets\error\test\error4 I:\allShare\1zmb\datas\datasets\error\test/output_images --interval 10 --resize --size 1920 1080




# import os
# import cv2
#
# # 在一个文件下多个视频，创建图片并分解
# videos_src_path = r'J:\other\datasets\2m_cat_person'
# # videos = os.listdir(videos_src_path)
# # print(videos)
# videoPath = []
# for path, dir_list, file_list in os.walk(videos_src_path):
#     for file_name in file_list:
#         videoPath.append(os.path.join(path, file_name))
#
# videoPath.sort()
# # print(videoPath)
# # # 遍历文件夹下的视频文件
# for each_video in videoPath:
#     print('Video Name :', each_video)
#     # get the name of each video, and make the directory to save frames
#     # 截取视频文件的后缀，保留名称
#     each_video_name = each_video.split('.', 2)[0]
#     print(each_video_name)
#     # 创建视频同名的文件
#     os.makedirs(each_video_name, exist_ok=True)
#     # 保存处理好的图片路径
#     each_video_save_full_path = each_video_name + '/'
#     print(each_video_save_full_path)
#     # 获取全部视频的路径
#     each_video_full_path = os.path.join(videos_src_path, each_video)
#     print('path_all:' + each_video_full_path)
#     # 读取全部视频
#     cap = cv2.VideoCapture(each_video_full_path)
#     frame_count = 0
#     success = True
#     while (success):
#         success, frame = cap.read()
#         frame_count += 1
#         try:
#             if frame_count % 25 == 0 and frame.shape[0] >= 720 and frame.shape[1] >= 1280 and success:
#             # if frame_count % 1 == 0 and success:
#                 frame = cv2.resize(frame, (480, 288))
#                 # frame = cv2.flip(frame, -1)
#                 # frame = cv2.flip(frame, 1)
#                 print(each_video_save_full_path)
#                 cv2.imwrite(each_video_save_full_path + "%06d.jpg" % frame_count, frame)
#         except:
#             pass


