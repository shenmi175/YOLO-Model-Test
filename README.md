# YOLO-Model-Test

本项目提供一套基于 [YOLOv8](https://github.com/ultralytics/ultralytics) 的模型测试与数据增强工具，能够方便地在自定义数据集上评估模型精度、生成混淆矩阵，并支持通过图形界面或命令行完成推理和图像增强。

## 目录结构

```text
YOLO-Model-Test/
├── configs/               # 默认配置文件
│   └── default.yaml
├── models/                # 示例模型权重
│   └── best.pt
├── src/                   # 主要代码
│   ├── augment/           # 图像增强实现
│   ├── datasets/          # 数据集加载与统计
│   ├── inference/         # YOLO 推理封装
│   ├── metrics/           # 评估指标与混淆矩阵
│   ├── model_manager/     # 多模型管理
│   └── ui/                # GUI 界面
├── test_data/             # 示例测试集
├── aug_main.py            # 启动增强 GUI
├── main.py                # 推理入口，可使用 CLI 或 GUI
├── requirements.txt       # 依赖列表
└── tools/                 # 辅助脚本
    └── xml2txt.py         # VOC 标注转为文本格式
```

## 安装依赖

```bash
pip install -r requirements.txt
```

部分功能依赖 OpenCV、Pillow、albumentations 等库，如未包含在 `requirements.txt` 中，请根据实际情况安装。

## 快速开始

### 使用命令行推理

```bash
python main.py \
  --config configs/default.yaml \
  --model models/best.pt \
  --data test_data \
  --output output
```

上述命令会按照配置文件和命令行参数执行推理，并在 `output/` 目录生成预测结果及评估指标。

### 使用 GUI 推理

运行：

```bash
python main.py --gui
```

在图形界面中可以选择模型、数据集、输出目录以及是否保存带标注的图片。界面底部带有进度条，可实时查看推理进度。

## 数据增强

执行 `python aug_main.py` 打开增强界面，设置：

1. **输入目录** 与 **输出目录**
2. **增强数量**：要处理的图片数
3. **每图次数**：同一张图片的增强重复次数，默认为 1
4. 各种增强操作及其概率

点击 **开始增强** 后，会在进度条中显示整体进度，每完成一次增强都会更新进度值。增强后的图片和对应标注将保存到输出目录，文件名会带上 `_aug_序号` 后缀。

## 输出与日志

- 评估结果和预测文本保存至 `output/` 下的对应目录
- 若启用保存图片，带标注的结果图也会放在 `output/` 内保持原目录结构
- 日志文件位于 `logs/`，包括 `run.log`（详细运行信息）和 `debug.log`（仅错误信息）

## 其他工具

- `tools/xml2txt.py`：将 Pascal VOC XML 标注转换为简单的 `label xmin ymin xmax ymax` 格式，可选 `--normalize` 按图像尺寸归一化

## 运行测试

项目中包含基础的 `pytest` 测试，可在根目录执行：

```bash
pytest -q
```

确保主要组件在修改后仍能正常工作。

## 贡献

欢迎提交 Issue 或 Pull Request 反馈问题与改进建议。