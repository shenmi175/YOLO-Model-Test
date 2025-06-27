# YOLO-Model-Test
验证模型在测试集上的效果.

有测试图片和xml文件以及训练好的yolov8模型。

测试集包含多个文件夹，每个文件夹下都有图片以及对应的xml标注文件。

1，每个文件夹下都输出混淆矩阵以及混淆概率矩阵，还有精确率、召回率、F1，map50。最后输出总体全部文件夹图片的测试指标

2，可供选择是否需要保存预测后带有标注框的图片，放入到指定目录下。文件夹命名与结构和测试数据集相同

3，创建GUI界面以选择参数

4，使用进度条





```python
YOLO-Model-Test/
├── README.md
├── requirements.txt                   # 依赖列表
├── configs/
│   └── default.yaml                   # 模型路径、阈值、数据集目录等默认配置
├── models/
│   └── best.pt
├── test_data/                         # 测试数据（保持原有结构）
│   ├── test1/
│   └── test2/
├── src/
│   ├── cli.py                         # 命令行入口，可启动批量推理或生成报告
│   ├── config.py                      # 配置加载与管理
│   ├── log_setup.py                   # 日志初始化，记录测试过程
│   ├── utils/
│   │   ├── __init__.py
│   │   └── file_utils.py              # 通用文件/路径工具
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── xml_loader.py              # 解析 XML 标注
│   │   └── dataset_stats.py           # 统计并可视化数据集分布
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py               # 调用 YOLOv8 模型进行推理
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── evaluator.py               # 计算混淆矩阵、PR/F1、mAP@[.5:.95] 等
│   │   └── confusion.py               # 混淆矩阵与混淆概率矩阵可视化
│   ├── model_manager/
│   │   ├── __init__.py
│   │   └── loader.py                  # 管理多个模型权重和版本
│   └── ui/
│       ├── __init__.py
│       └── gui.py                     # 提供 GUI 界面和进度条
├── output/                            # 保存预测图片和评估结果
├── logs/                              # 日志文件
└── tests/                             # 单元测试与持续集成脚本
    ├── test_dataset.py
    ├── test_predictor.py
    └── test_evaluator.py

```

**模块说明**

- **CLI 与配置管理**：`cli.py` 结合 `config.py` 读取 `configs/default.yaml`，支持在命令行执行推理、输出指标或批量保存预测图像。
- **日志与报告**：`log_setup.py` 初始化日志系统，并在测试结束后导出 Markdown/HTML 报告，可记录每个模型及数据集的表现。
- **数据集处理**：`dataset/xml_loader.py` 负责加载图片与标注，`dataset_stats.py` 计算类别分布、标注框尺寸等统计信息。
- **推理与评估**：`predictor.py` 使用指定模型权重对图片推理。`metrics/evaluator.py` 计算多种指标并生成混淆矩阵；必要时可在 `metrics/confusion.py` 实现可视化。
- **模型管理**：`model_manager/loader.py` 方便在多个模型间切换或记录版本信息。
- **GUI**：`ui/gui.py` 提供参数选择界面和进度条，对 `predictor` 与 `evaluator` 封装。
- **单元测试与 CI**：`tests/` 编写覆盖数据加载、推理、评估等功能的测试脚本，可与持续集成工具（如 GitHub Actions）结合，保证更新的稳定性。
