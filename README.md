# YOLO-Model-Test
验证模型在测试集上的效果.

有测试图片和xml文件以及训练好的yolov8模型。

测试集包含多个文件夹，每个文件夹下都有图片以及对应的xml标注文件。

1，每个文件夹下都输出混淆矩阵以及混淆概率矩阵，还有精确率、召回率、F1，map50。最后输出总体全部文件夹图片的测试指标

2，可供选择是否需要保存预测后带有标注框的图片，放入到指定目录下，保存时会保持与测试数据集一致的目录结构

3，创建GUI界面以选择参数，可浏览 ``models`` 目录下的 ``.pt`` 或 ``.onnx`` 模型，
   并选择数据集根目录

4，使用进度条显示推理进度，并在保存图片时同样显示进度

5，每次运行会在日志目录下生成 ``run.log`` 和 ``debug.log`` 两个文件，其中 ``run.log``
   记录模型、数据集及指标等信息，``debug.log`` 仅记录错误或异常

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
- **路径解析**：配置文件中使用的相对路径均以项目根目录为基准，可在任意目录调用脚本。
- **混淆矩阵脚本**：`confusion_cli.py` 会递归处理数据集下的所有子文件夹，为每个层级以及整体数据生成混淆矩阵和概率矩阵，结果保存在 `output/数据集N/子文件夹` 结构中，便于查看分类效果。
- **日志与报告**：`log_setup.py` 初始化日志系统，并在测试结束后导出 Markdown/HTML 报告，可记录每个模型及数据集的表现。
- **数据集处理**：`dataset/xml_loader.py` 负责加载图片与标注，`dataset_stats.py` 计算类别分布、标注框尺寸等统计信息。
- **推理与评估**：`predictor.py` 使用指定模型权重对图片推理。`metrics/evaluator.py` 计算多种指标并生成混淆矩阵；必要时可在 `metrics/confusion.py` 实现可视化。
- **模型管理**：`model_manager/loader.py` 方便在多个模型间切换或记录版本信息。
- **GUI**：`ui/gui.py` 提供参数选择界面和进度条，可浏览模型文件和数据集目录，对 `predictor` 与 `evaluator` 封装。
- **单元测试与 CI**：`tests/` 编写覆盖数据加载、推理、评估等功能的测试脚本，可与持续集成工具（如 GitHub Actions）结合，保证更新的稳定性。

针对“如何开始”：

1. **编写配置管理**  
   在 `configs/default.yaml` 填入模型路径、置信度阈值、数据集目录等基础配置，然后在 `src/config.py` 实现读取并解析这些配置。
2. **实现数据集加载**  
   `src/datasets/xml_loader.py` 负责从 `test_data/` 读取图片和对应的 XML 标注，可先完成这一部分，确保能得到图片路径和标注框信息。
3. **构建推理和评估流程**  
   - 在 `src/inference/predictor.py` 调用 YOLOv8 模型，对输入图片进行推理。  
   - 在 `src/metrics/evaluator.py` 计算混淆矩阵、Precision/Recall/F1、mAP 等指标，并按 README 所述生成混淆概率矩阵等可视化结果。
4. **提供命令行或 GUI 入口**  
   `src/cli.py` 或 `src/ui/gui.py` 可以作为统一入口，允许用户选择模型、数据集和输出目录。根据需要还能加入日志功能（`src/log_setup.py`）。
5. **补充单元测试**  
   仿照 `tests/` 目录的规划，为数据加载、模型推理和评估模块编写基础测试，确保后续修改不会破坏现有功能。
6. **生成混淆矩阵**
使用 `python src/confusion_cli.py --model models/best.pt --data test_data --output output` 运行，结果会写入 `output/test_data1/`（下次运行为 `test_data2/` 等），其中包含数据集下所有层级子文件夹以及整体数据的混淆矩阵和概率矩阵图像