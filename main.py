import argparse

from src.config import Config
from src.ui import gui


def parse_args() -> argparse.Namespace:
    """
    --config 配置文件路径（通常是 yaml 或 json）

    --model 模型文件路径（例如 yolov8.pt）

    --data 数据集目录（如包含 images 和 labels 的目录）

    --output 输出目录（结果、预测等保存位置）

    --save-images 是否保存带有标注的图片（加这个参数会保存，默认不保存）

    --no-save 不保存预测结果的 txt 文件（加了就不保存，默认会保存）

    --gui 用 GUI 图形界面选参数（直接点选，不用命令行）

    """
    parser = argparse.ArgumentParser(description="Automated YOLO model testing")
    parser.add_argument("--config", help="Config file", default=None)
    parser.add_argument("--model", help="Model path", default=None)
    parser.add_argument("--data", help="Dataset directory", default=None)
    parser.add_argument("--output", help="Output directory", default=None)
    parser.add_argument("--save-images", action="store_true", help="Save annotated images")
    parser.add_argument("--img-dir", help="Directory for saved images", default=None)
    parser.add_argument("--no-save", action="store_true", help="Do not save prediction txt")
    parser.add_argument("--gui", action="store_true", help="Launch GUI for parameter selection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gui:
        gui.launch()
        return
    cfg = Config.from_file(args.config)
    if args.model:
        cfg.model_path = args.model
    if args.data:
        cfg.data_dir = args.data
    if args.output:
        cfg.output_dir = args.output
    if args.no_save:
        cfg.save_predictions = False
    if args.save_images:
        cfg.save_images = True
    img_dir = args.img_dir

    gui.run_evaluation(
        cfg.model_path,
        cfg.data_dir,
        cfg.output_dir,
        cfg.save_predictions,
        cfg.save_images,
        None,
        img_dir,
    )


if __name__ == "__main__":
    main()
