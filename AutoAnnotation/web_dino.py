import torch
from PIL import Image, ImageDraw
import gradio as gr
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 模型相关配置
model_id = "IDEA-Research/grounding-dino-base"
save_dir = "../models/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和处理器（只加载一次）
processor = AutoProcessor.from_pretrained(model_id, cache_dir=save_dir)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=save_dir).to(device)


def detect_objects(image, text_labels="person,cat,face,dog"):
    # 1. 处理类别输入
    if isinstance(text_labels, str):
        labels = [t.strip().lower() for t in text_labels.split(",") if t.strip()]
    else:
        labels = ["person", "cat", "face", "dog"]  # 默认

    text = [labels]  # 保持二维列表

    # 2. 处理图片格式
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    orig_img = image.copy()

    # 3. 推理
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.2,
        target_sizes=[image.size[::-1]],
    )

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    boxes = results[0]["boxes"].cpu().numpy()
    labels_out = results[0]["text_labels"]
    scores = results[0]["scores"].cpu().numpy()

    for bbox, label, score in zip(boxes, labels_out, scores):
        x1, y1, x2, y2 = map(int, bbox)
        label_text = f"{label} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(y1 - 15, 0)), label_text, fill="red")

    return orig_img, draw_img


# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## Grounding DINO Zero-Shot 检测 Demo")

    with gr.Row():
        with gr.Column():
            inp_img = gr.Image(type="pil", label="上传图片")
            inp_labels = gr.Textbox(label="检测类别（逗号分隔）", value="person,cat,face,dog")
            btn = gr.Button("检测")
        with gr.Column():
            out_orig = gr.Image(type="pil", label="原图")
            out_detect = gr.Image(type="pil", label="检测结果")

    btn.click(
        detect_objects,
        inputs=[inp_img, inp_labels],
        outputs=[out_orig, out_detect],
    )

demo.launch()
