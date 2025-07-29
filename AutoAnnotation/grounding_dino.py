"""@misc{liu2023grounding,
      title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
      author={Shilong Liu and Zhaoyang Zeng and Tianhe Ren and Feng Li and Hao Zhang and Jie Yang and Chunyuan Li and Jianwei Yang and Hang Su and Jun Zhu and Lei Zhang},
      year={2023},
      eprint={2303.05499},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}"""

import requests
import torch
from PIL import Image, ImageDraw
from IPython.display import display
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot


image_path = "/kaggle/working/cats/['cat']_1_0.76.jpg"
image = Image.open(image_path)


# text = "each person. hand."
text = [["person", "cat", "face", "dog"]]

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.3,
    text_threshold=0.2,
    target_sizes=[image.size[::-1]]
)

"""
rersults输出示例：
[{'scores': tensor([0.7410, 0.3004, 0.3725], device='cuda:0'),
  'boxes': tensor([[2.0137e+00, 1.5610e+00, 1.2040e+02, 1.3470e+02],
          [2.3169e+01, 1.7312e+01, 4.9428e+01, 4.3809e+01],
          [1.1266e-02, 1.8569e-01, 4.8561e+01, 1.3219e+02]], device='cuda:0'),
  'text_labels': ['cat', 'face', 'person'],
  'labels': ['cat', 'face', 'person']}]
"""


img = Image.open(image_path)
draw = ImageDraw.Draw(img)

boxes = results[0]['boxes']
labels = results[0]['labels']
scores = results[0]['scores']

boxes = boxes.cpu().numpy()
scores = scores.cpu().numpy()  # 确保score也是可索引的numpy

for bbox, label, score in zip(boxes, labels, scores):
    x1, y1, x2, y2 = map(int, bbox)
    # 标签与分数合成一行，保留两位小数
    label_text = f"{label} {score:.2f}"
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, y1 - 0), label_text, fill="red")

display(img)












