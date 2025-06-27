from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from datasets.xml_loader import Annotation, Box


def iou(box1: Box, box2: Box) -> float:
    x1 = max(box1.xmin, box2.xmin)
    y1 = max(box1.ymin, box2.ymin)
    x2 = min(box1.xmax, box2.xmax)
    y2 = min(box1.ymax, box2.ymax)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
    area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
    union = area1 + area2 - inter
    return inter / union if union else 0.0


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float
    map50: float
    confusion_matrix: List[List[int]]
    confusion_prob: List[List[float]]
    tp: int
    fp: int
    fn: int


class Evaluator:
    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold

    def evaluate(self, annotations: List[Annotation], predictions: Dict[str, List[Box]]) -> EvalResult:
        tp = fp = fn = 0
        pred_matches: List[int] = []
        total_gt = 0
        for ann in annotations:
            gts = ann.boxes
            total_gt += len(gts)
            preds = predictions.get(ann.image_path, [])
            matched_gt: set[int] = set()
            for pred in preds:
                best_iou = 0.0
                best_j = -1
                for j, gt in enumerate(gts):
                    i = iou(pred, gt)
                    if i >= self.iou_threshold and i > best_iou:
                        best_iou = i
                        best_j = j
                if best_j >= 0 and best_j not in matched_gt:
                    tp += 1
                    matched_gt.add(best_j)
                    pred_matches.append(1)
                else:
                    fp += 1
                    pred_matches.append(0)
            fn += len(gts) - len(matched_gt)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        # compute simple mAP@0.5 from accumulated detections
        ap = 0.0
        if total_gt:
            tp_cum = fp_cum = 0
            recall_prev = 0.0
            for m in pred_matches:
                if m:
                    tp_cum += 1
                else:
                    fp_cum += 1
                recall_cur = tp_cum / total_gt
                prec_cur = tp_cum / (tp_cum + fp_cum)
                ap += (recall_cur - recall_prev) * prec_cur
                recall_prev = recall_cur
        map50 = ap

        confusion = [[tp, fp], [fn, 0]]
        conf_prob = []
        for row in confusion:
            s = sum(row)
            conf_prob.append([c / s if s else 0.0 for c in row])

        return EvalResult(
            precision=precision,
            recall=recall,
            f1=f1,
            map50=map50,
            confusion_matrix=confusion,
            confusion_prob=conf_prob,
            tp=tp,
            fp=fp,
            fn=fn,
        )