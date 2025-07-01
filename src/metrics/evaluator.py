from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.datasets.xml_loader import Annotation, Box


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
    labels: List[str]
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

        label_set = {b.label for ann in annotations for b in ann.boxes}
        for boxes in predictions.values():
            for b in boxes:
                label_set.add(b.label)
        labels = sorted(label_set)
        labels.append("background")
        bg_idx = len(labels) - 1
        idx = {l: i for i, l in enumerate(labels)}

        confusion = [[0 for _ in labels] for _ in labels]

        for ann in annotations:
            gts = ann.boxes
            total_gt += len(gts)
            preds = predictions.get(ann.image_path, [])
            matched_gt: set[int] = set()
            pred_used: set[int] = set()

            for i, pred in enumerate(preds):
                best_i = 0.0
                best_j = -1
                for j, gt in enumerate(gts):
                    if j in matched_gt:
                        continue
                    iv = iou(pred, gt)
                    if iv >= self.iou_threshold and iv > best_i:
                        best_i = iv
                        best_j = j
                if best_j >= 0:
                    matched_gt.add(best_j)
                    pred_used.add(i)
                    tp += 1
                    pred_matches.append(1)
                    p_idx = idx.get(pred.label, bg_idx)
                    g_idx = idx.get(gts[best_j].label, bg_idx)
                    confusion[p_idx][g_idx] += 1
                else:
                    fp += 1
                    pred_matches.append(0)
                    p_idx = idx.get(pred.label, bg_idx)
                    confusion[p_idx][bg_idx] += 1

            for j, gt in enumerate(gts):
                if j not in matched_gt:
                    fn += 1
                    g_idx = idx.get(gt.label, bg_idx)
                    confusion[bg_idx][g_idx] += 1

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

        conf_prob: List[List[float]] = []
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
            labels=labels,
            tp=tp,
            fp=fp,
            fn=fn,
        )