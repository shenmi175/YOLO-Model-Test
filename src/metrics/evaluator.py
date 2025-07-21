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
    def __init__(self, iou_threshold: float = 0.5, class_names: List[str] | None = None) -> None:
        self.iou_threshold = iou_threshold
        self.class_names = class_names

    def evaluate(self, annotations: List[Annotation], predictions: Dict[str, List[Box]]) -> EvalResult:
        tp = fp = fn = 0
        pred_matches: List[int] = []
        total_gt = 0

        if self.class_names is not None:
            labels = list(self.class_names)
        else:
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
            if self.class_names is not None:
                gts = [b for b in ann.boxes if b.label in self.class_names]
            else:
                gts = ann.boxes

            total_gt += len(gts)
            preds = predictions.get(ann.image_path, [])

            if self.class_names is not None:
                preds = [b for b in preds if b.label in self.class_names]
            matched_gt: set[int] = set()
            # pred_used: set[int] = set()

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
                    tp += 1
                    pred_matches.append(1)
                    p_idx = idx.get(pred.label, bg_idx)
                    g_idx = idx.get(gts[best_j].label, bg_idx)
                    confusion[g_idx][p_idx] += 1
                else:
                    fp += 1
                    pred_matches.append(0)
                    p_idx = idx.get(pred.label, bg_idx)
                    confusion[bg_idx][p_idx] += 1

            for j, gt in enumerate(gts):
                if j not in matched_gt:
                    fn += 1
                    g_idx = idx.get(gt.label, bg_idx)
                    confusion[g_idx][bg_idx] += 1

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
    def evaluate_groups(
        self,
        annotations: List[Annotation],
        predictions: Dict[str, List[Box]],
        groups: Dict[str, List[Annotation]],
    ) -> Dict[str, EvalResult]:
        """Evaluate multiple groups of annotations in a single pass."""

        if self.class_names is not None:
            labels = list(self.class_names)
        else:
            label_set = {b.label for ann in annotations for b in ann.boxes}
            for boxes in predictions.values():
                for b in boxes:
                    label_set.add(b.label)
            labels = sorted(label_set)

        labels.append("background")
        bg_idx = len(labels) - 1
        idx = {l: i for i, l in enumerate(labels)}

        # Mapping from annotation image path to groups it belongs to
        ann_groups: Dict[str, List[str]] = {a.image_path: ["overall"] for a in annotations}
        for name, anns in groups.items():
            for ann in anns:
                ann_groups.setdefault(ann.image_path, ["overall"]).append(name)

        # Initialize accumulators
        acc: Dict[str, Dict[str, object]] = {}
        for name in ["overall", *groups.keys()]:
            acc[name] = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "pred_matches": [],
                "total_gt": 0,
                "confusion": [[0 for _ in labels] for _ in labels],
            }

        for ann in annotations:
            if self.class_names is not None:
                gts = [b for b in ann.boxes if b.label in self.class_names]
            else:
                gts = ann.boxes

            preds = predictions.get(ann.image_path, [])
            if self.class_names is not None:
                preds = [b for b in preds if b.label in self.class_names]

            matched_gt: set[int] = set()
            ann_conf = [[0 for _ in labels] for _ in labels]
            ann_pred_matches: List[int] = []
            tp_add = fp_add = 0

            for pred in preds:
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
                    tp_add += 1
                    ann_pred_matches.append(1)
                    p_idx = idx.get(pred.label, bg_idx)
                    g_idx = idx.get(gts[best_j].label, bg_idx)
                    ann_conf[g_idx][p_idx] += 1
                else:
                    fp_add += 1
                    ann_pred_matches.append(0)
                    p_idx = idx.get(pred.label, bg_idx)
                    ann_conf[bg_idx][p_idx] += 1

            fn_add = 0
            for j, gt in enumerate(gts):
                if j not in matched_gt:
                    fn_add += 1
                    g_idx = idx.get(gt.label, bg_idx)
                    ann_conf[g_idx][bg_idx] += 1

            for name in ann_groups.get(ann.image_path, ["overall"]):
                store = acc[name]
                store["tp"] = int(store["tp"]) + tp_add
                store["fp"] = int(store["fp"]) + fp_add
                store["fn"] = int(store["fn"]) + fn_add
                store["total_gt"] = int(store["total_gt"]) + len(gts)
                cast_pm = store["pred_matches"]
                assert isinstance(cast_pm, list)
                cast_pm.extend(ann_pred_matches)
                conf = store["confusion"]
                assert isinstance(conf, list)
                for r in range(len(labels)):
                    row = conf[r]
                    for c in range(len(labels)):
                        row[c] += ann_conf[r][c]

        results: Dict[str, EvalResult] = {}
        for name, data in acc.items():
            tp = int(data["tp"])
            fp = int(data["fp"])
            fn = int(data["fn"])
            total_gt = int(data["total_gt"])
            pred_matches = data["pred_matches"]  # type: ignore[list-item]
            confusion = data["confusion"]  # type: ignore[list-item]

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

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

            results[name] = EvalResult(
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

        return results