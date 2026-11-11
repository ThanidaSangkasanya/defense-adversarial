"""
compute_detection_metrics  —  Precision / Recall / mAP@0.5
Works with any YOLOv5 variant loaded via load_yolov5_for_inference().
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_attack.config import (
    CLOAKED_DIR, LABELS_DIR, PRETRAINED_WEIGHTS,
    FINETUNED_DIR, IMG_SIZE, CONF_THRESHOLD, DEVICE
)
from adaptive_attack.utils import load_yolov5_for_inference


# ─────────────────────────────────────────────────────────────────────────────
#  IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def _iou(b1, b2):
    """b1, b2: [x1, y1, x2, y2]"""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
#  GT label loader  (YOLO format  →  pixel coords)
# ─────────────────────────────────────────────────────────────────────────────

def _load_gt(label_path, img_w, img_h):
    if not label_path or not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or int(parts[0]) != 0:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append([
                (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                (cx + w / 2) * img_w, (cy + h / 2) * img_h,
            ])
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
#  Single-image detection  (AutoShape model)
# ─────────────────────────────────────────────────────────────────────────────

def _detect(model, img_np, img_size, conf_thr):
    """Returns list of [x1,y1,x2,y2,conf] for class-0 detections."""
    orig_conf = getattr(model, "conf", None)
    if orig_conf is not None:
        model.conf = conf_thr
    results = model(img_np, size=img_size)
    if orig_conf is not None:
        model.conf = orig_conf
    boxes = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().tolist():
        if int(cls) == 0 and float(conf) >= conf_thr:
            boxes.append([float(xyxy[0]), float(xyxy[1]),
                          float(xyxy[2]), float(xyxy[3]), float(conf)])
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
#  AP@0.5 from a list of (conf, tp) pairs
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ap(rec, prec):
    """11-point interpolation AP."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p = prec[rec >= thr].max() if np.any(rec >= thr) else 0.0
        ap += p / 11.0
    return ap


def _ap_from_detections(all_dets, n_gt):
    """
    all_dets : list of (conf, is_tp)  sorted by descending conf
    n_gt      : total GT boxes across all images
    """
    if n_gt == 0:
        return 0.0, 0.0, 0.0   # precision, recall, AP

    all_dets = sorted(all_dets, key=lambda x: -x[0])
    tp_cum = np.cumsum([d[1] for d in all_dets])
    fp_cum = np.cumsum([1 - d[1] for d in all_dets])

    rec  = tp_cum / n_gt
    prec = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = _compute_ap(rec, prec)
    final_prec = prec[-1] if len(prec) else 0.0
    final_rec  = rec[-1]  if len(rec)  else 0.0
    return float(final_prec), float(final_rec), float(ap)


# ─────────────────────────────────────────────────────────────────────────────
#  Main metric function
# ─────────────────────────────────────────────────────────────────────────────

def compute_detection_metrics(
    model,
    image_dir,
    label_dir,
    conf_thr=0.4,
    iou_thr=0.5,
    img_size=416,
    strip_suffix="_attack",
    device="cuda:0",
    allowed_basenames=None,
):
    """
    Returns dict with keys: precision, recall, map50
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(exts)])

    all_dets = []   # (conf, is_tp)
    n_gt     = 0

    for fname in files:
        base = os.path.splitext(fname)[0]
        stem = base[: -len(strip_suffix)] if strip_suffix and base.endswith(strip_suffix) else base

        if allowed_basenames is not None and stem not in allowed_basenames:
            continue

        img_path   = os.path.join(image_dir, fname)
        label_path = os.path.join(label_dir, stem + ".txt")

        img_pil = Image.open(img_path).convert("RGB")
        img_np  = np.array(img_pil)
        w, h    = img_pil.size

        gt_boxes  = _load_gt(label_path, w, h)
        det_boxes = _detect(model, img_np, img_size, conf_thr)

        n_gt   += len(gt_boxes)
        matched = [False] * len(gt_boxes)

        for det in sorted(det_boxes, key=lambda x: -x[4]):
            best_iou = 0.0
            best_j   = -1
            for j, gt in enumerate(gt_boxes):
                if matched[j]:
                    continue
                iou = _iou(det[:4], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_j   = j
            is_tp = best_iou >= iou_thr
            if is_tp:
                matched[best_j] = True
            all_dets.append((det[4], int(is_tp)))

    precision, recall, map50 = _ap_from_detections(all_dets, n_gt)
    return {"precision": precision, "recall": recall, "map50": map50}


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: load model + run full evaluation on a dataset
# ─────────────────────────────────────────────────────────────────────────────

def _get_model_list(dataset_name, model_name):
    adaptive_path = os.path.join(FINETUNED_DIR, f"{model_name}_adaptive_{dataset_name}.pt")
    weights = PRETRAINED_WEIGHTS.get(model_name)
    if weights is None:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Expected one of {list(PRETRAINED_WEIGHTS.keys())}")
    models = [(f"{model_name} (original)", "pretrained", weights)]
    if os.path.exists(adaptive_path):
        models.append((f"{model_name} (adaptive, mixed)", "local", adaptive_path))
    return models


def run_evaluation(dataset_name, model_name="yolov5n"):
    cloaked_dir = CLOAKED_DIR[dataset_name]
    label_dir   = LABELS_DIR[dataset_name]

    model_list = _get_model_list(dataset_name, model_name)

    results = []
    for label, kind, path in model_list:
        print(f"\n[Eval] {label}  ({path})")
        model = load_yolov5_for_inference(path, device=DEVICE, conf=CONF_THRESHOLD)
        metrics = compute_detection_metrics(
            model, cloaked_dir, label_dir,
            conf_thr=CONF_THRESHOLD, iou_thr=0.5,
            img_size=IMG_SIZE, strip_suffix="_attack", device=DEVICE,
        )
        print(f"  Precision: {metrics['precision']*100:.2f}%  "
              f"Recall: {metrics['recall']*100:.2f}%  "
              f"mAP@0.5: {metrics['map50']*100:.2f}%")
        results.append({"label": label, **metrics})

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cs2", choices=["cs2", "cf"])
    parser.add_argument("--model",   default="yolov5n",
                        choices=["yolov5n", "yolov5s", "yolov5m"])
    args = parser.parse_args()
    run_evaluation(args.dataset, args.model)