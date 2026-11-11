"""
Step 5: Full Evaluation — DSR + Recall + Precision + mAP@0.5

เก็บ DSR เดิมไว้ (evaluate_quick) และเพิ่มชุด metric แบบ Phase 1:
  - Recall, Precision (ที่ conf=0.4, IoU=0.5 — ตรงกับ get_cloak.py)
  - mAP@0.5 (average precision จาก PR curve เต็มช่วง confidence)

วัดทั้งบนภาพ CLEAN และ CLOAKED แยกกัน เพื่อดู gap ที่ cloak สร้างได้จริง
(เหมือน check_clean_dsr.py) แล้วรวมทุกอย่างลง Excel + กราฟเดียว

Usage:
    python adaptive_attack/evaluation.py --dataset cs2
    python adaptive_attack/evaluation.py --dataset cs2 --epsilon 16
"""
import os
import sys
import argparse
import json
import time
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_attack.config import (
    DATASETS, LABELS_DIR, CLOAKED_DIR, FINETUNED_DIR, PRETRAINED_YOLOV5N,
    EPSILON_VALUES, CONF_THRESHOLD, IMG_SIZE, DEVICE
)
from adaptive_attack.utils import (
    load_yolov5_for_inference, load_finetuned_model,
    wrap_raw_model_for_inference, compute_dsr, load_yolo_label,
    get_train_test_split
)

EXCEL_PATH_TMPL = os.path.join(FINETUNED_DIR, "adaptive_eval_summary_{dataset}.xlsx")
CHART_PATH_TMPL = os.path.join(FINETUNED_DIR, "adaptive_eval_summary_{dataset}.png")


# ──────────────────────────────────────────────
# Detection metrics: Recall / Precision / mAP@0.5
# ──────────────────────────────────────────────

def _iou_xyxy(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _load_gt_xyxy(label_path, img_w, img_h):
    boxes = []
    for cls, xc, yc, w, h in load_yolo_label(label_path):
        x1 = (xc - w/2) * img_w
        y1 = (yc - h/2) * img_h
        x2 = (xc + w/2) * img_w
        y2 = (yc + h/2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def compute_detection_metrics(model, image_dir, label_dir, conf_thr=0.4,
                              iou_thr=0.5, img_size=416, strip_suffix="",
                              device="cuda:0", allowed_basenames=None):
    """
    Computes:
      - Recall / Precision  at fixed conf_thr (matches get_cloak.py's --conf convention)
      - mAP@0.5             integrated over all confidence thresholds present

    Returns dict with recall, precision, map50, n_images, n_gt, n_pred
    """
    from PIL import Image
    import torch

    exts = ("*.png", "*.jpg", "*.jpeg")
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    img_paths.sort()

    if allowed_basenames is not None:
        def _basename_of(p):
            b = os.path.splitext(os.path.basename(p))[0]
            if strip_suffix and b.endswith(strip_suffix):
                b = b[: -len(strip_suffix)]
            return b
        img_paths = [p for p in img_paths if _basename_of(p) in allowed_basenames]

    if not img_paths:
        return {"recall": 0.0, "precision": 0.0, "map50": 0.0,
               "n_images": 0, "n_gt": 0, "n_pred": 0}

    # Collect all predictions (with confidence) + all GT, per-image, for AP calc
    all_preds = []  # list of (confidence, is_tp) filled in after matching
    total_gt = 0
    total_pred_fixed_conf = 0
    total_tp_fixed_conf = 0

    with torch.no_grad():
        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            fname = os.path.splitext(os.path.basename(img_path))[0]
            if strip_suffix and fname.endswith(strip_suffix):
                fname = fname[: -len(strip_suffix)]
            label_path = os.path.join(label_dir, fname + ".txt")
            gt_boxes = _load_gt_xyxy(label_path, w, h)
            total_gt += len(gt_boxes)

            orig_conf = model.conf
            model.conf = 0.001  # low threshold: capture all preds for mAP curve
            results = model(np.array(img), size=img_size)  # matches get_cloak.py's size=MODEL_INPUT
            model.conf = orig_conf
            preds = results.xyxy[0].detach().cpu().numpy()  # x1,y1,x2,y2,conf,cls

            # sort by confidence desc for greedy matching (mAP + fixed-conf both need this)
            preds = preds[preds[:, 4].argsort()[::-1]]

            matched_gt = [False] * len(gt_boxes)
            for x1, y1, x2, y2, conf, cls in preds:
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(gt_boxes):
                    if matched_gt[j]:
                        continue
                    iou = _iou_xyxy([x1, y1, x2, y2], gt)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                is_tp = best_iou >= iou_thr
                if is_tp:
                    matched_gt[best_j] = True
                all_preds.append((float(conf), bool(is_tp)))

                if conf >= conf_thr:
                    total_pred_fixed_conf += 1
                    if is_tp:
                        total_tp_fixed_conf += 1

    # Fixed-confidence Recall / Precision (matches get_cloak.py convention)
    recall = total_tp_fixed_conf / total_gt if total_gt > 0 else 0.0
    precision = total_tp_fixed_conf / total_pred_fixed_conf if total_pred_fixed_conf > 0 else 0.0

    # mAP@0.5 — integrate precision-recall curve across all confidence levels
    map50 = _compute_ap(all_preds, total_gt)

    return {
        "recall": recall, "precision": precision, "map50": map50,
        "n_images": len(img_paths), "n_gt": total_gt,
        "n_pred": total_pred_fixed_conf,
    }


def _compute_ap(preds, total_gt):
    """
    preds: list of (confidence, is_tp), NOT necessarily sorted.
    Standard all-point interpolated AP (COCO/VOC-style), single class.
    """
    if total_gt == 0 or len(preds) == 0:
        return 0.0

    preds = sorted(preds, key=lambda x: x[0], reverse=True)
    tp = np.array([1 if p[1] else 0 for p in preds], dtype=np.float64)
    fp = 1 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / total_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-16)

    # Make precision envelope monotonically decreasing (standard AP trick)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate area under PR curve
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return float(ap)


# ──────────────────────────────────────────────
# Model set-up helper
# ──────────────────────────────────────────────

def _get_model_list(dataset_name, model_name):
    """
    Looks for:
      1. yolov5n (original) — Phase 1 baseline
      2. yolov5n_adaptive_{dataset}.pt — Phase 3 single-stage mixed fine-tune
         (matches your diagram: clean+cloaked -> one fine-tune -> new model)
    """
    adaptive_path = os.path.join(FINETUNED_DIR, f"{model_name}_adaptive_{dataset_name}.pt")

    models = [(f"{model_name} (original)", "pretrained", PRETRAINED_YOLOV5N)]
    if os.path.exists(adaptive_path):
        models.append((f"{model_name} (adaptive, mixed)", "local", adaptive_path))
    return models


def _load_model(kind, path, conf):
    if kind == "pretrained":
        return load_yolov5_for_inference(path, device=DEVICE, conf=conf)
    raw_model = load_finetuned_model(path, device=DEVICE)
    return wrap_raw_model_for_inference(raw_model, conf=conf)


# ──────────────────────────────────────────────
# Full evaluation: DSR (kept) + Recall/Precision/mAP (new) + Excel + chart
# ──────────────────────────────────────────────

def evaluate_full(dataset_name, model_name="yolov5n", epsilon=8, iou_thr=0.5,
                  test_ratio=0.2, split_seed=42):
    pipeline_start = time.time()

    clean_dir = DATASETS[dataset_name]
    label_dir = LABELS_DIR[dataset_name]
    cloaked_dir = CLOAKED_DIR[dataset_name]

    models = _get_model_list(dataset_name, model_name)

    # 80/20 split — evaluate ONLY on the held-out 20% test set, so results
    # reflect generalization, not memorization of the training frames.
    _, test_basenames = get_train_test_split(clean_dir, test_ratio=test_ratio, seed=split_seed)

    print(f"{'='*70}")
    print(f" Full Evaluation — DSR + Recall + Precision + mAP@0.5")
    print(f" Dataset: {dataset_name} | epsilon={epsilon}")
    print(f" Evaluating on HELD-OUT TEST SET only: {len(test_basenames)} frames "
          f"({test_ratio*100:.0f}% split, seed={split_seed})")
    print(f"{'='*70}")

    if not os.path.isdir(cloaked_dir):
        print(f"\n[ERROR] Cloaked images not found: {cloaked_dir}")
        print(f"  Generate cloaked images with the Invisibility Cloak pipeline "
              f"(get_cloak.py) and place them at this path first.")
        sys.exit(1)

    rows = []
    for name, kind, path in models:
        print(f"\n{'-'*50}\n {name}\n{'-'*50}")
        t_start = time.time()

        model = _load_model(kind, path, CONF_THRESHOLD)

        # DSR — test set only
        dsr, dsr_details = compute_dsr(model, cloaked_dir, CONF_THRESHOLD,
                                       img_size=IMG_SIZE, device=DEVICE,
                                       allowed_basenames=test_basenames,
                                       strip_suffix="_attack")
        n_defended = sum(1 for d in dsr_details.values() if d["defended"])
        n_total = len(dsr_details)

        # Recall / Precision / mAP on CLEAN test-set images
        metrics_clean = compute_detection_metrics(
            model, clean_dir, label_dir, conf_thr=CONF_THRESHOLD,
            iou_thr=iou_thr, img_size=IMG_SIZE, strip_suffix="", device=DEVICE,
            allowed_basenames=test_basenames,
        )
        # Recall / Precision / mAP on CLOAKED test-set images
        metrics_cloak = compute_detection_metrics(
            model, cloaked_dir, label_dir, conf_thr=CONF_THRESHOLD,
            iou_thr=iou_thr, img_size=IMG_SIZE, strip_suffix="_attack", device=DEVICE,
            allowed_basenames=test_basenames,
        )

        elapsed = time.time() - t_start

        row = {
            "Model": name,
            "DSR": round(dsr, 4),
            "Defended": n_defended,
            "Bypassed": n_total - n_defended,
            "Recall_clean": round(metrics_clean["recall"], 4),
            "Precision_clean": round(metrics_clean["precision"], 4),
            "mAP50_clean": round(metrics_clean["map50"], 4),
            "Recall_cloak": round(metrics_cloak["recall"], 4),
            "Precision_cloak": round(metrics_cloak["precision"], 4),
            "mAP50_cloak": round(metrics_cloak["map50"], 4),
            "Recall_drop": round(metrics_clean["recall"] - metrics_cloak["recall"], 4),
            "mAP50_drop": round(metrics_clean["map50"] - metrics_cloak["map50"], 4),
            "Eval_time_s": round(elapsed, 1),
        }
        rows.append(row)

        print(f"  DSR:              {dsr*100:6.2f}%  ({n_defended}/{n_total} defended)")
        print(f"  Recall  clean/cloak: {metrics_clean['recall']*100:6.2f}% / {metrics_cloak['recall']*100:6.2f}%")
        print(f"  Precision clean/cloak: {metrics_clean['precision']*100:6.2f}% / {metrics_cloak['precision']*100:6.2f}%")
        print(f"  mAP@0.5 clean/cloak: {metrics_clean['map50']*100:6.2f}% / {metrics_cloak['map50']*100:6.2f}%")
        print(f"  Time (this model, metrics only): {elapsed:.1f}s")

    total_eval_time = time.time() - pipeline_start

    df_new = pd.DataFrame(rows)
    df_new.insert(0, "Epsilon", epsilon)
    df_new.insert(0, "Dataset", dataset_name.upper())
    df_new["Total_pipeline_time_s"] = round(total_eval_time, 1)  # same value every row, for reference

    print(f"\n{'='*70}")
    print(f" TOTAL EVALUATION TIME: {total_eval_time:.1f}s ({total_eval_time/60:.2f} min) "
          f"for {len(models)} model(s)")
    print(f"{'='*70}")

    # ── Save / append to Excel ──
    excel_path = EXCEL_PATH_TMPL.format(dataset=dataset_name)
    if os.path.exists(excel_path):
        df_old = pd.read_excel(excel_path)
        df_old = df_old[~((df_old["Dataset"] == df_new["Dataset"].iloc[0]) &
                          (df_old["Epsilon"] == epsilon) &
                          (df_old["Model"].isin(df_new["Model"])))]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_excel(excel_path, index=False)
    print(f"\n Excel saved: {excel_path}")

    _plot_summary(df_new, dataset_name, epsilon)

    return df_new


def _plot_summary(df, dataset_name, epsilon):
    """4-panel chart: DSR, Recall, Precision, mAP50 — clean vs cloak where applicable."""
    models = df["Model"].tolist()
    x = np.arange(len(models))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Adaptive Attack Evaluation — {dataset_name.upper()} | epsilon={epsilon}/255",
                fontsize=13)

    # Panel 1: DSR
    ax = axes[0, 0]
    dsr_vals = (df["DSR"] * 100).tolist()
    bars = ax.bar(x, dsr_vals, color="steelblue", width=0.5)
    for b, v in zip(bars, dsr_vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{v:.1f}%",
               ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("DSR (%)"); ax.set_title("Defense Success Rate (higher = cloak wins)")
    ax.set_ylim(0, 105)

    # Panel 2: Recall clean vs cloak
    ax = axes[0, 1]
    r_clean = (df["Recall_clean"] * 100).tolist()
    r_cloak = (df["Recall_cloak"] * 100).tolist()
    ax.bar(x - w/2, r_clean, w, label="Clean", color="mediumseagreen")
    ax.bar(x + w/2, r_cloak, w, label="Cloaked", color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Recall (%)"); ax.set_title("Recall — Clean vs Cloaked")
    ax.legend(fontsize=8); ax.set_ylim(0, 105)

    # Panel 3: Precision clean vs cloak
    ax = axes[1, 0]
    p_clean = (df["Precision_clean"] * 100).tolist()
    p_cloak = (df["Precision_cloak"] * 100).tolist()
    ax.bar(x - w/2, p_clean, w, label="Clean", color="mediumseagreen")
    ax.bar(x + w/2, p_cloak, w, label="Cloaked", color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Precision (%)"); ax.set_title("Precision — Clean vs Cloaked")
    ax.legend(fontsize=8); ax.set_ylim(0, 105)

    # Panel 4: mAP@0.5 clean vs cloak
    ax = axes[1, 1]
    m_clean = (df["mAP50_clean"] * 100).tolist()
    m_cloak = (df["mAP50_cloak"] * 100).tolist()
    ax.bar(x - w/2, m_clean, w, label="Clean", color="mediumseagreen")
    ax.bar(x + w/2, m_cloak, w, label="Cloaked", color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("mAP@0.5 (%)"); ax.set_title("mAP@0.5 — Clean vs Cloaked")
    ax.legend(fontsize=8); ax.set_ylim(0, 105)

    plt.tight_layout()
    chart_path = CHART_PATH_TMPL.format(dataset=dataset_name)
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f" Chart saved: {chart_path}")


# ──────────────────────────────────────────────
# Kept for backward-compat: quick DSR-only check (used by run_all.py --only_eval)
# ──────────────────────────────────────────────

def evaluate_quick(dataset_name, model_name="yolov5n", epsilon=8):
    """Thin wrapper — now runs the full evaluation (DSR + Recall/Precision/mAP)."""
    return evaluate_full(dataset_name, model_name, epsilon)


def main():
    parser = argparse.ArgumentParser(description="Step 5: Full evaluation (DSR + Recall/Precision/mAP)")
    parser.add_argument("--dataset", type=str, default="cs2", choices=["cs2", "cf"])
    parser.add_argument("--model", type=str, default="yolov5n")
    parser.add_argument("--epsilon", type=int, default=8)
    parser.add_argument("--all_epsilon", action="store_true")
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Held-out test fraction, must match training (default: 0.2)")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Split seed, must match training (default: 42)")
    args = parser.parse_args()

    epsilons = EPSILON_VALUES if args.all_epsilon else [args.epsilon]
    for eps in epsilons:
        evaluate_full(args.dataset, args.model, eps, args.iou_thr,
                      test_ratio=args.test_ratio, split_seed=args.split_seed)


if __name__ == "__main__":
    main()