"""
Repeated 50/50 Holdout Evaluation for Adaptive Attack — NanoDet-Plus
======================================================================
Same methodology as finetune.py (YOLOv5n version), ported to NanoDet's
API. Per round:

  1. Shuffle cloaked images, split 50/50 -> Set A (eval-only) / Set B (finetune-only)
  2. Original NanoDet detects Set A -> BEFORE metrics
  3. Fine-tune a fresh NanoDet copy on Set B only
  4. Fine-tuned NanoDet detects the SAME Set A -> AFTER metrics
  5. Record before/after comparison

Repeat for N rounds with a different shuffle each time.

Usage:
    python adaptive_attack/finetune_nanodet.py --dataset cs2 --rounds 3
"""
import os
import sys
import argparse
import time
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_attack.config import NANODET_CLOAKED_DIR, LABELS_DIR, FINETUNED_DIR, DEVICE
from adaptive_attack.config import NANODET_FINETUNE_LR, NANODET_FINETUNE_EPOCHS, NANODET_BATCH_SIZE
from adaptive_attack.utils import load_yolo_label, save_detection_visualizations  # shared, generic utils
from adaptive_attack.utils_nanodet import (
    load_nanodet_for_inference, load_nanodet_for_training, build_val_pipeline,
    save_nanodet_checkpoint, load_nanodet_checkpoint,
    detect_nanodet, compute_dsr_nanodet, compute_nanodet_loss,
)

EXCEL_PATH_TMPL = os.path.join(FINETUNED_DIR, "repeated_split_eval_nanodet_{dataset}.xlsx")
CHART_PATH_TMPL = os.path.join(FINETUNED_DIR, "repeated_split_eval_nanodet_{dataset}.png")


def list_cloaked_basenames(cloaked_dir, strip_suffix="_attack"):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [f for f in os.listdir(cloaked_dir) if f.lower().endswith(exts)]
    basenames = []
    for f in files:
        b = os.path.splitext(f)[0]
        if strip_suffix and b.endswith(strip_suffix):
            b = b[: -len(strip_suffix)]
        basenames.append(b)
    return sorted(basenames)


def _gt_boxes_for(label_dir, basename, img_w, img_h):
    boxes = []
    for cls, xc, yc, bw, bh in load_yolo_label(os.path.join(label_dir, basename + ".txt")):
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def _iou_xyxy(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter/union if union > 0 else 0.0


def compute_metrics_nanodet(model, nano_cfg, pipeline, cloak_dir, label_dir,
                            conf_thr=0.4, iou_thr=0.5, allowed_basenames=None,
                            strip_suffix="_attack", device="cuda:0"):
    """Recall/Precision for NanoDet — same convention as evaluation.py's compute_detection_metrics."""
    from PIL import Image
    import glob

    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(cloak_dir, "**", ext), recursive=True))
    paths.sort()

    if allowed_basenames is not None:
        def _b(p):
            b = os.path.splitext(os.path.basename(p))[0]
            if strip_suffix and b.endswith(strip_suffix):
                b = b[:-len(strip_suffix)]
            return b
        paths = [p for p in paths if _b(p) in allowed_basenames]

    if not paths:
        return {"recall": 0.0, "precision": 0.0}

    total_gt, total_tp, total_pred = 0, 0, 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        b = os.path.splitext(os.path.basename(p))[0]
        if strip_suffix and b.endswith(strip_suffix):
            b = b[:-len(strip_suffix)]

        gt_boxes = _gt_boxes_for(label_dir, b, w, h)
        total_gt += len(gt_boxes)

        boxes = detect_nanodet(model, nano_cfg, pipeline, img, conf_thr, device)
        total_pred += len(boxes)

        matched = [False] * len(gt_boxes)
        for x1, y1, x2, y2, conf in boxes:
            for j, gb in enumerate(gt_boxes):
                if not matched[j] and _iou_xyxy([x1, y1, x2, y2], gb) >= iou_thr:
                    matched[j] = True
                    total_tp += 1
                    break

    recall = total_tp / total_gt if total_gt > 0 else 0.0
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    return {"recall": recall, "precision": precision}


def run_one_round(round_idx, dataset_name, cloaked_dir, label_dir, all_basenames,
                  epochs, lr, batch_size, save_models):
    seed = 1000 + round_idx
    rng = random.Random(seed)
    shuffled = all_basenames.copy()
    rng.shuffle(shuffled)

    half = len(shuffled) // 2
    set_a = set(shuffled[:half])
    set_b = set(shuffled[half:])

    print(f"\n{'#'*70}")
    print(f" [NanoDet] ROUND {round_idx} | shuffle seed={seed}")
    print(f" Set A (eval-only): {len(set_a)} | Set B (finetune-only): {len(set_b)}")
    print(f"{'#'*70}")

    t_start = time.time()

    # 1. BEFORE — original model on Set A
    print(f"\n[Round {round_idx}] Evaluating ORIGINAL NanoDet on Set A...")
    model, nano_cfg, pipeline = load_nanodet_for_inference(device=DEVICE)

    dsr_before, dsr_details_before = compute_dsr_nanodet(
        model, nano_cfg, pipeline, cloaked_dir, conf_threshold=0.4,
        device=DEVICE, allowed_basenames=set_a,
    )
    metrics_before = compute_metrics_nanodet(
        model, nano_cfg, pipeline, cloaked_dir, label_dir,
        allowed_basenames=set_a, device=DEVICE,
    )
    n_total_a = len(dsr_details_before)
    n_defended_before = sum(1 for d in dsr_details_before.values() if d["defended"])
    print(f"  BEFORE — DSR: {dsr_before*100:.2f}% ({n_defended_before}/{n_total_a}) | "
          f"Recall: {metrics_before['recall']*100:.2f}% | "
          f"Precision: {metrics_before['precision']*100:.2f}%")

    # 2. Fine-tune fresh model on Set B
    print(f"\n[Round {round_idx}] Fine-tuning NanoDet on Set B ({len(set_b)} images)...")
    train_model, train_cfg, train_pipeline = load_nanodet_for_training(device=DEVICE)

    from PIL import Image
    set_b_list = sorted(set_b)
    exts = (".png", ".jpg", ".jpeg", ".bmp")

    def find_path(basename):
        for f in os.listdir(cloaked_dir):
            if os.path.splitext(f)[0] == f"{basename}_attack" and f.lower().endswith(exts):
                return os.path.join(cloaked_dir, f)
        return None

    optimizer = optim.SGD(train_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    n_train = int(len(set_b_list) * 0.9)
    train_names, val_names = set_b_list[:n_train], set_b_list[n_train:]

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        random.Random(epoch).shuffle(train_names)
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, len(train_names), batch_size):
            batch_names = train_names[i:i+batch_size]
            imgs, gts = [], []
            for bn in batch_names:
                p = find_path(bn)
                if p is None:
                    continue
                img = Image.open(p).convert("RGB")
                w, h = img.size
                gts.append(_gt_boxes_for(label_dir, bn, w, h))
                imgs.append(img)
            if not imgs:
                continue

            optimizer.zero_grad()
            try:
                loss = compute_nanodet_loss(train_model, train_cfg, train_pipeline,
                                            imgs, gts, device=DEVICE)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            except Exception as e:
                print(f"    [WARNING] batch error: {e}")

        avg_train = epoch_loss / max(n_batches, 1)

        # validation
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_names), batch_size):
                batch_names = val_names[i:i+batch_size]
                imgs, gts = [], []
                for bn in batch_names:
                    p = find_path(bn)
                    if p is None:
                        continue
                    img = Image.open(p).convert("RGB")
                    w, h = img.size
                    gts.append(_gt_boxes_for(label_dir, bn, w, h))
                    imgs.append(img)
                if not imgs:
                    continue
                try:
                    loss = compute_nanodet_loss(train_model, train_cfg, train_pipeline,
                                                imgs, gts, device=DEVICE)
                    val_loss += loss.item()
                    val_batches += 1
                except Exception:
                    continue
        avg_val = val_loss / max(val_batches, 1)
        print(f"    Epoch {epoch:3d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.detach().clone() for k, v in train_model.state_dict().items()}

    if best_state is not None:
        train_model.load_state_dict(best_state)
    train_model.eval()

    if save_models:
        model_path = os.path.join(FINETUNED_DIR, f"nanodet_round{round_idx}_{dataset_name}.pt")
        save_nanodet_checkpoint(train_model, model_path)

    # 3. AFTER — fine-tuned model on same Set A
    # Use a FRESH val pipeline for evaluation, NOT the train_pipeline
    # (train pipeline has augmentation / different settings, unsuitable
    # for consistent before/after comparison).
    print(f"\n[Round {round_idx}] Evaluating FINE-TUNED NanoDet on Set A...")
    val_pipeline = build_val_pipeline(train_cfg)

    dsr_after, dsr_details_after = compute_dsr_nanodet(
        train_model, train_cfg, val_pipeline, cloaked_dir, conf_threshold=0.4,
        device=DEVICE, allowed_basenames=set_a,
    )
    metrics_after = compute_metrics_nanodet(
        train_model, train_cfg, val_pipeline, cloaked_dir, label_dir,
        allowed_basenames=set_a, device=DEVICE,
    )
    n_defended_after = sum(1 for d in dsr_details_after.values() if d["defended"])
    print(f"  AFTER  — DSR: {dsr_after*100:.2f}% ({n_defended_after}/{n_total_a}) | "
          f"Recall: {metrics_after['recall']*100:.2f}% | "
          f"Precision: {metrics_after['precision']*100:.2f}%")

    round_time = time.time() - t_start
    print(f"\n[Round {round_idx}] Total round time: {round_time:.1f}s")

    return {
        "Round": round_idx, "Split_seed": seed,
        "SetA_size": len(set_a), "SetB_size": len(set_b),
        "DSR_before": round(dsr_before, 4), "DSR_after": round(dsr_after, 4),
        "DSR_drop": round(dsr_before - dsr_after, 4),
        "Recall_before": round(metrics_before["recall"], 4),
        "Recall_after": round(metrics_after["recall"], 4),
        "Precision_before": round(metrics_before["precision"], 4),
        "Precision_after": round(metrics_after["precision"], 4),
        "Round_time_s": round(round_time, 1),
    }


def run_repeated_eval(dataset_name, n_rounds=3, epochs=None, lr=None,
                      batch_size=None, save_models=False):
    epochs = epochs or NANODET_FINETUNE_EPOCHS
    lr = lr or NANODET_FINETUNE_LR
    batch_size = batch_size or NANODET_BATCH_SIZE

    cloaked_dir = NANODET_CLOAKED_DIR[dataset_name]
    label_dir = LABELS_DIR[dataset_name]

    if not os.path.isdir(cloaked_dir):
        print(f"[ERROR] Cloaked images not found: {cloaked_dir}")
        print(f"  Generate them first: python get_cloak.py --game {dataset_name} --model nanodet")
        sys.exit(1)

    all_basenames = list_cloaked_basenames(cloaked_dir)
    print(f"{'='*70}")
    print(f" [NanoDet] Repeated 50/50 Holdout Evaluation — {dataset_name.upper()}")
    print(f" Total cloaked images: {len(all_basenames)}")
    print(f" Rounds: {n_rounds} | Epochs/round: {epochs} | LR: {lr}")
    print(f"{'='*70}")

    pipeline_start = time.time()
    rows = []
    for r in range(1, n_rounds + 1):
        row = run_one_round(r, dataset_name, cloaked_dir, label_dir,
                            all_basenames, epochs, lr, batch_size, save_models)
        rows.append(row)

    total_time = time.time() - pipeline_start
    df = pd.DataFrame(rows)

    summary, summary_std = {"Round": "MEAN"}, {"Round": "STD"}
    for col in ["DSR_before", "DSR_after", "DSR_drop", "Recall_before", "Recall_after",
               "Precision_before", "Precision_after", "Round_time_s"]:
        summary[col] = round(df[col].mean(), 4)
        summary_std[col] = round(df[col].std(), 4)

    df_out = pd.concat([df, pd.DataFrame([summary, summary_std])], ignore_index=True)
    df_out.insert(0, "Dataset", dataset_name.upper())
    df_out.insert(1, "Model", "nanodet")

    print(f"\n{'='*70}\n SUMMARY across {n_rounds} rounds (NanoDet-Plus)\n{'='*70}")
    print(f" DSR:  before={summary['DSR_before']*100:.2f}%  after={summary['DSR_after']*100:.2f}%")
    print(f" Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    os.makedirs(FINETUNED_DIR, exist_ok=True)
    excel_path = EXCEL_PATH_TMPL.format(dataset=dataset_name)
    df_out.to_excel(excel_path, index=False)
    print(f"\n Excel saved: {excel_path}")

    _plot_rounds(df, dataset_name, n_rounds)
    return df_out


def _plot_rounds(df, dataset_name, n_rounds):
    rounds = df["Round"].tolist()
    x = np.arange(len(rounds))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"NanoDet-Plus — Repeated 50/50 Holdout — {dataset_name.upper()} ({n_rounds} rounds)")

    panels = [("DSR_before", "DSR_after", "DSR (%)", axes[0]),
             ("Recall_before", "Recall_after", "Recall (%)", axes[1]),
             ("Precision_before", "Precision_after", "Precision (%)", axes[2])]

    for before_col, after_col, ylabel, ax in panels:
        ax.bar(x - w/2, (df[before_col]*100).tolist(), w, label="Before", color="steelblue")
        ax.bar(x + w/2, (df[after_col]*100).tolist(), w, label="After", color="tomato")
        ax.set_xticks(x); ax.set_xticklabels([f"Round {r}" for r in rounds])
        ax.set_ylabel(ylabel); ax.set_title(ylabel.split(" ")[0])
        ax.legend(fontsize=8); ax.set_ylim(0, 105)

    plt.tight_layout()
    chart_path = CHART_PATH_TMPL.format(dataset=dataset_name)
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f" Chart saved: {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Repeated 50/50 holdout — NanoDet-Plus")
    parser.add_argument("--dataset", type=str, default="cs2", choices=["cs2", "cf"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_models", action="store_true")
    args = parser.parse_args()

    run_repeated_eval(args.dataset, args.rounds, args.epochs, args.lr,
                      args.batch_size, args.save_models)


if __name__ == "__main__":
    main()