"""
Repeated 20/80 Holdout Evaluation for Adaptive Attack
=======================================================
Paper spec: "we use 20% of the datasets for adversarial training
because we assume cheaters have much fewer collected samples than protectors."

  Set A (80%)  -> held out, used ONLY for evaluation
  Set B (20%)  -> used ONLY to fine-tune

Per round:
  1. Shuffle all cloaked images, split into Set A 80% / Set B 20%
     (different shuffle each round)
  2. Run the ORIGINAL (pretrained) aimbot on Set A -> BEFORE metrics
     (DSR, Recall, Precision, mAP@0.5)
  3. Fine-tune a FRESH copy of the original aimbot on Set B only
  4. Run the FINE-TUNED aimbot on Set A (same held-out set as step 2,
     never touched during fine-tuning) -> AFTER metrics
  5. Record the before/after comparison for this round

Repeat for N rounds (default 3) with a different shuffle each time.

All outputs (Excel, Chart, Visualizations) go under:
  adaptive_attack/finetuned_models/adaptive_original/

Usage:
    python adaptive_original/finetune.py --dataset cs2
    python adaptive_original/finetune.py --dataset cs2 --rounds 3 --epochs 10
    python adaptive_original/finetune.py --dataset cs2 --finetune_ratio 0.2
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_original.config import (
    CLOAKED_DIR, LABELS_DIR, PRETRAINED_YOLOV5N, FINETUNED_DIR,
    ADAPTIVE_ORIGINAL_DIR,
    IMG_SIZE, CONF_THRESHOLD, DEVICE
)
from adaptive_original.utils import (
    AimbotDataset, load_yolov5n_raw, load_yolov5_for_inference,
    wrap_raw_model_for_inference, train_detection_model,
    compute_dsr, save_full_model, save_detection_visualizations,
)
from adaptive_original.evaluation import compute_detection_metrics

# ── All outputs go inside adaptive_original/ ──
EXCEL_PATH_TMPL = os.path.join(ADAPTIVE_ORIGINAL_DIR, "adaptive_original_{dataset}.xlsx")
CHART_PATH_TMPL = os.path.join(ADAPTIVE_ORIGINAL_DIR, "adaptive_original_{dataset}.png")


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


def run_one_round(round_idx, dataset_name, cloaked_dir, label_dir,
                  all_basenames, epochs, lr, batch_size,
                  save_models, model_name="yolov5n", viz_images=10,
                  finetune_ratio=0.2):
    """
    finetune_ratio=0.2 means:
      - Set B (fine-tune) = 20% of data   <- paper: "cheaters have fewer samples"
      - Set A (eval)      = 80% of data   <- larger held-out set for robust evaluation
    """
    seed = 1000 + round_idx  # different shuffle every round
    rng = random.Random(seed)
    shuffled = all_basenames.copy()
    rng.shuffle(shuffled)

    # 20% fine-tune, 80% eval (per paper)
    n_finetune = max(1, int(len(shuffled) * finetune_ratio))
    set_b = set(shuffled[:n_finetune])   # fine-tuning (20%)
    set_a = set(shuffled[n_finetune:])   # held-out eval (80%)

    print(f"\n{'#'*70}")
    print(f" ROUND {round_idx} | shuffle seed={seed} | finetune_ratio={finetune_ratio}")
    print(f" Set A (eval-only):     {len(set_a)} images ({(1-finetune_ratio)*100:.0f}%)")
    print(f" Set B (finetune-only): {len(set_b)} images ({finetune_ratio*100:.0f}%)")
    print(f"{'#'*70}")

    t_round_start = time.time()

    # ── 1. BEFORE: original model on Set A ──
    print(f"\n[Round {round_idx}] Evaluating ORIGINAL model on Set A (before fine-tune)...")
    original_model = load_yolov5_for_inference(PRETRAINED_YOLOV5N, device=DEVICE, conf=CONF_THRESHOLD)

    dsr_before, dsr_details_before = compute_dsr(
        original_model, cloaked_dir, CONF_THRESHOLD, img_size=IMG_SIZE,
        device=DEVICE, allowed_basenames=set_a, strip_suffix="_attack",
    )
    metrics_before = compute_detection_metrics(
        original_model, cloaked_dir, label_dir, conf_thr=CONF_THRESHOLD,
        iou_thr=0.5, img_size=IMG_SIZE, strip_suffix="_attack", device=DEVICE,
        allowed_basenames=set_a,
    )

    n_total_a = len(dsr_details_before)
    n_defended_before = sum(1 for d in dsr_details_before.values() if d["defended"])
    print(f"  BEFORE — DSR: {dsr_before*100:.2f}% ({n_defended_before}/{n_total_a} defended) | "
          f"Recall: {metrics_before['recall']*100:.2f}% | "
          f"Precision: {metrics_before['precision']*100:.2f}% | "
          f"mAP@0.5: {metrics_before['map50']*100:.2f}%")

    if viz_images > 0:
        viz_dir_before = os.path.join(ADAPTIVE_ORIGINAL_DIR,
                                      f"round{round_idx}_{dataset_name}", "before")
        saved_before = save_detection_visualizations(
            original_model, cloaked_dir, label_dir, viz_dir_before,
            basenames=set_a, conf_threshold=CONF_THRESHOLD,
            strip_suffix="_attack", max_images=viz_images,
        )
        print(f"  Saved {len(saved_before)} BEFORE visualization images -> {viz_dir_before}")

    # ── 2. Fine-tune a FRESH model on Set B (20%) ──
    print(f"\n[Round {round_idx}] Fine-tuning FRESH model on Set B "
          f"({len(set_b)} cloaked images, {finetune_ratio*100:.0f}%)...")
    train_model = load_yolov5n_raw(device=DEVICE)

    train_dataset = AimbotDataset(
        image_dir=cloaked_dir, label_dir=label_dir, img_size=IMG_SIZE,
        subset_ratio=1.0, strip_suffix="_attack", allowed_basenames=set_b,
    )
    if len(train_dataset) == 0:
        print(f"  [ERROR] Empty training set for round {round_idx}. Skipping.")
        return None

    train_model, history = train_detection_model(
        train_model, train_dataset, epochs=epochs, lr=lr,
        batch_size=batch_size, val_split=0.1, device=DEVICE, verbose=True,
    )

    if save_models:
        model_path = os.path.join(ADAPTIVE_ORIGINAL_DIR,
                                  f"{model_name}_round{round_idx}_{dataset_name}.pt")
        save_full_model(train_model, model_path)

    # ── 3. AFTER: fine-tuned model on the SAME Set A (80%) ──
    print(f"\n[Round {round_idx}] Evaluating FINE-TUNED model on Set A (after fine-tune)...")
    finetuned_model = wrap_raw_model_for_inference(train_model, conf=CONF_THRESHOLD)

    dsr_after, dsr_details_after = compute_dsr(
        finetuned_model, cloaked_dir, CONF_THRESHOLD, img_size=IMG_SIZE,
        device=DEVICE, allowed_basenames=set_a, strip_suffix="_attack",
    )
    metrics_after = compute_detection_metrics(
        finetuned_model, cloaked_dir, label_dir, conf_thr=CONF_THRESHOLD,
        iou_thr=0.5, img_size=IMG_SIZE, strip_suffix="_attack", device=DEVICE,
        allowed_basenames=set_a,
    )

    n_defended_after = sum(1 for d in dsr_details_after.values() if d["defended"])
    print(f"  AFTER  — DSR: {dsr_after*100:.2f}% ({n_defended_after}/{n_total_a} defended) | "
          f"Recall: {metrics_after['recall']*100:.2f}% | "
          f"Precision: {metrics_after['precision']*100:.2f}% | "
          f"mAP@0.5: {metrics_after['map50']*100:.2f}%")

    if viz_images > 0:
        viz_dir_after = os.path.join(ADAPTIVE_ORIGINAL_DIR,
                                     f"round{round_idx}_{dataset_name}", "after")
        saved_after = save_detection_visualizations(
            finetuned_model, cloaked_dir, label_dir, viz_dir_after,
            basenames=set_a, conf_threshold=CONF_THRESHOLD,
            strip_suffix="_attack", max_images=viz_images,
        )
        print(f"  Saved {len(saved_after)} AFTER visualization images -> {viz_dir_after}")

    round_time = time.time() - t_round_start
    print(f"\n[Round {round_idx}] Total round time: {round_time:.1f}s")

    row = {
        "Round": round_idx,
        "Split_seed": seed,
        "SetA_size (eval 80%)": len(set_a),
        "SetB_size (finetune 20%)": len(set_b),
        "DSR_before": round(dsr_before, 4),
        "DSR_after": round(dsr_after, 4),
        "DSR_drop": round(dsr_before - dsr_after, 4),
        "Recall_before": round(metrics_before["recall"], 4),
        "Recall_after": round(metrics_after["recall"], 4),
        "Precision_before": round(metrics_before["precision"], 4),
        "Precision_after": round(metrics_after["precision"], 4),
        "mAP50_before": round(metrics_before["map50"], 4),
        "mAP50_after": round(metrics_after["map50"], 4),
        "Round_time_s": round(round_time, 1),
    }
    return row


def run_repeated_eval(dataset_name, model_name="yolov5n", n_rounds=3,
                      epochs=10, lr=3e-5, batch_size=16,
                      save_models=False, viz_images=10,
                      finetune_ratio=0.2):
    """
    finetune_ratio=0.2 -> 20% fine-tune / 80% eval (paper default)
    All outputs saved under ADAPTIVE_ORIGINAL_DIR.
    """
    cloaked_dir = CLOAKED_DIR[dataset_name]
    label_dir = LABELS_DIR[dataset_name]

    if not os.path.isdir(cloaked_dir):
        print(f"[ERROR] Cloaked images not found: {cloaked_dir}")
        print(f"  Generate cloaked images with the Invisibility Cloak pipeline "
              f"(get_cloak.py) and place them at this path first.")
        sys.exit(1)

    os.makedirs(ADAPTIVE_ORIGINAL_DIR, exist_ok=True)

    all_basenames = list_cloaked_basenames(cloaked_dir)
    print(f"{'='*70}")
    print(f" Repeated Holdout Evaluation — {dataset_name.upper()}")
    print(f" Output folder: {ADAPTIVE_ORIGINAL_DIR}")
    print(f" Split: {finetune_ratio*100:.0f}% fine-tune / {(1-finetune_ratio)*100:.0f}% eval  "
          f"(paper: cheaters have fewer samples)")
    print(f" Total cloaked images: {len(all_basenames)}")
    print(f" Rounds: {n_rounds} | Fine-tune epochs/round: {epochs} | LR: {lr}")
    print(f"{'='*70}")

    if len(all_basenames) < 4:
        print(f"[ERROR] Not enough cloaked images ({len(all_basenames)}) for split.")
        sys.exit(1)

    pipeline_start = time.time()
    rows = []
    for r in range(1, n_rounds + 1):
        row = run_one_round(
            r, dataset_name, cloaked_dir, label_dir, all_basenames,
            epochs, lr, batch_size, save_models, model_name, viz_images,
            finetune_ratio=finetune_ratio,
        )
        if row is not None:
            rows.append(row)

    total_time = time.time() - pipeline_start

    if not rows:
        print("[ERROR] No successful rounds.")
        return None

    df = pd.DataFrame(rows)

    # Summary rows: mean + std across rounds
    numeric_cols = ["DSR_before", "DSR_after", "DSR_drop",
                    "Recall_before", "Recall_after",
                    "Precision_before", "Precision_after",
                    "mAP50_before", "mAP50_after", "Round_time_s"]

    summary = {"Round": "MEAN",
               "Split_seed": "-",
               "SetA_size (eval 80%)": "-",
               "SetB_size (finetune 20%)": "-"}
    for col in numeric_cols:
        summary[col] = round(df[col].mean(), 4)

    summary_std = {"Round": "STD",
                   "Split_seed": "-",
                   "SetA_size (eval 80%)": "-",
                   "SetB_size (finetune 20%)": "-"}
    for col in numeric_cols:
        summary_std[col] = round(df[col].std(), 4)

    df_out = pd.concat([df, pd.DataFrame([summary, summary_std])], ignore_index=True)
    df_out.insert(0, "Dataset", dataset_name.upper())

    print(f"\n{'='*70}")
    print(f" SUMMARY across {n_rounds} rounds  "
          f"[{finetune_ratio*100:.0f}% fine-tune / {(1-finetune_ratio)*100:.0f}% eval]")
    print(f"{'='*70}")
    print(f" DSR:       before={summary['DSR_before']*100:.2f}%  after={summary['DSR_after']*100:.2f}%  "
          f"(std before={summary_std['DSR_before']*100:.2f}pp, after={summary_std['DSR_after']*100:.2f}pp)")
    print(f" Recall:    before={summary['Recall_before']*100:.2f}%  after={summary['Recall_after']*100:.2f}%")
    print(f" Precision: before={summary['Precision_before']*100:.2f}%  after={summary['Precision_after']*100:.2f}%")
    print(f" mAP@0.5:   before={summary['mAP50_before']*100:.2f}%  after={summary['mAP50_after']*100:.2f}%")
    print(f" Total pipeline time: {total_time:.1f}s ({total_time/60:.1f} min)")

    excel_path = EXCEL_PATH_TMPL.format(dataset=dataset_name)
    df_out.to_excel(excel_path, index=False)
    print(f"\n Excel saved: {excel_path}")

    _plot_rounds(df, dataset_name, n_rounds, finetune_ratio)

    return df_out


def _plot_rounds(df, dataset_name, n_rounds, finetune_ratio=0.2):
    rounds = df["Round"].tolist()
    x = np.arange(len(rounds))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Adaptive Original — {dataset_name.upper()} ({n_rounds} rounds) | "
        f"{finetune_ratio*100:.0f}% fine-tune / {(1-finetune_ratio)*100:.0f}% eval",
        fontsize=13
    )

    panels = [
        ("DSR_before", "DSR_after", "DSR (%)", axes[0, 0]),
        ("Recall_before", "Recall_after", "Recall (%)", axes[0, 1]),
        ("Precision_before", "Precision_after", "Precision (%)", axes[1, 0]),
        ("mAP50_before", "mAP50_after", "mAP@0.5 (%)", axes[1, 1]),
    ]

    for before_col, after_col, ylabel, ax in panels:
        before_vals = (df[before_col] * 100).tolist()
        after_vals = (df[after_col] * 100).tolist()
        ax.bar(x - w/2, before_vals, w, label="Before fine-tune", color="steelblue")
        ax.bar(x + w/2, after_vals, w, label="After fine-tune", color="tomato")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Round {r}" for r in rounds])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split(" ")[0])
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    chart_path = CHART_PATH_TMPL.format(dataset=dataset_name)
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f" Chart saved: {chart_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Original — 20%% fine-tune / 80%% eval per paper"
    )
    parser.add_argument("--dataset", type=str, default="cs2", choices=["cs2", "cf"])
    parser.add_argument("--model", type=str, default="yolov5n")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_models", action="store_true",
                        help="Save each round's fine-tuned model to disk")
    parser.add_argument("--viz_images", type=int, default=10,
                        help="Number of before/after sample images to save with "
                             "GT (green) + predicted (red) boxes drawn, per round. "
                             "Set to 0 to disable.")
    parser.add_argument("--finetune_ratio", type=float, default=0.2,
                        help="Fraction used for fine-tuning (default: 0.2 = 20%% per paper). "
                             "Remaining fraction is held out for evaluation.")
    args = parser.parse_args()

    run_repeated_eval(
        args.dataset, args.model, args.rounds, args.epochs,
        args.lr, args.batch_size, args.save_models, args.viz_images,
        finetune_ratio=args.finetune_ratio,
    )


if __name__ == "__main__":
    main()