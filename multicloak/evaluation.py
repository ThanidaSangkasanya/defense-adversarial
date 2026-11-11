"""
eval_cloaked.py — Evaluate pre-cloaked images against cheating models
Loads already-cloaked PNG images and measures DSR, FPS, SSIM.
No refinement or noise generation — just pure evaluation.

Use after inject_cloak.py or inject_cloak_multirefine.py.

Usage:
    # Evaluate universal-only cloaked images:
    python eval_cloaked.py --game cs2 --cheating_model yolov5n yolov5s yolov5m

    # Evaluate multi-model refined images:
    python eval_cloaked.py --game cs2 --cheating_model yolov5n yolov5s yolov5m \
        --cloaked_dir C:\\Users\\tsang\\Dataset\\dataset_demo\\cs2\\frames_cloaked_multirefine_sum

    # Custom tag for result file naming:
    python eval_cloaked.py --game cs2 --cheating_model yolov5n yolov5s yolov5m \
        --cloaked_dir ... --tag multirefine_sum
"""

import os, sys, time, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_msssim import ssim as ssim_fn
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── ARGUMENTS ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--game',           required=True, type=str, choices=['cs2','cf'])
parser.add_argument('--cheating_model', nargs='+', required=True,
                    choices=['yolov5n','yolov5s','yolov5m','yolov5x'],
                    help='Cheating model(s) to evaluate against')
parser.add_argument('--cloaked_dir',    default=None, type=str,
                    help='Directory of cloaked images (default: frames_cloaked)')
parser.add_argument('--conf',      default=0.4,   type=float,
                    help='Confidence threshold (default: 0.4)')
parser.add_argument('--gpu',       default='0',   type=str)
parser.add_argument('--input_size', default=320,   type=int,
                    help='Detection input size (default: 320)')
parser.add_argument('--tag',       default='',     type=str,
                    help='Tag for result file naming')
args = parser.parse_args()

DEVICE = f'cuda:{args.gpu}'
GAME   = args.game

GAME_FOLDER = {'cs2': 'cs2', 'cf': 'cf'}

# ── Paths ─────────────────────────────────────────────────────
ORIG_PATH = os.path.join(r'C:\Users\tsang\Dataset\dataset_demo',
                         GAME_FOLDER[GAME], 'frames')
GT_PATH   = os.path.join(r'C:\Users\tsang\Dataset\dataset_demo',
                         GAME_FOLDER[GAME], 'labels')

if args.cloaked_dir:
    CLOAKED_PATH = args.cloaked_dir
else:
    CLOAKED_PATH = os.path.join(r'C:\Users\tsang\Dataset\dataset_demo',
                                GAME_FOLDER[GAME], 'frames_cloaked')

suffix     = f'_{args.tag}' if args.tag else ''
RESULT_DIR = 'result'
EVAL_XLSX  = os.path.join(RESULT_DIR, f'eval_cloaked{suffix}.xlsx')
EVAL_PNG   = os.path.join(RESULT_DIR, f'eval_cloaked{suffix}.png')
os.makedirs(RESULT_DIR, exist_ok=True)

print(f"\n{'='*65}")
print(f"  Game            : {GAME.upper()}")
print(f"  Cheating models : {args.cheating_model}")
print(f"  Original frames : {ORIG_PATH}")
print(f"  Cloaked frames  : {CLOAKED_PATH}")
print(f"  GT labels       : {GT_PATH}")
print(f"  Conf threshold  : {args.conf}")
print(f"{'='*65}\n")


# ─── MODEL LOADER ────────────────────────────────────────────
def load_cheating_model(variant):
    """Load YOLOv5 WITH AutoShape — for evaluation."""
    sys.path.insert(0, os.path.abspath('.'))
    from models.common import DetectMultiBackend, AutoShape
    pt_path = os.path.join('pretrained_models', f'{variant}.pt')
    model   = DetectMultiBackend(pt_path, device=torch.device(DEVICE), fuse=True)
    model   = AutoShape(model)
    model.conf    = args.conf
    model.classes = [0]
    print(f"  [Cheating] {variant} loaded (conf={args.conf})")
    return model


# ─── GT LOADER ────────────────────────────────────────────────
def load_gt_boxes(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or int(parts[0]) != 0:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), \
                           float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


# ─── DRAW BOXES ──────────────────────────────────────────────
def draw_boxes(pil_img, pred_boxes, gt_boxes=None):
    img  = pil_img.copy()
    draw = ImageDraw.Draw(img)
    if gt_boxes:
        for (x1, y1, x2, y2) in gt_boxes:
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
    for b in pred_boxes:
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        conf = b[4] if len(b) > 4 else 0
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, max(0, y1 - 15)), f'{conf:.2f}', fill='red')
    return img


# ─── EVALUATE ONE CHEATING MODEL ─────────────────────────────
def evaluate_model(cheating_name, orig_files, cloaked_files):
    """Evaluate cloaked images against one cheating model."""
    print(f"\n{'─'*55}")
    print(f"  Evaluating: {cheating_name}")
    print(f"{'─'*55}")

    model = load_cheating_model(cheating_name)

    # Visualization dirs
    vis_dir = os.path.join(RESULT_DIR, 'visualization', GAME,
                           f'eval_{cheating_name}{suffix}')
    box_dir = os.path.join(vis_dir, 'boxes')
    os.makedirs(box_dir, exist_ok=True)

    succ_num   = 0
    total_time = 0.0
    ssim_vals  = []
    log_rows   = []

    N = len(cloaked_files)
    for i, (orig_f, cloak_f) in enumerate(zip(orig_files, cloaked_files)):
        fname = os.path.splitext(os.path.basename(cloak_f))[0]

        # Load images
        orig_pil  = Image.open(orig_f).convert('RGB')
        cloak_pil = Image.open(cloak_f).convert('RGB')
        orig_w, orig_h = orig_pil.size

        # ── Run cheating model ────────────────────────────────
        t0 = time.time()
        cloak_np = np.array(cloak_pil)
        results  = model(cloak_np, size=args.input_size)
        t_eval   = time.time() - t0
        total_time += t_eval

        # Parse detections
        pred_boxes = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().tolist():
            if int(cls) == 0:
                pred_boxes.append([int(xyxy[0]), int(xyxy[1]),
                                   int(xyxy[2]), int(xyxy[3]), float(conf)])

        target_succ = 1 if len(pred_boxes) == 0 else 0
        succ_num += target_succ

        # ── SSIM ──────────────────────────────────────────────
        orig_t  = torch.from_numpy(np.array(orig_pil).transpose(2, 0, 1)).float().unsqueeze(0)
        cloak_t = torch.from_numpy(np.array(cloak_pil).transpose(2, 0, 1)).float().unsqueeze(0)
        ssim_val = ssim_fn(orig_t, cloak_t, data_range=255, size_average=False).item()
        ssim_vals.append(ssim_val)

        # Progress
        fps     = (i + 1) / total_time if total_time > 0 else 0
        dsr_now = succ_num / (i + 1)

        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == N:
            print(f"  [{i+1:4d}/{N}]  Det:{len(pred_boxes)}  "
                  f"DSR:{dsr_now:.3f}  SSIM:{ssim_val:.3f}  FPS:{fps:.1f}")

        # Save visualisation
        gt_label = os.path.join(GT_PATH, fname + '.txt')
        gt_boxes = load_gt_boxes(gt_label, orig_w, orig_h)
        vis = draw_boxes(cloak_pil, pred_boxes, gt_boxes)
        vis.save(os.path.join(box_dir, f'{fname}.jpg'))

        log_rows.append({
            'Index':      i,
            'File':       os.path.basename(cloak_f),
            'GT_boxes':   len(gt_boxes),
            'Detections': len(pred_boxes),
            'Success':    target_succ,
            'SSIM':       round(ssim_val, 4),
            'Time_s':     round(t_eval, 4),
        })

    # ── Per-image log ─────────────────────────────────────────
    log_dir  = os.path.join(RESULT_DIR, 'log', GAME)
    os.makedirs(log_dir, exist_ok=True)
    log_xlsx = os.path.join(log_dir, f'{GAME}_{cheating_name}_eval{suffix}.xlsx')
    pd.DataFrame(log_rows).to_excel(log_xlsx, index=False)
    print(f"  Log → {log_xlsx}")

    # ── Aggregate ─────────────────────────────────────────────
    final_dsr = succ_num / N
    final_fps = N / total_time if total_time > 0 else 0
    avg_ssim  = np.mean(ssim_vals)

    print(f"\n  ── {cheating_name} on {GAME.upper()} ──")
    print(f"  DSR  : {final_dsr*100:.2f}%")
    print(f"  FPS  : {final_fps:.2f}")
    print(f"  SSIM : {avg_ssim:.4f}")

    del model
    torch.cuda.empty_cache()

    return {
        'Cheating Model': cheating_name,
        'Dataset':        f'AVCA-{GAME.upper()}',
        'DSR(%)':         round(final_dsr * 100, 2),
        'FPS':            round(final_fps, 2),
        'SSIM':           round(avg_ssim, 4),
    }


# ─── SAVE RESULTS ────────────────────────────────────────────
def save_results(rows):
    df = pd.DataFrame(rows)
    df.to_excel(EVAL_XLSX, index=False)
    print(f"\n  Results → {EVAL_XLSX}")

    # Chart
    labels = [r['Cheating Model'] for r in rows]
    dsrs   = [r['DSR(%)'] for r in rows]
    fpss   = [r['FPS'] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    bars = ax.bar(labels, dsrs, color='steelblue', width=0.5)
    for bar, val in zip(bars, dsrs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('DSR (%)')
    ax.set_ylim(0, 115)
    ax.set_title('Defense Success Rate')

    ax = axes[1]
    bars = ax.bar(labels, fpss, color='darkorange', width=0.5)
    for bar, val in zip(bars, fpss):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(fpss) * 0.02, f'{val:.1f}',
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('FPS')
    ax.set_title('Evaluation Speed')

    plt.suptitle(f'Evaluation — {GAME.upper()}{suffix}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(EVAL_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart  → {EVAL_PNG}")


# ─── MAIN ────────────────────────────────────────────────────
def main():
    # List original and cloaked files
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    orig_files = sorted([os.path.join(ORIG_PATH, f)
                         for f in os.listdir(ORIG_PATH) if f.lower().endswith(exts)])
    cloak_files = sorted([os.path.join(CLOAKED_PATH, f)
                          for f in os.listdir(CLOAKED_PATH) if f.lower().endswith(exts)])

    print(f"  Original : {len(orig_files)} images")
    print(f"  Cloaked  : {len(cloak_files)} images")

    if len(orig_files) != len(cloak_files):
        print(f"  WARNING: count mismatch! Using min({len(orig_files)}, {len(cloak_files)})")
        n = min(len(orig_files), len(cloak_files))
        orig_files  = orig_files[:n]
        cloak_files = cloak_files[:n]

    all_rows = []
    for cheat_name in args.cheating_model:
        row = evaluate_model(cheat_name, orig_files, cloak_files)
        all_rows.append(row)

    print(f"\n{'='*65}")
    print(f"  SUMMARY — {GAME.upper()}")
    print(f"{'='*65}")
    summary_df = pd.DataFrame(all_rows)
    print(summary_df.to_string(index=False))

    save_results(all_rows)

    print(f"\n{'='*65}")
    print(f"  ALL DONE")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()