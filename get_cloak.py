"""
get_cloak.py — Apply Universal Cloak & Evaluate (v3)
เพิ่ม Recall metric จาก ground truth labels

Usage:
    python get_cloak.py --game cs2 --model yolov5n
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pytorch_msssim import ssim as ssim_fn
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── ARGUMENTS ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--game',    required=True, type=str,
                    choices=['cs2','cf','valorant','overwatch'])
parser.add_argument('--model',   required=True, type=str,
                    choices=['yolov5n','nanodet','rtdetr'])
parser.add_argument('--epsilon', default=8,    type=int)
parser.add_argument('--conf',       default=0.4,  type=float,
                    help='Confidence threshold for DSR (default: 0.4)')
parser.add_argument('--conf_recall', default=0.2,  type=float,
                    help='Confidence threshold for Recall (default: 0.2)')
parser.add_argument('--iou_thr', default=0.5,  type=float,
                    help='IoU threshold for TP (default: 0.5)')
parser.add_argument('--gpu',     default='0',  type=str)
parser.add_argument('--tag',     default='',   type=str,
                    help='suffix for output files e.g. demo')
args = parser.parse_args()

DEVICE = f'cuda:{args.gpu}'
EPS    = args.epsilon / 255.0
GAME   = args.game
MODEL  = args.model

MODEL_SIZE  = {'yolov5n':320, 'nanodet':320, 'rtdetr':640}
MODEL_INPUT = MODEL_SIZE[MODEL]

GAME_FOLDER = {'cs2':'CS2','cf':'CF','valorant':'Valorant','overwatch':'Overwatch'}
DATA_PATH   = os.path.join(r'C:\Users\tsang\Dataset', GAME_FOLDER[GAME])
# DATA_PATH = r'C:\Users\tsang\Downloads\Demo_CF'
GT_PATH     = os.path.join(r'C:\Users\tsang\Dataset\dataset_demo',
                            GAME_FOLDER[GAME], 'labels')
NOISE_PATH  = os.path.join('universal_cloak', GAME, MODEL, 'universal_noise.pt')

# Baseline Recall (before cloak) at conf=0.2 from Phase 1 evaluation
BASELINE_RECALL = {
    ('yolov5n', 'cs2'):       0.9427,
    ('yolov5n', 'cf'):        0.7500,
    ('yolov5n', 'valorant'):  0.6792,
    ('yolov5n', 'overwatch'): 0.1268,
    ('rtdetr',  'cs2'):       0.9868,
    ('rtdetr',  'cf'):        1.0000,
    ('rtdetr',  'valorant'):  0.8755,
    ('rtdetr',  'overwatch'): 0.3152,
    ('nanodet', 'cs2'):       0.9824,
    ('nanodet', 'cf'):        0.9900,
    ('nanodet', 'valorant'):  0.7509,
    ('nanodet', 'overwatch'): 0.2138,
}
RECALL_BEFORE = BASELINE_RECALL.get((MODEL, GAME), None)

suffix           = f'_{args.tag}' if args.tag else ''
ATTACK_DIR       = os.path.join('result','evaluation', GAME, MODEL, f'attack{suffix}')
ATTACK_CLEAN_DIR = os.path.join('result','evaluation', GAME, MODEL, f'attack_clean{suffix}')
LOG_DIR     = os.path.join('result','log', GAME)
os.makedirs(ATTACK_DIR,       exist_ok=True)
os.makedirs(ATTACK_CLEAN_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

EVAL_XLSX = os.path.join('result', f'evaluation_summary{suffix}.xlsx')
EVAL_PNG  = os.path.join('result', f'evaluation_summary{suffix}.png')

print(f"\n{'='*55}")
print(f"  Game  : {GAME}  |  Model : {MODEL}")
print(f"  Data  : {DATA_PATH}")
print(f"  GT    : {GT_PATH}")
print(f"  Noise : {NOISE_PATH}")
print(f"{'='*55}\n")

# Load noise
if not os.path.exists(NOISE_PATH):
    raise FileNotFoundError(f"Noise not found: {NOISE_PATH}\nRun train_cloak.py first.")
noise_model_size = torch.load(NOISE_PATH).to(DEVICE)
print(f"  Noise loaded: {noise_model_size.shape}")


# ─── GROUND TRUTH LOADER ─────────────────────────────────────
def load_gt_boxes(label_path, img_w, img_h):
    """
    Load YOLO format labels → xyxy pixel coords
    YOLO format: class cx cy w h (normalized)
    Returns list of [x1, y1, x2, y2]
    """
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls != 0:  # person only
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), \
                           float(parts[3]), float(parts[4])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


# ─── IoU ─────────────────────────────────────────────────────
def compute_iou(box1, box2):
    """box: [x1,y1,x2,y2]"""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_tp_fn(gt_boxes, pred_boxes, iou_thr=0.5):
    """
    Returns TP, FN for one image
    TP = GT boxes matched by a pred box (IoU >= iou_thr)
    FN = GT boxes not matched
    """
    if len(gt_boxes) == 0:
        return 0, 0
    matched = [False] * len(gt_boxes)
    for pb in pred_boxes:
        for j, gb in enumerate(gt_boxes):
            if not matched[j] and compute_iou(pb[:4], gb) >= iou_thr:
                matched[j] = True
                break
    tp = sum(matched)
    fn = len(gt_boxes) - tp
    return tp, fn


# ─── LOAD DATASET ────────────────────────────────────────────
def load_image_paths(folder):
    exts  = ('.jpg','.jpeg','.png','.bmp')
    paths = sorted([os.path.join(folder,f) for f in os.listdir(folder)
                    if f.lower().endswith(exts)])
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    print(f"  Found {len(paths)} images")
    return paths


# ─── APPLY NOISE ─────────────────────────────────────────────
def apply_noise_original(img_pil):
    orig_w, orig_h = img_pil.size
    img_t = torch.from_numpy(
        np.array(img_pil).transpose(2,0,1)
    ).float().unsqueeze(0).to(DEVICE) / 255.0

    noise_orig = F.interpolate(
        noise_model_size.unsqueeze(0),
        size=(orig_h, orig_w),
        mode='bilinear', align_corners=False
    ).squeeze(0)

    noise_clamped = torch.clamp(noise_orig, -EPS, EPS)
    adv_t  = torch.clamp(img_t + noise_clamped.unsqueeze(0), 0.0, 1.0)
    adv_np = (adv_t[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    return adv_pil, adv_t, img_t


# ─── MODEL LOADERS ───────────────────────────────────────────
def load_yolov5n():
    sys.path.insert(0, os.path.abspath('.'))
    from models.common import DetectMultiBackend, AutoShape
    model = DetectMultiBackend(
        os.path.join('pretrained_models','yolov5n.pt'),
        device=torch.device(DEVICE), fuse=True)
    model = AutoShape(model)
    model.conf    = args.conf
    model.classes = [0]
    print("  [YOLOv5n] Loaded.")
    return model

def load_nanodet():
    root = r'C:\Users\tsang\nanodet'
    sys.path.insert(0, root)
    from nanodet.util import cfg, load_config, Logger as NL
    from nanodet.model.arch import build_model
    from nanodet.util import load_model_weight
    from nanodet.data.transform import Pipeline
    load_config(cfg, os.path.join(root,'config','nanodet-plus-m_320.yml'))
    model = build_model(cfg.model)
    ckpt  = torch.load(os.path.join(root,'nanodet','nanodet-plus-m_320.pth'),
                       map_location='cpu')
    load_model_weight(model, ckpt, NL(0, use_tensorboard=False))
    model = model.to(DEVICE).eval()
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    print("  [NanoDet-Plus] Loaded.")
    return model, cfg, pipeline

def load_rtdetr():
    from ultralytics import YOLO
    model = YOLO(r'C:\Users\tsang\RT-DETR\rtdetr-l.pt')
    print("  [RT-DETR] Loaded.")
    return model


# ─── DETECTION ───────────────────────────────────────────────
def detect_yolov5(model, adv_pil, conf_thr=None):
    if conf_thr is None: conf_thr = args.conf
    adv_np  = np.array(adv_pil)
    orig_conf = model.conf
    model.conf = conf_thr
    results = model(adv_np, size=MODEL_INPUT)
    model.conf = orig_conf
    boxes   = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().tolist():
        if int(cls) == 0:
            boxes.append([int(xyxy[0]),int(xyxy[1]),
                          int(xyxy[2]),int(xyxy[3]),float(conf)])
    return boxes

def detect_nanodet(model, nano_cfg, pipeline, adv_pil, conf_thr=None):
    if conf_thr is None: conf_thr = args.conf
    import cv2
    from nanodet.data.collate import naive_collate
    from nanodet.data.batch_process import stack_batch_img
    adv_np  = np.array(adv_pil)
    img_bgr = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]
    meta = dict(img_info={'id':0,'file_name':None,'height':h,'width':w},
                raw_img=img_bgr, img=img_bgr)
    meta = pipeline(None, meta, nano_cfg.data.val.input_size)
    meta['img'] = torch.from_numpy(meta['img'].transpose(2,0,1)).to(DEVICE)
    meta = naive_collate([meta])
    meta['img'] = stack_batch_img(meta['img'], divisible=32)
    with torch.no_grad():
        results = model.inference(meta)
    boxes = []
    for d in results[0].get(0, []):
        if d[-1] >= conf_thr:
            boxes.append([int(d[0]),int(d[1]),int(d[2]),int(d[3]),float(d[-1])])
    return boxes

def detect_rtdetr(model, adv_pil, conf_thr=None):
    if conf_thr is None: conf_thr = args.conf
    adv_np  = np.array(adv_pil)
    results = model.predict(source=adv_np, conf=conf_thr,
                            classes=[0], verbose=False)
    boxes = []
    for box in results[0].boxes:
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        boxes.append([x1,y1,x2,y2,float(box.conf[0])])
    return boxes


# ─── DRAW BOXES ──────────────────────────────────────────────
def draw_boxes(pil_img, pred_boxes, gt_boxes):
    img  = pil_img.copy()
    draw = ImageDraw.Draw(img)
    # GT boxes — green
    for (x1,y1,x2,y2) in gt_boxes:
        draw.rectangle([x1,y1,x2,y2], outline='green', width=2)
    # Pred boxes — red
    for (x1,y1,x2,y2,conf) in pred_boxes:
        draw.rectangle([x1,y1,x2,y2], outline='red', width=3)
        draw.text((x1, max(0,y1-15)), f'{conf:.2f}', fill='red')
    return img


# ─── SUMMARY SAVE ────────────────────────────────────────────
def save_summary(game, model_name, dsr, recall, recall_before, avg_ssim, avg_fps, n_images, total_time_s):
    drop = (recall_before - recall) / recall_before if recall_before and recall_before > 0 else None
    row = {
        'Game':           game.upper(),
        'Model':          model_name,
        'Images':         n_images,
        'DSR':            round(dsr,    4),
        'Recall_after':   round(recall, 4),
        'Recall_before':  round(recall_before, 4) if recall_before else None,
        'Recall_drop':    round(drop, 4) if drop is not None else None,
        'AvgSSIM':        round(avg_ssim, 4),
        'AvgFPS':         round(avg_fps,  2),
        'TotalTime_s':    round(total_time_s, 2),
    }
    if os.path.exists(EVAL_XLSX):
        df = pd.read_excel(EVAL_XLSX)
        df = df[~((df['Game']==row['Game'])&(df['Model']==row['Model']))]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(EVAL_XLSX, index=False)
    print(f"  Summary → {EVAL_XLSX}")

    # Chart — DSR and Recall side by side
    if len(df) >= 1:
        labels  = [f"{r['Game']}\n{r['Model']}" for _,r in df.iterrows()]
        dsrs    = [v*100 for v in df['DSR'].tolist()]
        recalls = [v*100 for v in df['Recall_drop'].tolist()]
        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(df)*1.2), 5))
        b1 = ax.bar(x - w/2, dsrs,    w, label='DSR (%)',          color='steelblue')
        b2 = ax.bar(x + w/2, recalls, w, label='Recall Drop (%)',   color='tomato')
        for bar, val in zip(b1, dsrs):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+1, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=8)
        for bar, val in zip(b2, recalls):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+1, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, 120)
        ax.set_title('Universal Cloak — DSR & Recall Drop Rate per Game × Model')
        ax.legend()
        plt.tight_layout()
        plt.savefig(EVAL_PNG, dpi=150); plt.close()
        print(f"  Chart  → {EVAL_PNG}")


# ─── MAIN ────────────────────────────────────────────────────
def main():
    print("\n[1] Loading image paths …")
    img_paths = load_image_paths(DATA_PATH)

    print(f"\n[2] Loading model: {MODEL} …")
    nano_cfg = nano_pipeline = None
    if MODEL == 'yolov5n':
        detect_model = load_yolov5n()
    elif MODEL == 'nanodet':
        detect_model, nano_cfg, nano_pipeline = load_nanodet()
    elif MODEL == 'rtdetr':
        detect_model = load_rtdetr()

    print(f"\n[3] Evaluating {len(img_paths)} images …")
    succ_num   = 0
    total_time = 0.0
    ssim_vals  = []
    log_rows   = []
    total_tp   = 0
    total_fn   = 0

    for i, img_path in enumerate(img_paths):
        img_pil  = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size

        # Load GT labels
        fname     = os.path.splitext(os.path.basename(img_path))[0]
        gt_label  = os.path.join(GT_PATH, fname + '.txt')
        gt_boxes  = load_gt_boxes(gt_label, orig_w, orig_h)

        t_start = time.time()

        # Apply noise
        adv_pil, adv_t, img_t = apply_noise_original(img_pil)

        # Detect — conf=0.4 for DSR
        if MODEL == 'yolov5n':
            pred_boxes = detect_yolov5(detect_model, adv_pil, args.conf)
        elif MODEL == 'nanodet':
            pred_boxes = detect_nanodet(detect_model, nano_cfg, nano_pipeline, adv_pil, args.conf)
        elif MODEL == 'rtdetr':
            pred_boxes = detect_rtdetr(detect_model, adv_pil, args.conf)

        # Detect — conf=0.2 for Recall
        if MODEL == 'yolov5n':
            pred_boxes_recall = detect_yolov5(detect_model, adv_pil, args.conf_recall)
        elif MODEL == 'nanodet':
            pred_boxes_recall = detect_nanodet(detect_model, nano_cfg, nano_pipeline, adv_pil, args.conf_recall)
        elif MODEL == 'rtdetr':
            pred_boxes_recall = detect_rtdetr(detect_model, adv_pil, args.conf_recall)

        t_elapsed  = time.time() - t_start
        total_time += t_elapsed

        # DSR
        target_succ = 1 if len(pred_boxes) == 0 else 0
        succ_num   += target_succ

        # Recall — TP/FN (using conf=0.2)
        tp, fn = compute_tp_fn(gt_boxes, pred_boxes_recall, args.iou_thr)
        total_tp += tp
        total_fn += fn

        # SSIM
        ssim_val = ssim_fn(
            (img_t*255).cpu(), (adv_t*255).cpu(),
            data_range=255, size_average=False
        ).item()
        ssim_vals.append(ssim_val)

        fps      = (i+1) / total_time
        dsr_now  = succ_num / (i+1)
        recall_now = total_tp / (total_tp + total_fn) if (total_tp+total_fn) > 0 else 0.0

        print(f"  [{i+1:3d}/{len(img_paths)}] "
              f"GT:{len(gt_boxes)} Pred:{len(pred_boxes)} "
              f"DSR:{dsr_now:.3f} Recall:{recall_now:.3f} "
              f"SSIM:{ssim_val:.3f} FPS:{fps:.1f}")

        # Save attack image with GT (green) + pred (red) boxes
        # Save with bounding boxes (gt=green, pred=red)
        adv_vis = draw_boxes(adv_pil, pred_boxes, gt_boxes)
        adv_vis.save(os.path.join(ATTACK_DIR, f'{fname}_attack.jpg'))
        # Save clean (no bounding boxes)
        adv_pil.save(os.path.join(ATTACK_CLEAN_DIR, f'{fname}_attack.png'))

        log_rows.append({
            'Index':   i,
            'File':    os.path.basename(img_path),
            'GT_boxes':    len(gt_boxes),
            'Pred_boxes':  len(pred_boxes),
            'TP':      tp,
            'FN':      fn,
            'TSucc':   target_succ,
            'DSR':     round(dsr_now,   4),
            'Recall':  round(recall_now,4),
            'SSIM':    round(ssim_val,  4),
            'Time_s':  round(t_elapsed, 4),
            'FPS':     round(fps,       2),
        })

    # Save per-image log
    log_path = os.path.join(LOG_DIR, f'{GAME}_{MODEL}_eval.xlsx')
    pd.DataFrame(log_rows).to_excel(log_path, index=False)
    print(f"\n  Log → {log_path}")

    final_dsr    = succ_num / len(img_paths)
    final_recall = total_tp / (total_tp + total_fn) if (total_tp+total_fn) > 0 else 0.0
    final_ssim   = np.mean(ssim_vals)
    final_fps    = len(img_paths) / total_time

    total_min = total_time / 60
    print(f"\n{'='*55}")
    print(f"  RESULTS : {GAME.upper()} / {MODEL}")
    print(f"  Total runtime : {total_time:.1f}s ({total_min:.1f} min)")
    print(f"  DSR     : {final_dsr:.4f} ({final_dsr*100:.1f}%)")
    print(f"  Recall after  : {final_recall:.4f} ({final_recall*100:.1f}%)")
    if RECALL_BEFORE:
        drop = (RECALL_BEFORE - final_recall) / RECALL_BEFORE * 100
        print(f"  Recall before : {RECALL_BEFORE:.4f}")
        print(f"  Recall drop   : {drop:.1f}%")
    print(f"  AvgSSIM : {final_ssim:.4f}")
    print(f"  AvgFPS  : {final_fps:.2f}")
    print(f"{'='*55}\n")

    save_summary(GAME, MODEL, final_dsr, final_recall, RECALL_BEFORE,
                 final_ssim, final_fps, len(img_paths), total_time)

if __name__ == '__main__':
    main()