"""
get_cloak.py — Apply Universal Cloak & Evaluate (v2)
ปรับปรุง:
- apply noise บน original size image
- save ภาพ original size พร้อม bounding box ทุกโมเดล
- รองรับ yolov5n, nanodet, rtdetr และ 4 เกม

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
parser.add_argument('--conf',    default=0.4,  type=float)
parser.add_argument('--gpu',     default='0',  type=str)
args = parser.parse_args()

DEVICE = f'cuda:{args.gpu}'
EPS    = args.epsilon / 255.0
GAME   = args.game
MODEL  = args.model

MODEL_SIZE  = {'yolov5n':416, 'nanodet':320, 'rtdetr':640}
MODEL_INPUT = MODEL_SIZE[MODEL]

GAME_FOLDER = {'cs2':'CS2','cf':'CF','valorant':'Valorant','overwatch':'Overwatch'}
DATA_PATH   = os.path.join(r'C:\Users\tsang\Dataset', GAME_FOLDER[GAME])
NOISE_PATH  = os.path.join('universal_cloak', GAME, MODEL, 'universal_noise.pt')

ATTACK_DIR  = os.path.join('result','evaluation', GAME, MODEL, 'attack')
LOG_DIR     = os.path.join('result','log', GAME)
os.makedirs(ATTACK_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

EVAL_XLSX = os.path.join('result','evaluation_summary.xlsx')
EVAL_PNG  = os.path.join('result','evaluation_summary.png')

print(f"\n{'='*55}")
print(f"  Game  : {GAME}  |  Model : {MODEL}")
print(f"  Data  : {DATA_PATH}")
print(f"  Noise : {NOISE_PATH}")
print(f"{'='*55}\n")

# Load noise (model input size)
if not os.path.exists(NOISE_PATH):
    raise FileNotFoundError(f"Noise not found: {NOISE_PATH}\nRun train_cloak.py first.")
noise_model_size = torch.load(NOISE_PATH).to(DEVICE)   # (3, S, S)
print(f"  Noise loaded: {noise_model_size.shape}")


# ─── HELPERS ─────────────────────────────────────────────────
def load_image_paths(folder):
    exts  = ('.jpg','.jpeg','.png','.bmp')
    paths = sorted([os.path.join(folder,f) for f in os.listdir(folder)
                    if f.lower().endswith(exts)])
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    print(f"  Found {len(paths)} images")
    return paths

def apply_noise_original(img_pil):
    """
    img_pil: PIL RGB original size
    1. resize to model input → apply noise → resize noise back → add to original
    Returns: adv PIL image (original size), adv tensor (1,3,H,W)
    """
    orig_w, orig_h = img_pil.size
    img_t = torch.from_numpy(
        np.array(img_pil).transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.0

    # Resize noise from model input size to original size
    noise_orig = F.interpolate(
        noise_model_size.unsqueeze(0),
        size=(orig_h, orig_w),
        mode='bilinear', align_corners=False
    ).squeeze(0)                                    # (3,H,W)

    # Clamp and apply
    noise_clamped = torch.clamp(noise_orig, -EPS, EPS)
    adv_t = torch.clamp(img_t + noise_clamped.unsqueeze(0), 0.0, 1.0)
    adv_np = (adv_t[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    return adv_pil, adv_t, img_t


# ─── MODEL LOADERS ───────────────────────────────────────────
def load_yolov5n():
    sys.path.insert(0, os.path.abspath('.'))
    from models.common import DetectMultiBackend, AutoShape
    weights = os.path.join('pretrained_models','yolov5n.pt')
    model   = DetectMultiBackend(weights, device=torch.device(DEVICE), fuse=True)
    model   = AutoShape(model)
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


# ─── DETECTION (returns boxes list [(x1,y1,x2,y2,conf)]) ─────
def detect_yolov5(model, adv_pil):
    adv_np  = np.array(adv_pil)
    results = model(adv_np, size=MODEL_INPUT)
    boxes   = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().tolist():
        if int(cls) == 0:
            boxes.append((int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3]),conf))
    return boxes

def detect_nanodet(model, nano_cfg, pipeline, adv_pil):
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
    dets  = results[0].get(0, [])
    for d in dets:
        if d[-1] >= args.conf:
            x1,y1,x2,y2 = int(d[0]),int(d[1]),int(d[2]),int(d[3])
            boxes.append((x1,y1,x2,y2,float(d[-1])))
    return boxes

def detect_rtdetr(model, adv_pil):
    adv_np  = np.array(adv_pil)
    results = model.predict(source=adv_np, conf=args.conf,
                            classes=[0], verbose=False)
    boxes = []
    for box in results[0].boxes:
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        boxes.append((x1,y1,x2,y2,float(box.conf[0])))
    return boxes


# ─── DRAW BOUNDING BOXES ─────────────────────────────────────
def draw_boxes(pil_img, boxes):
    img   = pil_img.copy()
    draw  = ImageDraw.Draw(img)
    for (x1,y1,x2,y2,conf) in boxes:
        draw.rectangle([x1,y1,x2,y2], outline='red', width=3)
        draw.text((x1,max(0,y1-15)), f'{conf:.2f}', fill='red')
    return img


# ─── SUMMARY SAVE ────────────────────────────────────────────
def save_summary(game, model_name, dsr, avg_ssim, avg_fps, n_images):
    row = {'Game':game.upper(),'Model':model_name,'Images':n_images,
           'DSR':round(dsr,4),'AvgSSIM':round(avg_ssim,4),'AvgFPS':round(avg_fps,2)}
    if os.path.exists(EVAL_XLSX):
        df = pd.read_excel(EVAL_XLSX)
        df = df[~((df['Game']==row['Game'])&(df['Model']==row['Model']))]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(EVAL_XLSX, index=False)
    print(f"  Summary → {EVAL_XLSX}")
    if len(df) >= 1:
        labels = [f"{r['Game']}\n{r['Model']}" for _,r in df.iterrows()]
        dsrs   = df['DSR'].tolist()
        fig,ax = plt.subplots(figsize=(max(8,len(df)*1.0),5))
        bars   = ax.bar(labels,[d*100 for d in dsrs],color='steelblue',width=0.55)
        for bar,val in zip(bars,dsrs):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+1,
                    f'{val*100:.1f}%',ha='center',va='bottom',fontsize=9)
        ax.set_ylabel('Defense Success Rate (%)')
        ax.set_ylim(0,115)
        ax.set_title('Universal Cloak — DSR per Game × Model')
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

    for i, img_path in enumerate(img_paths):
        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size

        t_start = time.time()

        # Apply noise at original size
        adv_pil, adv_t, img_t = apply_noise_original(img_pil)

        # Detect on adversarial image
        if MODEL == 'yolov5n':
            boxes = detect_yolov5(detect_model, adv_pil)
        elif MODEL == 'nanodet':
            boxes = detect_nanodet(detect_model, nano_cfg, nano_pipeline, adv_pil)
        elif MODEL == 'rtdetr':
            boxes = detect_rtdetr(detect_model, adv_pil)

        t_elapsed  = time.time() - t_start
        total_time += t_elapsed

        target_succ = 1 if len(boxes) == 0 else 0
        succ_num   += target_succ

        # SSIM (compare original vs adversarial at original size)
        ssim_val = ssim_fn(
            (img_t*255).cpu(), (adv_t*255).cpu(),
            data_range=255, size_average=False
        ).item()
        ssim_vals.append(ssim_val)

        fps     = (i+1) / total_time
        dsr_now = succ_num / (i+1)

        print(f"  [{i+1:3d}/{len(img_paths)}] "
              f"Dets:{len(boxes)} TSucc:{target_succ} "
              f"DSR:{dsr_now:.3f} SSIM:{ssim_val:.3f} "
              f"FPS:{fps:.1f}")

        # Save attack image with bounding boxes (original size)
        adv_with_boxes = draw_boxes(adv_pil, boxes)
        fname = os.path.splitext(os.path.basename(img_path))[0]
        adv_with_boxes.save(os.path.join(ATTACK_DIR, f'{fname}_attack.jpg'))

        log_rows.append({
            'Index':   i,
            'File':    os.path.basename(img_path),
            'TSucc':   target_succ,
            'Dets':    len(boxes),
            'DSR':     round(dsr_now,4),
            'SSIM':    round(ssim_val,4),
            'Time_s':  round(t_elapsed,4),
            'FPS':     round(fps,2),
        })

    # Save per-image log
    log_path = os.path.join(LOG_DIR, f'{GAME}_{MODEL}_eval.xlsx')
    pd.DataFrame(log_rows).to_excel(log_path, index=False)
    print(f"\n  Log → {log_path}")

    final_dsr  = succ_num / len(img_paths)
    final_ssim = np.mean(ssim_vals)
    final_fps  = len(img_paths) / total_time

    print(f"\n{'='*55}")
    print(f"  RESULTS : {GAME.upper()} / {MODEL}")
    print(f"  DSR     : {final_dsr:.4f} ({final_dsr*100:.1f}%)")
    print(f"  AvgSSIM : {final_ssim:.4f}")
    print(f"  AvgFPS  : {final_fps:.2f}")
    print(f"{'='*55}\n")

    save_summary(GAME, MODEL, final_dsr, final_ssim, final_fps, len(img_paths))

if __name__ == '__main__':
    main()