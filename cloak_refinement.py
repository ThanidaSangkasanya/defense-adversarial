# Cloak Refinement for Model Variability (4.2)
# Pipeline matches get_cloak.py: noise applied at ORIGINAL image size
# AutoShape handles resize during detection

import os, sys, time, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pytorch_msssim import ssim as ssim_fn
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger

# ─── ARGUMENTS ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="4.2 Cloak Refinement")
parser.add_argument('--game',       required=True, type=str,
                    choices=['cs2','cf','valorant','overwatch'])
parser.add_argument('--model',      required=True, type=str,
                    choices=['yolov5n','nanodet','rtdetr'])
parser.add_argument('--target_model', default=None, type=str,
                    choices=['yolov5n','nanodet','rtdetr'])
parser.add_argument('--dataset_dir', required=True, type=str)
parser.add_argument('--universal_cloak_path', required=True, type=str)
parser.add_argument('--max_query',  default=5,     type=int)
parser.add_argument('--theta',      default=0.4,   type=float)
parser.add_argument('--epsilon',    default=8,     type=int)
parser.add_argument('--lr',         default=0.005, type=float)
parser.add_argument('--gpu',        default='0',   type=str)
parser.add_argument('--label_dir',  default=None,  type=str)
parser.add_argument('--save_dir',   default='./result/refinement', type=str)
parser.add_argument('--save_images', action='store_true')
args = parser.parse_args()

DEVICE = f'cuda:{args.gpu}'
EPS    = args.epsilon / 255.0

MODEL_SIZE  = {'yolov5n': 320, 'nanodet': 320, 'rtdetr': 640}
MODEL_INPUT = MODEL_SIZE[args.model]

# Auto-organize: result/refinement/cs2/yolov5n/
args.save_dir = os.path.join(args.save_dir, args.game, args.model)

_bce = nn.BCELoss()


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ═══════════════════════════════════════════════════════════════

def load_raw_model(model_name):
    """Raw model for gradient (NO AutoShape)"""
    if model_name == 'yolov5n':
        sys.path.insert(0, os.path.abspath('.'))
        from models.common import DetectMultiBackend
        model = DetectMultiBackend(
            os.path.join('pretrained_models', 'yolov5n.pt'),
            device=torch.device(DEVICE), fuse=True)
        model.eval()
        logger.info("[YOLOv5n] Raw model loaded (for gradient)")
        return model
    elif model_name == 'nanodet':
        root = r'C:\Users\tsang\nanodet'
        sys.path.insert(0, root)
        from nanodet.util import cfg, load_config, Logger as NL
        from nanodet.model.arch import build_model
        from nanodet.util import load_model_weight
        load_config(cfg, os.path.join(root, 'config', 'nanodet-plus-m_320.yml'))
        model = build_model(cfg.model)
        ckpt  = torch.load(os.path.join(root, 'nanodet', 'nanodet-plus-m_320.pth'),
                           map_location='cpu')
        load_model_weight(model, ckpt, NL(0, use_tensorboard=False))
        model = model.to(DEVICE).eval()
        logger.info("[NanoDet] Raw model loaded")
        return model
    elif model_name == 'rtdetr':
        from ultralytics import YOLO
        yolo  = YOLO(r'C:\Users\tsang\RT-DETR\rtdetr-l.pt')
        model = yolo.model.to(DEVICE).eval()
        logger.info("[RT-DETR] Raw model loaded")
        return model


def load_detect_model(model_name, conf_thr=0.4):
    """Detection model with NMS (same as get_cloak.py)"""
    if model_name == 'yolov5n':
        sys.path.insert(0, os.path.abspath('.'))
        from models.common import DetectMultiBackend, AutoShape
        model = DetectMultiBackend(
            os.path.join('pretrained_models', 'yolov5n.pt'),
            device=torch.device(DEVICE), fuse=True)
        model = AutoShape(model)
        model.conf    = conf_thr
        model.classes = [0]
        logger.info(f"[YOLOv5n] AutoShape loaded (conf={conf_thr})")
        return model
    elif model_name == 'nanodet':
        root = r'C:\Users\tsang\nanodet'
        sys.path.insert(0, root)
        from nanodet.util import cfg, load_config, Logger as NL
        from nanodet.model.arch import build_model
        from nanodet.util import load_model_weight
        from nanodet.data.transform import Pipeline
        load_config(cfg, os.path.join(root, 'config', 'nanodet-plus-m_320.yml'))
        model = build_model(cfg.model)
        ckpt  = torch.load(os.path.join(root, 'nanodet', 'nanodet-plus-m_320.pth'),
                           map_location='cpu')
        load_model_weight(model, ckpt, NL(0, use_tensorboard=False))
        model = model.to(DEVICE).eval()
        pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.input_size)
        logger.info(f"[NanoDet] Detect model loaded (conf={conf_thr})")
        return (model, cfg, pipeline)
    elif model_name == 'rtdetr':
        from ultralytics import YOLO
        model = YOLO(r'C:\Users\tsang\RT-DETR\rtdetr-l.pt')
        logger.info(f"[RT-DETR] Detect model loaded (conf={conf_thr})")
        return model


# ═══════════════════════════════════════════════════════════════
#  DETECTION with NMS (same as get_cloak.py)
# ═══════════════════════════════════════════════════════════════

def detect_boxes(detect_model, model_type, adv_pil, conf_thr=0.4):
    adv_np = np.array(adv_pil)
    if model_type == 'yolov5n':
        orig_conf = detect_model.conf
        detect_model.conf = conf_thr
        results = detect_model(adv_np, size=MODEL_INPUT)
        detect_model.conf = orig_conf
        boxes = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().tolist():
            if int(cls) == 0:
                boxes.append([int(xyxy[0]),int(xyxy[1]),
                              int(xyxy[2]),int(xyxy[3]),float(conf)])
        return boxes
    elif model_type == 'nanodet':
        import cv2
        from nanodet.data.collate import naive_collate
        from nanodet.data.batch_process import stack_batch_img
        model, cfg, pipeline = detect_model
        img_bgr = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        meta = dict(img_info={'id':0,'file_name':None,'height':h,'width':w},
                    raw_img=img_bgr, img=img_bgr)
        meta = pipeline(None, meta, cfg.data.val.input_size)
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
    elif model_type == 'rtdetr':
        results = detect_model.predict(source=adv_np, conf=conf_thr,
                                       classes=[0], verbose=False)
        boxes = []
        for box in results[0].boxes:
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
            boxes.append([x1,y1,x2,y2,float(box.conf[0])])
        return boxes


def check_dsr_with_nms(detect_model, model_type, adv_pil, conf_thr=0.4):
    boxes = detect_boxes(detect_model, model_type, adv_pil, conf_thr)
    return len(boxes) == 0, boxes


# ═══════════════════════════════════════════════════════════════
#  LOSS / GRADIENT (for refinement at model_input_size)
# ═══════════════════════════════════════════════════════════════

def get_loss_and_max_conf_yolov5(model, adv):
    preds = model(adv)
    raw   = preds[0] if isinstance(preds, (list, tuple)) else preds
    conf_for_loss = torch.sigmoid(raw[..., 4:5])
    loss = _bce(conf_for_loss, torch.zeros_like(conf_for_loss))
    max_conf = raw[..., 4].max().item()
    return loss, max_conf

def get_loss_and_max_conf_nanodet(model, adv):
    mean = torch.tensor([103.53, 116.28, 123.675], device=DEVICE).view(1,3,1,1) / 255.0
    std  = torch.tensor([57.375, 57.12,  58.395],  device=DEVICE).view(1,3,1,1) / 255.0
    img_norm = (adv - mean) / std
    feats = model.backbone(img_norm)
    feats = model.fpn(feats)
    preds = model.head(feats)
    total = torch.tensor(0.0, device=DEVICE)
    max_conf = 0.0
    for p in preds:
        if isinstance(p, (list, tuple)):
            p = p[0]
        prob = torch.sigmoid(p[..., 0:1])
        total = total + _bce(prob, torch.zeros_like(prob))
        max_conf = max(max_conf, prob.max().item())
    return total, max_conf

def get_loss_and_max_conf_rtdetr(model, adv):
    out = model(adv)
    if isinstance(out, dict) and 'pred_logits' in out:
        logits = out['pred_logits']
        person = torch.sigmoid(logits[:, :, 0:1])
        return _bce(person, torch.zeros_like(person)), person.max().item()
    raw  = out[0] if isinstance(out, (list, tuple)) else out
    conf = torch.sigmoid(raw[..., 4:5])
    return _bce(conf, torch.zeros_like(conf)), conf.max().item()

def get_loss_and_max_conf(model, adv, model_type):
    if model_type == 'yolov5n':
        return get_loss_and_max_conf_yolov5(model, adv)
    elif model_type == 'nanodet':
        return get_loss_and_max_conf_nanodet(model, adv)
    elif model_type == 'rtdetr':
        return get_loss_and_max_conf_rtdetr(model, adv)


# ═══════════════════════════════════════════════════════════════
#  IMAGE LOADING — ORIGINAL SIZE (same as get_cloak.py)
# ═══════════════════════════════════════════════════════════════

def load_image_paths(folder):
    exts  = ('.jpg', '.jpeg', '.png', '.bmp')
    paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(exts)])
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    return paths


# ═══════════════════════════════════════════════════════════════
#  NOISE APPLICATION — same as get_cloak.py
#  noise (model_size) → resize to original → apply → detect
# ═══════════════════════════════════════════════════════════════

def apply_noise_at_original_size(img_pil, noise_model_size, epsilon):
    """
    Same pipeline as get_cloak.py:
    1) noise at model_input_size → resize to original size (interpolation 1)
    2) apply to original image
    3) AutoShape will resize during detection (interpolation 2)
    """
    orig_w, orig_h = img_pil.size
    img_t = torch.from_numpy(
        np.array(img_pil).transpose(2, 0, 1)
    ).float().unsqueeze(0).to(DEVICE) / 255.0

    # Resize noise to original image size
    noise_orig = F.interpolate(
        noise_model_size.unsqueeze(0),
        size=(orig_h, orig_w),
        mode='bilinear', align_corners=False
    ).squeeze(0)
    noise_clamped = torch.clamp(noise_orig, -epsilon, epsilon)

    adv_t  = torch.clamp(img_t + noise_clamped.unsqueeze(0), 0.0, 1.0)
    adv_np = (adv_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    return adv_pil, adv_t, img_t


# ═══════════════════════════════════════════════════════════════
#  GT LABELS & DRAWING
# ═══════════════════════════════════════════════════════════════

def load_gt_boxes(label_path, img_w, img_h):
    if not label_path or not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if int(parts[0]) != 0:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), \
                           float(parts[3]), float(parts[4])
            boxes.append([
                (cx - w/2) * img_w, (cy - h/2) * img_h,
                (cx + w/2) * img_w, (cy + h/2) * img_h
            ])
    return boxes


def draw_boxes(pil_img, pred_boxes, gt_boxes):
    img  = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2) in gt_boxes:
        draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
    for box in pred_boxes:
        x1, y1, x2, y2, conf = box[0], box[1], box[2], box[3], box[4]
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, max(0, y1-15)), f'{conf:.2f}', fill='red')
    return img


# ═══════════════════════════════════════════════════════════════
#  CORE: 4.2 Cloak Refinement (per-frame)
#
#  Gradient refinement: at model_input_size (416)
#  DSR check:           at ORIGINAL size via AutoShape (same as get_cloak.py)
# ═══════════════════════════════════════════════════════════════

def refine_cloak_single_frame(
    raw_model,
    detect_model,
    model_type: str,
    img_pil: Image.Image,          # original size PIL
    delta_u: torch.Tensor,         # (3, model_input, model_input)
    epsilon: float,
    theta: float   = 0.4,
    max_query: int = 5,
    alpha_i: float = 0.005,
    beta1: float   = 0.9,
    beta2: float   = 0.999,
    eps_adam: float = 1e-8,
):
    """
    4.2 per-frame refinement.
    - Gradient optimization at model_input_size (416)
    - DSR check at original image size via AutoShape (same as get_cloak.py)
    """
    S = MODEL_INPUT

    # ── Step 1: Universal Cloak Efficacy Assessment ──
    # Apply noise at ORIGINAL size (same pipeline as get_cloak.py)
    adv_pil, _, _ = apply_noise_at_original_size(img_pil, delta_u, epsilon)
    success, boxes = check_dsr_with_nms(detect_model, model_type, adv_pil, theta)

    if success:
        max_conf = max((b[4] for b in boxes), default=0.0)
        return delta_u.clone(), 1, True, max_conf

    max_conf_step1 = max((b[4] for b in boxes), default=0.0)

    # ── Step 2: Conditional Cloak Refinement ──
    # Work at model_input_size for gradient
    x_resized = F.interpolate(
        torch.from_numpy(np.array(img_pil).transpose(2,0,1)).float().unsqueeze(0) / 255.0,
        size=(S, S), mode='bilinear', align_corners=False
    ).to(DEVICE)  # (1, 3, S, S)

    delta = delta_u.clone().detach().requires_grad_(True)

    m = torch.zeros_like(delta)
    v = torch.zeros_like(delta)
    n_queries = 1

    for t in range(1, max_query + 1):
        # Gradient step at model_input_size
        adv = torch.clamp(x_resized + delta.unsqueeze(0), 0.0, 1.0)
        loss, raw_max = get_loss_and_max_conf(raw_model, adv, model_type)
        n_queries += 1

        # DSR check at ORIGINAL size (same as get_cloak.py)
        adv_pil_check, _, _ = apply_noise_at_original_size(img_pil, delta.detach(), epsilon)
        nms_success, nms_boxes = check_dsr_with_nms(detect_model, model_type, adv_pil_check, theta)

        if nms_success:
            return delta.detach(), n_queries, True, raw_max

        # Backward
        loss.backward()
        grad = delta.grad.clone()

        # Adam moment updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        mu_t = m / (1 - beta1 ** t)
        s_t  = v / (1 - beta2 ** t)

        adaptive_step = alpha_i * mu_t / (s_t.sqrt() + eps_adam)

        with torch.no_grad():
            delta_new = delta - adaptive_step
            delta_new = torch.clamp(delta_new, -epsilon, epsilon)
            delta.copy_(delta_new)

        delta.grad.zero_()

    # Final NMS check at original size
    adv_pil_final, _, _ = apply_noise_at_original_size(img_pil, delta.detach(), epsilon)
    success_final, final_boxes = check_dsr_with_nms(detect_model, model_type, adv_pil_final, theta)
    n_queries += 1

    final_max = max((b[4] for b in final_boxes), default=0.0)
    return delta.detach(), n_queries, success_final, final_max


# ═══════════════════════════════════════════════════════════════
#  EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════

def evaluate_refinement():
    logger.info(f"Game: {args.game} | Proxy: {args.model} | "
                f"Target: {args.target_model or args.model}")
    logger.info(f"Input size: {MODEL_INPUT} | Max query: {args.max_query} | "
                f"Theta: {args.theta} | Epsilon: {args.epsilon}/255 | LR: {args.lr}")
    logger.info(f"Pipeline: noise at original size (same as get_cloak.py)")

    # ── Load models ──
    logger.info("Loading raw model (for gradient)...")
    raw_model = load_raw_model(args.model)

    logger.info("Loading detect model (for DSR with NMS)...")
    detect_model = load_detect_model(args.model, args.theta)

    target_detect = None
    if args.target_model:
        if args.target_model == args.model:
            target_detect = detect_model
        else:
            logger.info(f"Loading target detect model: {args.target_model}")
            target_detect = load_detect_model(args.target_model, args.theta)

    # ── Load Universal Cloak ──
    logger.info(f"Loading Universal Cloak: {args.universal_cloak_path}")
    delta_u = torch.load(args.universal_cloak_path, map_location=DEVICE)
    logger.info(f"  Shape: {delta_u.shape}  Range: [{delta_u.min():.4f}, {delta_u.max():.4f}]")

    if delta_u.dim() == 4:
        delta_u = delta_u.squeeze(0)
    delta_u = delta_u.to(DEVICE)

    if delta_u.shape[-1] != MODEL_INPUT:
        logger.info(f"  Resizing cloak {delta_u.shape[-1]} -> {MODEL_INPUT}")
        delta_u = F.interpolate(
            delta_u.unsqueeze(0),
            size=(MODEL_INPUT, MODEL_INPUT),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        delta_u = torch.clamp(delta_u, -EPS, EPS)

    # ── Load dataset (original size paths) ──
    logger.info(f"Loading dataset: {args.dataset_dir}")
    img_paths = load_image_paths(args.dataset_dir)
    n_frames = len(img_paths)
    logger.info(f"  {n_frames} frames (original size)")

    # ── Output dirs ──
    os.makedirs(args.save_dir, exist_ok=True)
    ATTACK_DIR       = os.path.join(args.save_dir, 'attack')
    ATTACK_CLEAN_DIR = os.path.join(args.save_dir, 'attack_clean')
    if args.save_images:
        os.makedirs(ATTACK_DIR, exist_ok=True)
        os.makedirs(ATTACK_CLEAN_DIR, exist_ok=True)
        logger.info(f"  Save attack     -> {ATTACK_DIR}")
        logger.info(f"  Save clean      -> {ATTACK_CLEAN_DIR}")

    # ── Metrics ──
    total_success_proxy  = 0
    total_success_target = 0
    total_queries        = 0
    total_time           = 0.0
    ssim_vals            = []
    log_rows             = []

    # ── Result paths ──
    RESULT_DIR = 'result'
    LOG_DIR    = os.path.join(RESULT_DIR, 'log', args.game)
    os.makedirs(LOG_DIR, exist_ok=True)
    EVAL_XLSX  = os.path.join(RESULT_DIR, 'refinement_summary.xlsx')
    EVAL_PNG   = os.path.join(RESULT_DIR, 'refinement_summary.png')

    logger.info("=" * 60)
    logger.info("4.2 Cloak Refinement - Evaluation Begin")
    logger.info("=" * 60)

    for idx in range(n_frames):
        img_pil = Image.open(img_paths[idx]).convert('RGB')
        orig_w, orig_h = img_pil.size

        # ── Refine ──
        t0 = time.time()
        delta_refined, n_q, success_proxy, max_conf = refine_cloak_single_frame(
            raw_model    = raw_model,
            detect_model = detect_model,
            model_type   = args.model,
            img_pil      = img_pil,
            delta_u      = delta_u,
            epsilon      = EPS,
            theta        = args.theta,
            max_query    = args.max_query,
            alpha_i      = args.lr,
        )
        elapsed = time.time() - t0

        # ── Build final cloaked image at original size (same as get_cloak.py) ──
        adv_pil, adv_t, img_t = apply_noise_at_original_size(
            img_pil, delta_refined, EPS)

        # ── Target model transfer eval ──
        success_target = success_proxy
        if target_detect is not None and target_detect is not detect_model:
            success_target, _ = check_dsr_with_nms(
                target_detect, args.target_model, adv_pil, args.theta)
        elif target_detect is detect_model:
            success_target = success_proxy

        # ── SSIM at original size ──
        sv = ssim_fn(
            (img_t * 255).cpu(),
            (adv_t * 255).detach().cpu(),
            data_range=255, size_average=True
        ).item()
        ssim_vals.append(sv)

        # ── Accumulate ──
        total_success_proxy  += int(success_proxy)
        total_success_target += int(success_target)
        total_queries        += n_q
        total_time           += elapsed

        logger.info(
            f"Frame:{idx:>4d} | "
            f"ProxySucc:{int(success_proxy)} TargetSucc:{int(success_target)} | "
            f"Query:{n_q} MaxConf:{max_conf:.4f} | "
            f"Time:{elapsed:.4f}s SSIM:{sv:.3f}"
        )

        # ── Per-frame log ──
        log_rows.append({
            'Index':       idx,
            'File':        os.path.basename(img_paths[idx]),
            'ProxySucc':   int(success_proxy),
            'TargetSucc':  int(success_target),
            'Query':       n_q,
            'MaxConf':     round(max_conf, 4),
            'SSIM':        round(sv, 4),
            'Time_s':      round(elapsed, 4),
        })

        if (idx + 1) % 50 == 0 or idx == n_frames - 1:
            dsr_p = total_success_proxy / (idx + 1) * 100
            dsr_t = total_success_target / (idx + 1) * 100
            avg_q = total_queries / (idx + 1)
            fps   = (idx + 1) / total_time if total_time > 0 else 0
            avg_s = np.mean(ssim_vals)
            logger.info(
                f"  [Running] DSR_proxy:{dsr_p:.1f}% DSR_target:{dsr_t:.1f}% | "
                f"AvgQuery:{avg_q:.2f} FPS:{fps:.1f} SSIM:{avg_s:.3f}"
            )

        # ── Save images ──
        if args.save_images:
            fname = os.path.splitext(os.path.basename(img_paths[idx]))[0]

            # GT labels at original size
            gt_boxes = []
            if args.label_dir:
                gt_label = os.path.join(args.label_dir, fname + '.txt')
                gt_boxes = load_gt_boxes(gt_label, orig_w, orig_h)

            # Detection boxes at original size
            vis_boxes = detect_boxes(detect_model, args.model, adv_pil, args.theta)

            # 1) attack/ — with GT (green) + pred (red) boxes
            adv_vis = draw_boxes(adv_pil, vis_boxes, gt_boxes)
            adv_vis.save(os.path.join(ATTACK_DIR, f'{fname}_attack.jpg'))

            # 2) attack_clean/ — cloaked, no boxes
            adv_pil.save(os.path.join(ATTACK_CLEAN_DIR, f'{fname}_attack.png'))

    # ══════════════════════════════════════════════════════════
    #  SAVE RESULTS
    # ══════════════════════════════════════════════════════════

    final_dsr_p = total_success_proxy / n_frames
    final_dsr_t = total_success_target / n_frames
    final_ssim  = np.mean(ssim_vals)

    # ── 1) Per-frame log (Excel) ──
    log_path = os.path.join(LOG_DIR, f'{args.game}_{args.model}_refinement.xlsx')
    pd.DataFrame(log_rows).to_excel(log_path, index=False)
    logger.info(f"  Per-frame log -> {log_path}")

    # ── 2) Summary (Excel) — append row per game x model ──
    final_avg_q = total_queries / n_frames
    final_fps   = n_frames / total_time if total_time > 0 else 0
    row = {
        'Game':      args.game.upper(),
        'Model':     args.model,
        'Images':    n_frames,
        'DSR':       round(final_dsr_p, 4),
        'AvgQuery':  round(final_avg_q, 2),
        'FPS':       round(final_fps, 2),
        'AvgSSIM':   round(final_ssim, 4),
    }
    if os.path.exists(EVAL_XLSX):
        df = pd.read_excel(EVAL_XLSX)
        df = df[~((df['Game'] == row['Game']) & (df['Model'] == row['Model']))]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(EVAL_XLSX, index=False)
    logger.info(f"  Summary       -> {EVAL_XLSX}")

    # ── 3) Chart — DSR bar per game x model ──
    if len(df) >= 1:
        labels = [f"{r['Game']}\n{r['Model']}" for _, r in df.iterrows()]
        dsrs   = [v * 100 for v in df['DSR'].tolist()]
        ssims  = [v * 100 for v in df['AvgSSIM'].tolist()]

        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 5))
        b1 = ax.bar(x - w/2, dsrs,  w, label='DSR (%)',  color='steelblue')
        b2 = ax.bar(x + w/2, ssims, w, label='SSIM (%)', color='mediumseagreen')

        for bar, val in zip(b1, dsrs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=9)
        for bar, val in zip(b2, ssims):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, 115)
        ax.set_title('Cloak Refinement (4.2) - DSR & SSIM per Game x Model')
        ax.legend()
        plt.tight_layout()
        plt.savefig(EVAL_PNG, dpi=150)
        plt.close()
        logger.info(f"  Chart         -> {EVAL_PNG}")

    # ── Final log ──
    logger.info("=" * 60)
    logger.info("Final Results")
    logger.info("=" * 60)
    logger.info(f"Total frames:       {n_frames}")
    logger.info(f"DSR (proxy):        {final_dsr_p * 100:.2f}%")
    logger.info(f"DSR (target):       {final_dsr_t * 100:.2f}%")
    logger.info(f"Avg Query:          {total_queries / n_frames:.2f}")
    logger.info(f"Avg SSIM:           {final_ssim:.4f}")


if __name__ == '__main__':
    evaluate_refinement()