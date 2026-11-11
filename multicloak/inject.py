"""
inject_cloak_multirefine.py — Apply Universal Cloak + Multi-Model Refinement & save as PNG
Separates the expensive refinement step from evaluation so refined images
can be reused across multiple cheating-model evaluations.

Pipeline:
  1. Load universal noise
  2. For each frame:
     a. Apply universal noise → check ALL proxy models
     b. If any model still detects → refine using all models simultaneously
     c. Save the final adversarial image as PNG
  3. Run get_cloak_paper.py on the saved PNGs (no refinement needed)

Usage:
    # Multi-model refinement with sum strategy (default):
    python inject_cloak_multirefine.py --game cs2 --refine_models yolov5n yolov5s yolov5m

    # With max strategy:
    python inject_cloak_multirefine.py --game cs2 --refine_models yolov5n yolov5s yolov5m \
        --refine_strategy max

    # Then evaluate (no refinement — just load saved PNGs):
    python eval_cloaked.py --game cs2 --cheating_model yolov5n yolov5s yolov5m \
        --cloaked_dir C:\\Users\\tsang\\Dataset\\dataset_demo\\cs2\\frames_cloaked_multirefine_sum
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim as ssim_fn
from tqdm import tqdm

# ─── ARGUMENTS ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--game',           required=True, type=str, choices=['cs2','cf'])
parser.add_argument('--refine_models',  nargs='+', required=True,
                    choices=['yolov5n','yolov5s','yolov5m','yolov5x'],
                    help='Proxy models for simultaneous refinement')
parser.add_argument('--refine_strategy', default='sum', type=str,
                    choices=['sum', 'max'],
                    help='sum = average loss across all models, '
                         'max = backprop only through highest-confidence model (default: sum)')
parser.add_argument('--n_iter',    default=5,     type=int,
                    help='Max refinement iterations per frame (default: 5)')
parser.add_argument('--lr',        default=0.005, type=float,
                    help='Refinement learning rate (default: 0.005)')
parser.add_argument('--epsilon',   default=8,     type=int)
parser.add_argument('--conf',      default=0.4,   type=float,
                    help='Confidence threshold θ (default: 0.4)')
parser.add_argument('--gpu',       default='0',   type=str)
parser.add_argument('--noise_dir', default=None,  type=str,
                    help='Override noise directory')
parser.add_argument('--out_dir',   default=None,  type=str,
                    help='Override output directory for cloaked frames')
args = parser.parse_args()

DEVICE   = f'cuda:{args.gpu}'
EPS      = args.epsilon / 255.0
GAME     = args.game
STRATEGY = args.refine_strategy

GAME_FOLDER = {'cs2': 'cs2', 'cf': 'cf'}

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH = os.path.join(r'C:\Users\tsang\Dataset\dataset_demo',
                         GAME_FOLDER[GAME], 'frames')

if args.out_dir:
    OUT_PATH = args.out_dir
else:
    OUT_PATH = os.path.join(r'C:\Users\tsang\Dataset\dataset_demo',
                            GAME_FOLDER[GAME],
                            f'frames_cloaked_multirefine_{STRATEGY}')
os.makedirs(OUT_PATH, exist_ok=True)

if args.noise_dir:
    NOISE_DIR = args.noise_dir
else:
    NOISE_DIR = os.path.join('universal_cloak', GAME,
                             'ensemble_yolov5n_yolov5s_yolov5m')
NOISE_PATH = os.path.join(NOISE_DIR, 'universal_noise.pt')

BCEobj = nn.BCELoss()

print(f"\n{'='*65}")
print(f"  Game            : {GAME.upper()}")
print(f"  Refine models   : {args.refine_models}")
print(f"  Strategy        : {STRATEGY}")
print(f"  Input           : {DATA_PATH}")
print(f"  Output          : {OUT_PATH}")
print(f"  Noise           : {NOISE_PATH}")
print(f"  Refinement      : n_iter={args.n_iter}  lr={args.lr}  eps={args.epsilon}/255")
print(f"  Conf threshold  : {args.conf}")
print(f"{'='*65}\n")


# ─── LOAD UNIVERSAL NOISE ────────────────────────────────────
if not os.path.exists(NOISE_PATH):
    raise FileNotFoundError(f"Noise not found: {NOISE_PATH}")
global_noise = torch.load(NOISE_PATH).to(DEVICE)
print(f"  Universal Cloak loaded: {global_noise.shape}")


# ─── MODEL LOADER ────────────────────────────────────────────
def load_proxy_model(variant):
    """Load YOLOv5 WITHOUT AutoShape — for gradient-based refinement."""
    sys.path.insert(0, os.path.abspath('.'))
    from models.common import DetectMultiBackend
    pt_path = os.path.join('pretrained_models', f'{variant}.pt')
    model   = DetectMultiBackend(pt_path, device=torch.device(DEVICE), fuse=True)
    model.eval()
    print(f"  [Proxy] {variant} loaded")
    return model


# ─── ATTACK SUCCESS CHECK ────────────────────────────────────
def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def attack_success(preds, conf_thres=0.4):
    """Returns number of person detections (0 = success)."""
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = preds.detach()
    xc    = preds[..., 4] > 0.25
    x     = preds[0][xc[0]]
    if x.shape[0] == 0:
        return 0
    x[:, 5:] *= x[:, 4:5]
    box  = xywh2xyxy(x[:, :4])
    conf, j = x[:, 5:85].max(1, keepdim=True)
    x    = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
    x    = x[(x[:, 5:6] == 0).any(1)]
    return x.shape[0]


def check_all_models(models, x_adv, conf_thres=0.4):
    """Check if ALL proxy models are fooled. Returns (all_fooled, per_model_detections)."""
    per_model = []
    for model in models:
        with torch.no_grad():
            preds = model(x_adv)
        n_det = attack_success(preds, conf_thres)
        per_model.append(n_det)
    all_fooled = all(d == 0 for d in per_model)
    return all_fooled, per_model


# ─── LOSS HELPERS ─────────────────────────────────────────────
def compute_bce_loss(preds):
    """BCE loss — suppress objectness → 0."""
    raw = preds[0] if isinstance(preds, (list, tuple)) else preds
    obj = raw[..., 4:5].clamp(1e-6, 1 - 1e-6)
    return BCEobj(obj, torch.zeros_like(obj))


def compute_max_confidence(preds):
    """Get maximum objectness confidence."""
    raw = preds[0] if isinstance(preds, (list, tuple)) else preds
    return raw[..., 4].max().item()


# ─── MULTI-MODEL REFINEMENT PER FRAME ────────────────────────
def refine_frame(x, models, n_iter, eps, lr, strategy, conf_thres):
    """
    Apply universal noise + multi-model refinement to one frame.

    Returns: adv_img (tensor), query_count, success (bool), elapsed_time
    """
    t_start  = time.time()
    query    = 0
    n_models = len(models)
    _, _, H, W = x.shape

    # ── Stage A: Universal noise → check all models ───────────
    noise_resized = F.interpolate(
        global_noise.unsqueeze(0), size=(H, W),
        mode='bilinear', align_corners=False
    ).squeeze(0)                                        # (3, H, W)

    with torch.no_grad():
        eta   = torch.clamp(noise_resized, -eps, eps)
        x_adv = torch.clamp(x + eta.unsqueeze(0), 0.0, 1.0)

        all_fooled, per_model = check_all_models(models, x_adv, conf_thres)
        query += n_models

        if all_fooled:
            return x_adv, query, True, time.time() - t_start

    # ── Stage B: Multi-model refinement ───────────────────────
    eta = noise_resized.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([eta], lr=lr)

    x_input = x.detach().clone()
    all_fooled = False

    for k in range(n_iter):
        # Forward pass through all models
        noise_clamped = torch.clamp(eta, -eps, eps)
        adv_img = torch.clamp(x_input + noise_clamped.unsqueeze(0), 0.0, 1.0)

        all_preds = []
        for model in models:
            preds = model(adv_img)
            all_preds.append(preds)
        query += n_models

        # Check success
        det_counts = [attack_success(p, conf_thres) for p in all_preds]
        if all(d == 0 for d in det_counts):
            all_fooled = True
            break

        # Compute loss
        if strategy == 'sum':
            losses = [compute_bce_loss(p) for p in all_preds]
            loss = sum(losses) / len(losses)
        elif strategy == 'max':
            max_confs = [compute_max_confidence(p) for p in all_preds]
            hardest_idx = int(np.argmax(max_confs))
            loss = compute_bce_loss(all_preds[hardest_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for m in models:
            m.zero_grad()

    # Final adversarial image
    with torch.no_grad():
        noise_final = torch.clamp(eta, -eps, eps)
        x_adv = torch.clamp(x_input + noise_final.unsqueeze(0), 0.0, 1.0)

    return x_adv, query, all_fooled, time.time() - t_start


# ─── MAIN ────────────────────────────────────────────────────
def main():
    # Load proxy models
    print("[1] Loading proxy models …")
    models = [load_proxy_model(v) for v in args.refine_models]

    # List images
    exts  = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted([f for f in os.listdir(DATA_PATH) if f.lower().endswith(exts)])
    N = len(files)
    print(f"\n[2] Found {N} images in {DATA_PATH}\n")

    ssim_vals   = []
    query_vals  = []
    success_cnt = 0
    total_time  = 0.0

    for i, fname in enumerate(tqdm(files, desc='Refining')):
        img_pil = Image.open(os.path.join(DATA_PATH, fname)).convert('RGB')
        orig_w, orig_h = img_pil.size

        img_t = torch.from_numpy(
            np.array(img_pil).transpose(2, 0, 1)
        ).float().unsqueeze(0).to(DEVICE) / 255.0

        # ── Refine ────────────────────────────────────────────
        adv_t, query, success, elapsed = refine_frame(
            img_t, models, args.n_iter, EPS, args.lr,
            STRATEGY, args.conf
        )
        total_time += elapsed
        query_vals.append(query)
        if success:
            success_cnt += 1

        # ── SSIM ──────────────────────────────────────────────
        ssim_val = ssim_fn(
            (img_t * 255).cpu(), (adv_t * 255).detach().cpu(),
            data_range=255, size_average=False
        ).item()
        ssim_vals.append(ssim_val)

        # ── Save PNG ──────────────────────────────────────────
        adv_np   = (adv_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        out_name = os.path.splitext(fname)[0] + '.png'
        Image.fromarray(adv_np).save(os.path.join(OUT_PATH, out_name))

        # Progress
        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == N:
            fps = (i + 1) / total_time if total_time > 0 else 0
            dsr = success_cnt / (i + 1)
            print(f"  [{i+1:4d}/{N}]  Success:{success}  Q:{query}  "
                  f"DSR:{dsr:.3f}  SSIM:{ssim_val:.3f}  FPS:{fps:.1f}")

    # ── Summary ───────────────────────────────────────────────
    avg_ssim  = np.mean(ssim_vals)
    avg_query = np.mean(query_vals)
    final_dsr = success_cnt / N
    final_fps = N / total_time if total_time > 0 else 0

    print(f"\n{'='*65}")
    print(f"  DONE — {N} images saved to {OUT_PATH}")
    print(f"  Strategy   : {STRATEGY}")
    print(f"  DSR (proxy): {final_dsr*100:.2f}%")
    print(f"  Avg Query  : {avg_query:.2f}")
    print(f"  Avg SSIM   : {avg_ssim:.4f}")
    print(f"  FPS        : {final_fps:.2f}")
    print(f"  Total time : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*65}")
    print(f"\n  Next step: evaluate with cheating models:")
    print(f"    python eval_cloaked.py --game {GAME} \\")
    print(f"        --cheating_model yolov5n yolov5s yolov5m \\")
    print(f"        --cloaked_dir \"{OUT_PATH}\"")
    print()


if __name__ == '__main__':
    main()