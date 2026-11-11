"""
train_universal_cloak_ensemble.py — Train Universal Noise jointly on multiple YOLOv5 models
One noise pattern optimized as weighted average across all models in the ensemble.

Usage:
    # Default: equal-weight ensemble of yolov5n/s/m
    python train_universal_cloak_ensemble.py --game cs2

    # Custom weights (e.g. give more weight to larger model):
    python train_universal_cloak_ensemble.py --game cs2 --weights 1.0 1.0 1.5

    # Custom model list:
    python train_universal_cloak_ensemble.py --game cs2 --models yolov5n yolov5s yolov5m yolov5x
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim as ssim_fn
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── ARGUMENTS ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--game',      required=True,  type=str,
                    choices=['cs2','cf','valorant','overwatch'])
parser.add_argument('--models',    nargs='+',
                    default=['yolov5n','yolov5s','yolov5m'],
                    choices=['yolov5n','yolov5s','yolov5m','yolov5x'],
                    help='List of YOLOv5 models to ensemble (e.g. --models yolov5n yolov5s yolov5m)')
parser.add_argument('--weights',   nargs='+', type=float, default=None,
                    help='Per-model loss weights (same order as --models). '
                         'Default: equal weights. e.g. --weights 1.0 1.0 1.5')
parser.add_argument('--n_iter',    default=100,    type=int)
parser.add_argument('--lr',        default=0.0005, type=float)
parser.add_argument('--epsilon',   default=8,      type=int)
parser.add_argument('--gpu',       default='0',    type=str)
parser.add_argument('--batch_size',default=8,      type=int)
parser.add_argument('--ssim_w',    default=0.3,    type=float,
                    help='Weight of SSIM loss (default: 0.3)')
args = parser.parse_args()

# ── validate weights ──────────────────────────────────────────
MODEL_LIST = args.models
if args.weights is not None:
    assert len(args.weights) == len(MODEL_LIST), \
        f"--weights must have same length as --models ({len(MODEL_LIST)} models)"
    MODEL_WEIGHTS = [w / sum(args.weights) for w in args.weights]   # normalise → sum=1
else:
    MODEL_WEIGHTS = [1.0 / len(MODEL_LIST)] * len(MODEL_LIST)       # equal weight

DEVICE = f'cuda:{args.gpu}'
EPS    = args.epsilon / 255.0
GAME   = args.game

GAME_FOLDER = {'cs2':'CS2','cf':'CF','valorant':'Valorant','overwatch':'Overwatch'}
DATA_PATH   = os.path.join(r'C:\Users\tsang\Dataset', GAME_FOLDER[GAME])

# All YOLOv5 variants use 416 input size (adjust if needed)
MODEL_INPUT = 416

# ── output dirs ───────────────────────────────────────────────
ensemble_tag    = '_'.join(MODEL_LIST)          # e.g. "yolov5n_yolov5s_yolov5m"
NOISE_OUT_DIR   = os.path.join('universal_cloak', GAME, f'ensemble_{ensemble_tag}')
os.makedirs(NOISE_OUT_DIR, exist_ok=True)
NOISE_SAVE_PATH = os.path.join(NOISE_OUT_DIR, 'universal_noise.pt')

RESULT_DIR  = 'result'
os.makedirs(RESULT_DIR, exist_ok=True)
TIMING_XLSX = os.path.join(RESULT_DIR, 'cloak_timing_ensemble.xlsx')
TIMING_PNG  = os.path.join(RESULT_DIR, 'cloak_timing_ensemble.png')

print(f"\n{'='*65}")
print(f"  Game    : {GAME}")
print(f"  Ensemble: {MODEL_LIST}")
print(f"  Weights : {[round(w,3) for w in MODEL_WEIGHTS]}")
print(f"  Data    : {DATA_PATH}")
print(f"  Iter    : {args.n_iter}  LR : {args.lr}  EPS : {args.epsilon}/255")
print(f"  SSIM w  : {args.ssim_w}")
print(f"{'='*65}\n")


# ─── DATASET ────────────────────────────────────────────────
def load_images_original(folder):
    """Load images at ORIGINAL size → list of tensors (3,H,W)"""
    exts  = ('.jpg','.jpeg','.png','.bmp')
    paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(exts)])
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    print(f"  Found {len(paths)} images")
    tensors, sizes = [], []
    for p in paths:
        img = Image.open(p).convert('RGB')
        sizes.append(img.size)          # (W, H)
        t = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
        tensors.append(t)
    return tensors, sizes               # list of (3,H,W), list of (W,H)


def to_model_input(tensor, size):
    """Resize (3,H,W) → (1,3,size,size)"""
    return F.interpolate(tensor.unsqueeze(0), size=(size, size),
                         mode='bilinear', align_corners=False)


# ─── MODEL LOADER ────────────────────────────────────────────
def load_yolov5(variant: str):
    """Load any YOLOv5 variant from pretrained_models/<variant>.pt"""
    sys.path.insert(0, os.path.abspath('.'))
    from models.common import DetectMultiBackend
    pt_path = os.path.join('pretrained_models', f'{variant}.pt')
    model   = DetectMultiBackend(pt_path, device=torch.device(DEVICE), fuse=True)
    model.eval()
    print(f"  [{variant}] Loaded from {pt_path}")
    return model


def load_all_models(model_list):
    """Load every model in the ensemble, return as a list."""
    models = []
    for name in model_list:
        m = load_yolov5(name)
        models.append(m)
    return models


# ─── LOSS PER MODEL ──────────────────────────────────────────
_bce = nn.BCELoss()

def loss_yolov5_single(model, adv_batch):
    """
    YOLOv5 detection loss for one model.
    adv_batch : (B, 3, S, S)  already on DEVICE, values in [0,1]
    Returns scalar BCE loss (suppresses all confidence scores → 0).

    NOTE: DetectMultiBackend in eval mode already applies sigmoid to
    objectness/class scores inside the Detect layer. Do NOT apply
    torch.sigmoid() again here — that was causing a double-sigmoid
    bug that crushed the gradient and produced a too-weak cloak.
    """
    preds = model(adv_batch)
    raw = preds[0] if isinstance(preds, (list, tuple)) else preds
    conf = raw[..., 4:5].clamp(1e-6, 1 - 1e-6)   # clamp for BCE numerical safety
    return _bce(conf, torch.zeros_like(conf))


# ─── ENSEMBLE LOSS ───────────────────────────────────────────
def ensemble_loss(models, weights, adv_batch):
    """
    Compute weighted-average BCE loss across all models.
    Each model sees the SAME adversarial batch.
    """
    total = torch.tensor(0.0, device=DEVICE)
    for model, w in zip(models, weights):
        l = loss_yolov5_single(model, adv_batch)
        total = total + w * l
    return total


# ─── TRAINING ────────────────────────────────────────────────
def train_universal_cloak(img_tensors, img_sizes, models, model_weights):
    S  = MODEL_INPUT
    # Initialise noise at model input size
    noise = torch.empty(3, S, S, device=DEVICE).uniform_(-EPS, EPS)
    noise.requires_grad_(True)
    optimizer = torch.optim.Adam([noise], lr=args.lr)

    N        = len(img_tensors)
    BS       = args.batch_size
    best_loss  = float('inf')
    best_noise = noise.detach().clone()
    losses     = []
    t0         = time.time()

    num_models = len(models)
    print(f"  Training {args.n_iter} iter | batch={BS} | {N} images | "
          f"input={S}x{S} | {num_models} models")

    for it in range(1, args.n_iter + 1):
        idx        = torch.randperm(N)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, BS):
            batch_idx = idx[start:start + BS].tolist()
            # Resize to model input size
            batch = torch.stack([
                to_model_input(img_tensors[i], S)[0] for i in batch_idx
            ]).to(DEVICE)                                   # (b, 3, S, S)

            # Apply noise (clamp to ε-ball)
            n_c = torch.clamp(noise, -EPS, EPS)
            adv = torch.clamp(batch + n_c.unsqueeze(0), 0.0, 1.0)

            # ── Ensemble detection loss ───────────────────────
            det_loss = ensemble_loss(models, model_weights, adv)

            # ── SSIM perceptual loss ──────────────────────────
            ssim_val  = ssim_fn((batch * 255).detach().cpu(),
                                (adv   * 255).detach().cpu(),
                                data_range=255, size_average=True)
            ssim_loss = 1.0 - ssim_val.to(DEVICE)

            loss = det_loss + args.ssim_w * ssim_loss

            # ── Update ────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                noise.clamp_(-EPS, EPS)

            epoch_loss += det_loss.item()
            n_batches  += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)

        if avg < best_loss:
            best_loss  = avg
            best_noise = noise.detach().clone()

        if it % 30 == 0 or it == 1:
            print(f"    [{it:4d}/{args.n_iter}]  bce={avg:.6f}  "
                  f"best={best_loss:.6f}  t={time.time()-t0:.0f}s")

    total_time = time.time() - t0
    print(f"\n  Done — {total_time:.1f}s ({total_time/60:.1f} min)  best={best_loss:.6f}")
    return best_noise.cpu(), total_time, losses


# ─── TIMING ──────────────────────────────────────────────────
def save_timing(game, model_tag, total_time_sec, n_images, n_iter):
    row = {
        'Game': game.upper(), 'Models': model_tag,
        'Images': n_images, 'Iterations': n_iter,
        'TotalTime_s': round(total_time_sec, 2),
        'TimePerIter_s': round(total_time_sec / n_iter, 4)
    }
    if os.path.exists(TIMING_XLSX):
        df = pd.read_excel(TIMING_XLSX)
        df = df[~((df['Game'] == row['Game']) & (df['Models'] == row['Models']))]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(TIMING_XLSX, index=False)
    print(f"  Timing → {TIMING_XLSX}")

    if len(df) >= 1:
        labels = [f"{r['Game']}\n{r['Models']}" for _, r in df.iterrows()]
        times  = df['TotalTime_s'].tolist()
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 5))
        bars = ax.bar(labels, times, color='steelblue', width=0.55)
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(times) * 0.01,
                    f'{val:.0f}s', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('Total Training Time (s)')
        ax.set_title('Universal Cloak — Ensemble Training Time')
        ax.set_ylim(0, max(times) * 1.18)
        plt.tight_layout()
        plt.savefig(TIMING_PNG, dpi=150)
        plt.close()
        print(f"  Chart  → {TIMING_PNG}")


# ─── MAIN ────────────────────────────────────────────────────
def main():
    print("\n[1] Loading images (original size) …")
    img_tensors, img_sizes = load_images_original(DATA_PATH)
    print(f"  Sample size: {img_sizes[0]}")

    print(f"\n[2] Loading ensemble models: {MODEL_LIST} …")
    models = load_all_models(MODEL_LIST)
    print(f"  Weights (normalised): {[round(w, 4) for w in MODEL_WEIGHTS]}")

    print("\n[3] Training ensemble universal cloak …")
    best_noise, total_time, loss_curve = train_universal_cloak(
        img_tensors, img_sizes, models, MODEL_WEIGHTS)

    torch.save(best_noise, NOISE_SAVE_PATH)
    print(f"\n[4] Noise saved → {NOISE_SAVE_PATH}")
    print(f"    Shape : {best_noise.shape}  ({MODEL_INPUT}x{MODEL_INPUT})")
    print(f"    Range : [{best_noise.min():.4f}, {best_noise.max():.4f}]")

    loss_png = os.path.join(NOISE_OUT_DIR, 'loss_curve.png')
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, color='crimson')
    plt.xlabel('Iteration')
    plt.ylabel('Ensemble BCE Loss')
    plt.title(f'Loss Curve — {GAME.upper()} / {ensemble_tag}')
    plt.tight_layout()
    plt.savefig(loss_png, dpi=150)
    plt.close()
    print(f"    Loss curve → {loss_png}")

    print("\n[5] Saving timing …")
    save_timing(GAME, ensemble_tag, total_time, len(img_tensors), args.n_iter)

    print(f"\n{'='*65}")
    print(f"  COMPLETE : {GAME.upper()} / Ensemble={ensemble_tag}")
    print(f"  Time     : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()