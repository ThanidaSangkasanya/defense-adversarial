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
parser.add_argument('--model',     required=True,  type=str,
                    choices=['yolov5n','nanodet','rtdetr'])
parser.add_argument('--n_iter',    default=300,    type=int)
parser.add_argument('--lr',        default=0.0005, type=float)
parser.add_argument('--epsilon',   default=8,      type=int)
parser.add_argument('--gpu',       default='0',    type=str)
parser.add_argument('--batch_size',default=8,      type=int)
parser.add_argument('--ssim_w',    default=0.3,    type=float,
                    help='Weight of SSIM loss (default: 0.3)')
args = parser.parse_args()

DEVICE = f'cuda:{args.gpu}'
EPS    = args.epsilon / 255.0
GAME   = args.game
MODEL  = args.model

GAME_FOLDER = {'cs2':'CS2','cf':'CF','valorant':'Valorant','overwatch':'Overwatch'}
DATA_PATH   = os.path.join(r'C:\Users\tsang\Dataset', GAME_FOLDER[GAME])

# Model input sizes
MODEL_SIZE  = {'yolov5n': 416, 'nanodet': 320, 'rtdetr': 640}
MODEL_INPUT = MODEL_SIZE[MODEL]

NOISE_OUT_DIR   = os.path.join('universal_cloak', GAME, MODEL)
os.makedirs(NOISE_OUT_DIR, exist_ok=True)
NOISE_SAVE_PATH = os.path.join(NOISE_OUT_DIR, 'universal_noise.pt')

RESULT_DIR  = 'result'
os.makedirs(RESULT_DIR, exist_ok=True)
TIMING_XLSX = os.path.join(RESULT_DIR, 'cloak_timing.xlsx')
TIMING_PNG  = os.path.join(RESULT_DIR, 'cloak_timing.png')

print(f"\n{'='*55}")
print(f"  Game  : {GAME}  |  Model : {MODEL}")
print(f"  Data  : {DATA_PATH}")
print(f"  Iter  : {args.n_iter}  LR : {args.lr}  EPS : {args.epsilon}/255")
print(f"  SSIM weight : {args.ssim_w}")
print(f"{'='*55}\n")


# ─── LOAD DATASET (original size) ────────────────────────────
def load_images_original(folder):
    """Load images at ORIGINAL size → list of tensors (3,H,W)"""
    exts  = ('.jpg','.jpeg','.png','.bmp')
    paths = sorted([os.path.join(folder,f) for f in os.listdir(folder)
                    if f.lower().endswith(exts)])
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    print(f"  Found {len(paths)} images")
    tensors, sizes = [], []
    for p in paths:
        img = Image.open(p).convert('RGB')
        sizes.append(img.size)          # (W, H)
        t = torch.from_numpy(np.array(img).transpose(2,0,1)).float() / 255.0
        tensors.append(t)
    return tensors, sizes               # list of (3,H,W), list of (W,H)


def to_model_input(tensor, model_input_size):
    """Resize (3,H,W) → (1,3,S,S) for model"""
    img = tensor.unsqueeze(0)           # (1,3,H,W)
    return F.interpolate(img, size=(model_input_size, model_input_size),
                         mode='bilinear', align_corners=False)


def noise_to_original(noise_model, orig_h, orig_w):
    """Resize noise from model_input_size → original size"""
    n = noise_model.unsqueeze(0)        # (1,3,S,S)
    return F.interpolate(n, size=(orig_h, orig_w),
                         mode='bilinear', align_corners=False).squeeze(0)


# ─── MODEL LOADERS ───────────────────────────────────────────
def load_yolov5n():
    sys.path.insert(0, os.path.abspath('.'))
    from models.common import DetectMultiBackend
    model = DetectMultiBackend(
        os.path.join('pretrained_models','yolov5n.pt'),
        device=torch.device(DEVICE), fuse=True)
    model.eval()
    print("  [YOLOv5n] Loaded.")
    return model

def load_nanodet():
    root = r'C:\Users\tsang\nanodet'
    sys.path.insert(0, root)
    from nanodet.util import cfg, load_config, Logger as NL
    from nanodet.model.arch import build_model
    from nanodet.util import load_model_weight
    load_config(cfg, os.path.join(root,'config','nanodet-plus-m_320.yml'))
    model = build_model(cfg.model)
    ckpt  = torch.load(os.path.join(root,'nanodet','nanodet-plus-m_320.pth'),
                       map_location='cpu')
    load_model_weight(model, ckpt, NL(0, use_tensorboard=False))
    model = model.to(DEVICE).eval()
    print("  [NanoDet-Plus] Loaded.")
    return model, cfg

def load_rtdetr():
    from ultralytics import YOLO
    yolo  = YOLO(r'C:\Users\tsang\RT-DETR\rtdetr-l.pt')
    model = yolo.model.to(DEVICE).eval()
    print("  [RT-DETR] Loaded.")
    return model


# ─── LOSS FUNCTIONS ──────────────────────────────────────────
_bce = nn.BCELoss()

def loss_yolov5(model, adv_model):
    preds = model(adv_model)
    raw   = preds[0] if isinstance(preds,(list,tuple)) else preds
    conf  = torch.sigmoid(raw[...,4:5])
    return _bce(conf, torch.zeros_like(conf))

def loss_nanodet(model, adv_model):
    mean = torch.tensor([103.53,116.28,123.675],device=DEVICE).view(1,3,1,1)/255.0
    std  = torch.tensor([57.375, 57.12, 58.395],device=DEVICE).view(1,3,1,1)/255.0
    img_norm = (adv_model - mean) / std
    feats = model.backbone(img_norm)
    feats = model.fpn(feats)
    preds = model.head(feats)
    total = torch.tensor(0.0, device=DEVICE)
    for p in preds:
        if isinstance(p,(list,tuple)): p = p[0]
        prob  = torch.sigmoid(p[...,0:1])
        total = total + _bce(prob, torch.zeros_like(prob))
    return total

def loss_rtdetr(model, adv_model):
    out = model(adv_model)
    if isinstance(out,dict) and 'pred_logits' in out:
        logits = out['pred_logits']
        person = torch.sigmoid(logits[:,:,0:1])
        return _bce(person, torch.zeros_like(person))
    raw  = out[0] if isinstance(out,(list,tuple)) else out
    conf = torch.sigmoid(raw[...,4:5])
    return _bce(conf, torch.zeros_like(conf))


# ─── TRAINING ────────────────────────────────────────────────
def train_universal_cloak(img_tensors, img_sizes, model, model_type):
    """
    Train noise at MODEL_INPUT size.
    SSIM computed on original-size image for perceptual quality.
    """
    S  = MODEL_INPUT
    # Step 1 — init noise at model input size
    noise = torch.empty(3, S, S, device=DEVICE).uniform_(-EPS, EPS)
    noise.requires_grad_(True)
    optimizer = torch.optim.Adam([noise], lr=args.lr)

    N        = len(img_tensors)
    BS       = args.batch_size
    best_loss = float('inf')
    best_noise = noise.detach().clone()
    losses     = []
    t0         = time.time()

    print(f"  Training {args.n_iter} iter | batch={BS} | {N} images | input={S}x{S}")

    for it in range(1, args.n_iter+1):
        idx        = torch.randperm(N)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, BS):
            batch_idx = idx[start:start+BS].tolist()
            # Resize batch to model input size
            batch = torch.stack([
                to_model_input(img_tensors[i], S)[0] for i in batch_idx
            ]).to(DEVICE)                         # (b,3,S,S)

            # Step 2 — apply noise
            n_c     = torch.clamp(noise, -EPS, EPS)
            adv     = torch.clamp(batch + n_c.unsqueeze(0), 0.0, 1.0)

            # Step 3 — detection loss (BCE)
            if model_type == 'yolov5n':
                det_loss = loss_yolov5(model, adv)
            elif model_type == 'nanodet':
                det_loss = loss_nanodet(model, adv)
            elif model_type == 'rtdetr':
                det_loss = loss_rtdetr(model, adv)

            # SSIM loss — penalise if noise is visible
            # computed at model input size for speed
            ssim_val = ssim_fn((batch*255).detach().cpu(),
                               (adv*255).detach().cpu(),
                               data_range=255, size_average=True)
            ssim_loss = 1.0 - ssim_val.to(DEVICE)

            loss = det_loss + args.ssim_w * ssim_loss

            # Step 4 — update
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


# ─── TIMING SAVE ─────────────────────────────────────────────
def save_timing(game, model_name, total_time_sec, n_images, n_iter):
    row = {'Game': game.upper(), 'Model': model_name, 'Images': n_images,
           'Iterations': n_iter, 'TotalTime_s': round(total_time_sec,2),
           'TimePerIter_s': round(total_time_sec/n_iter,4)}
    if os.path.exists(TIMING_XLSX):
        df = pd.read_excel(TIMING_XLSX)
        df = df[~((df['Game']==row['Game'])&(df['Model']==row['Model']))]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(TIMING_XLSX, index=False)
    print(f"  Timing → {TIMING_XLSX}")
    if len(df) >= 1:
        labels = [f"{r['Game']}\n{r['Model']}" for _,r in df.iterrows()]
        times  = df['TotalTime_s'].tolist()
        fig,ax = plt.subplots(figsize=(max(8,len(df)*1.0),5))
        bars = ax.bar(labels, times, color='steelblue', width=0.55)
        for bar,val in zip(bars,times):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(times)*0.01,
                    f'{val:.0f}s', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('Total Training Time (s)')
        ax.set_title('Universal Cloak — Training Time per Game × Model')
        ax.set_ylim(0, max(times)*1.18)
        plt.tight_layout()
        plt.savefig(TIMING_PNG, dpi=150)
        plt.close()
        print(f"  Chart  → {TIMING_PNG}")


# ─── MAIN ────────────────────────────────────────────────────
def main():
    print("\n[1] Loading images (original size) …")
    img_tensors, img_sizes = load_images_original(DATA_PATH)
    print(f"  Sample size: {img_sizes[0]}")

    print(f"\n[2] Loading model: {MODEL} …")
    nano_cfg = None
    if MODEL == 'yolov5n':
        model = load_yolov5n()
    elif MODEL == 'nanodet':
        model, nano_cfg = load_nanodet()
    elif MODEL == 'rtdetr':
        model = load_rtdetr()

    print("\n[3] Training …")
    best_noise, total_time, loss_curve = train_universal_cloak(
        img_tensors, img_sizes, model, MODEL)

    torch.save(best_noise, NOISE_SAVE_PATH)
    print(f"\n[4] Noise saved → {NOISE_SAVE_PATH}")
    print(f"    Shape : {best_noise.shape}  (model input size {MODEL_INPUT}x{MODEL_INPUT})")
    print(f"    Range : [{best_noise.min():.4f}, {best_noise.max():.4f}]")

    loss_png = os.path.join(NOISE_OUT_DIR, 'loss_curve.png')
    plt.figure(figsize=(8,4))
    plt.plot(range(1,len(loss_curve)+1), loss_curve, color='crimson')
    plt.xlabel('Iteration'); plt.ylabel('BCE Loss')
    plt.title(f'Loss Curve — {GAME.upper()} / {MODEL}')
    plt.tight_layout(); plt.savefig(loss_png, dpi=150); plt.close()
    print(f"    Loss curve → {loss_png}")

    print("\n[5] Saving timing …")
    save_timing(GAME, MODEL, total_time, len(img_tensors), args.n_iter)

    print(f"\n{'='*55}")
    print(f"  COMPLETE : {GAME.upper()} / {MODEL}")
    print(f"  Time     : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"{'='*55}\n")

if __name__ == '__main__':
    main()