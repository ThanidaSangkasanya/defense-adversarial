"""
inject_cloak.py — Apply Universal Cloak to a folder of images & save as PNG

DEMO VERSION: default input is C:\\Users\\tsang\\Dataset\\demo\\frames
              default output is C:\\Users\\tsang\\Dataset\\demo\\frames_cloaked

Usage:
    python inject_cloak.py --game cs2
    python inject_cloak.py --game cs2 --data_dir D:\\x\\frames --out_dir D:\\x\\frames_cloaked
    python inject_cloak.py --game cf --noise_dir universal_cloak/cf/ensemble_yolov5n_yolov5s_yolov5m
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim as ssim_fn
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--game',    required=True, type=str, choices=['cs2', 'cf'])
parser.add_argument('--epsilon', default=8,     type=int)
parser.add_argument('--gpu',     default='0',   type=str)
parser.add_argument('--noise_dir', default=None, type=str,
                    help='Override noise directory '
                         '(default: universal_cloak/<game>/ensemble_yolov5n_yolov5s_yolov5m)')
parser.add_argument('--data_dir', default=r'C:\Users\tsang\Dataset\demo\frames', type=str,
                    help='Folder with input images')
parser.add_argument('--out_dir',  default=None, type=str,
                    help='Output folder (default: <data_dir>\\..\\frames_cloaked)')
args = parser.parse_args()

DEVICE = f'cuda:{args.gpu}'
EPS    = args.epsilon / 255.0
GAME   = args.game

DATA_PATH = args.data_dir
OUT_PATH  = args.out_dir or os.path.join(os.path.dirname(os.path.normpath(DATA_PATH)),
                                         'frames_cloaked')
os.makedirs(OUT_PATH, exist_ok=True)

if args.noise_dir:
    NOISE_DIR = args.noise_dir
else:
    NOISE_DIR = os.path.join('universal_cloak', GAME, 'ensemble_yolov5n_yolov5s_yolov5m')
NOISE_PATH = os.path.join(NOISE_DIR, 'universal_noise.pt')

print(f"\n{'='*60}")
print(f"  Game      : {GAME}")
print(f"  Input     : {DATA_PATH}")
print(f"  Output    : {OUT_PATH}")
print(f"  Noise     : {NOISE_PATH}")
print(f"  Epsilon   : {args.epsilon}/255")
print(f"{'='*60}\n")

if not os.path.exists(NOISE_PATH):
    raise FileNotFoundError(f"Noise not found: {NOISE_PATH}\n"
                            f"Run train_cloak_multimodel.py first.")
noise_model_size = torch.load(NOISE_PATH).to(DEVICE)
print(f"  Noise shape: {noise_model_size.shape}")

exts  = ('.jpg', '.jpeg', '.png', '.bmp')
if not os.path.isdir(DATA_PATH):
    raise FileNotFoundError(f"Folder not found: {DATA_PATH}")
files = sorted([f for f in os.listdir(DATA_PATH) if f.lower().endswith(exts)])
if not files:
    raise FileNotFoundError(f"No images in {DATA_PATH}")
print(f"  Found {len(files)} images\n")

# ── Warm-up (tiny dummy pass so first real image isn't slow) ──
_w = torch.zeros(1, 3, 64, 64, device=DEVICE)
_ = F.interpolate(noise_model_size.unsqueeze(0), size=(64, 64),
                  mode='bilinear', align_corners=False)
_ = ssim_fn((_w * 255).cpu(), (_w * 255).cpu(), data_range=255, size_average=False)
if torch.cuda.is_available():
    torch.cuda.synchronize()

ssim_vals = []
t0 = time.time()

for fname in tqdm(files, desc='Injecting cloak'):
    img_pil = Image.open(os.path.join(DATA_PATH, fname)).convert('RGB')
    orig_w, orig_h = img_pil.size

    img_t = torch.from_numpy(
        np.array(img_pil).transpose(2, 0, 1)
    ).float().unsqueeze(0).to(DEVICE) / 255.0        # (1, 3, H, W)

    # Resize noise to original image size
    noise_orig = F.interpolate(
        noise_model_size.unsqueeze(0),
        size=(orig_h, orig_w),
        mode='bilinear', align_corners=False
    )                                                  # (1, 3, H, W)
    noise_clamped = torch.clamp(noise_orig, -EPS, EPS)

    adv_t = torch.clamp(img_t + noise_clamped, 0.0, 1.0)

    ssim_val = ssim_fn((img_t * 255).cpu(), (adv_t * 255).cpu(),
                       data_range=255, size_average=False).item()
    ssim_vals.append(ssim_val)

    # Save as PNG (lossless)
    adv_np   = (adv_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    out_name = os.path.splitext(fname)[0] + '.png'
    Image.fromarray(adv_np).save(os.path.join(OUT_PATH, out_name))

avg_ssim = np.mean(ssim_vals)
print(f"\n{'='*60}")
print(f"  Done — {len(files)} images saved to {OUT_PATH}  ({time.time()-t0:.1f}s)")
print(f"  Average SSIM (universal only): {avg_ssim:.4f}")
print(f"{'='*60}\n")