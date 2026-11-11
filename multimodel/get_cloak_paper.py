"""
get_cloak_paper.py — Cloak Crafter: Universal Cloak + Refinement + Evaluation
Matches the paper's Table 2 format: Cheating Model | Dataset | DSR(%) | FPS | #Query | SSIM

Flow per image (following §4.2 of the Invisibility Cloak paper):
  1. Apply Universal Cloak  →  check proxy model  →  if no detection: success (query=1)
  2. If detection persists   →  refine noise (Adam, up to n_iter)  →  early stop on success
  3. Final evaluation with cheating model (may differ from proxy)

DEMO VERSION
  - default data  : C:\\Users\\tsang\\Dataset\\demo\\frames  (+ labels)
  - GPU warm-up before timing, so FPS is not polluted by CUDA/cuDNN init
  - saves 4 views per image: orig_boxes / attack_clean / attack_boxes / compare

Usage:
  python get_cloak_paper.py --game cs2 --cheating_model yolov5n
  python get_cloak_paper.py --game cs2 --cheating_model yolov5n yolov5s yolov5m
  python get_cloak_paper.py --game cs2 --proxy_model yolov5n --cheating_model yolov5s
  python get_cloak_paper.py --game cs2 --cheating_model yolov5n --data_dir D:\\x\\frames --gt_dir D:\\x\\labels
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
parser.add_argument('--game',           required=True, type=str, choices=['cs2', 'cf'])
parser.add_argument('--cheating_model', nargs='+', required=True,
                    choices=['yolov5n', 'yolov5s', 'yolov5m', 'yolov5x'],
                    help='Cheating model(s) to evaluate. Can pass multiple.')
parser.add_argument('--proxy_model',    default=None, type=str,
                    choices=['yolov5n', 'yolov5s', 'yolov5m', 'yolov5x'],
                    help='Proxy model for refinement (default: same as cheating_model)')
parser.add_argument('--n_iter',   default=5,     type=int,
                    help='Max refinement iterations (default: 5)')
parser.add_argument('--lr',       default=0.005, type=float,
                    help='Refinement learning rate (default: 0.005)')
parser.add_argument('--epsilon',  default=8,     type=int)
parser.add_argument('--conf',     default=0.4,   type=float,
                    help='Confidence threshold for DSR (default: 0.4)')
parser.add_argument('--gpu',      default='0',   type=str)
parser.add_argument('--noise_dir', default=None, type=str,
                    help='Override noise directory')
parser.add_argument('--input_size', default=320, type=int,
                    help='Detection input size for evaluation (default: 320)')
parser.add_argument('--use_universal', default=1, type=int,
                    help='1=use universal cloak, 0=random init (default: 1)')
parser.add_argument('--tag',      default='demo', type=str,
                    help="Suffix for result files (default: 'demo'; use '' for none)")
parser.add_argument('--data_dir', default=r'C:\Users\tsang\Dataset\demo\frames', type=str,
                    help='Folder with input images')
parser.add_argument('--gt_dir',   default=r'C:\Users\tsang\Dataset\demo\labels', type=str,
                    help='Folder with YOLO .txt labels (optional; missing = no green boxes)')
parser.add_argument('--warmup',   default=3, type=int,
                    help='GPU warm-up rounds before timing starts (default: 3, 0 = off)')
args = parser.parse_args()

DEVICE = f'cuda:{args.gpu}'
EPS    = args.epsilon / 255.0
GAME   = args.game

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH = args.data_dir
GT_PATH   = args.gt_dir

if args.noise_dir:
    NOISE_DIR = args.noise_dir
else:
    NOISE_DIR = os.path.join('universal_cloak', GAME,
                             'ensemble_yolov5n_yolov5s_yolov5m')
NOISE_PATH = os.path.join(NOISE_DIR, 'universal_noise.pt')

suffix     = f'_{args.tag}' if args.tag else ''
RESULT_DIR = 'result'
EVAL_XLSX  = os.path.join(RESULT_DIR, f'cloak_crafter_table2{suffix}.xlsx')
EVAL_PNG   = os.path.join(RESULT_DIR, f'cloak_crafter_table2{suffix}.png')
os.makedirs(RESULT_DIR, exist_ok=True)

BCEobj = nn.BCELoss()

print(f"\n{'='*65}")
print(f"  Game            : {GAME.upper()}")
print(f"  Cheating models : {args.cheating_model}")
print(f"  Proxy model     : {args.proxy_model or '(same as cheating)'}")
print(f"  Data            : {DATA_PATH}")
print(f"  GT labels       : {GT_PATH}")
print(f"  Noise           : {NOISE_PATH}")
print(f"  Refinement      : n_iter={args.n_iter}  lr={args.lr}  eps={args.epsilon}/255")
print(f"  Conf threshold  : {args.conf}")
print(f"  Warm-up rounds  : {args.warmup}")
print(f"{'='*65}\n")


# ─── UNIVERSAL CLOAK ─────────────────────────────────────────
global_noise = None
if args.use_universal == 1:
    if not os.path.exists(NOISE_PATH):
        raise FileNotFoundError(f"Noise not found: {NOISE_PATH}")
    global_noise = torch.load(NOISE_PATH).to(DEVICE)
    print(f"  Universal Cloak loaded: {global_noise.shape}")
else:
    print("  No Universal Cloak (random init)")


# ─── HELPERS ─────────────────────────────────────────────────
def sync():
    """Wait for the GPU so wall-clock timing is accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_image_paths(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(exts)])
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    return paths


def pil_to_tensor(img_pil):
    return torch.from_numpy(
        np.array(img_pil).transpose(2, 0, 1)
    ).float().unsqueeze(0).to(DEVICE) / 255.0          # (1, 3, H, W)


# ─── MODEL LOADERS ───────────────────────────────────────────
def load_proxy_model(variant):
    """Load YOLOv5 WITHOUT AutoShape — for gradient-based refinement."""
    sys.path.insert(0, os.path.abspath('.'))
    from models.common import DetectMultiBackend
    pt_path = os.path.join('pretrained_models', f'{variant}.pt')
    model   = DetectMultiBackend(pt_path, device=torch.device(DEVICE), fuse=True)
    model.eval()
    print(f"  [Proxy] {variant} loaded (DetectMultiBackend)")
    return model


def load_cheating_model(variant):
    """Load YOLOv5 WITH AutoShape — for evaluation (NMS + conf)."""
    sys.path.insert(0, os.path.abspath('.'))
    from models.common import DetectMultiBackend, AutoShape
    pt_path = os.path.join('pretrained_models', f'{variant}.pt')
    model   = DetectMultiBackend(pt_path, device=torch.device(DEVICE), fuse=True)
    model   = AutoShape(model)
    model.conf    = args.conf
    model.classes = [0]       # person only
    print(f"  [Cheating] {variant} loaded (AutoShape, conf={args.conf})")
    return model


# ─── ATTACK SUCCESS CHECK (on proxy raw output) ──────────────
def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def attack_success(preds, conf_thres=0.4):
    """Check if attack succeeds on proxy model raw output.
    Returns number of person detections (0 = success)."""
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = preds.detach()
    xc    = preds[..., 4] > 0.25       # objectness candidates
    x     = preds[0][xc[0]]
    if x.shape[0] == 0:
        return 0
    x[:, 5:] *= x[:, 4:5]              # conf = obj_conf * cls_conf
    box  = xywh2xyxy(x[:, :4])
    conf, j = x[:, 5:85].max(1, keepdim=True)
    x    = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
    x    = x[(x[:, 5:6] == 0).any(1)]  # class 0 = person
    return x.shape[0]


# ─── FindNoise MODULE (from paper) ───────────────────────────
class FindNoise(nn.Module):
    def __init__(self, model, epsilon, noise):
        super().__init__()
        self.model   = model
        self.epsilon = epsilon
        self.noise   = noise
        self.noise.requires_grad = True

    def forward(self, x):
        noise   = torch.clamp(self.noise, -self.epsilon, self.epsilon)
        adv_img = torch.clamp(x + noise, 0.0, 1.0)
        preds   = self.model(adv_img)
        return preds

    def get_adv(self, x):
        with torch.no_grad():
            noise   = torch.clamp(self.noise, -self.epsilon, self.epsilon)
            adv_img = torch.clamp(x + noise, 0.0, 1.0)
        return adv_img


# ─── CLOAK CRAFTER (§4.1 + §4.2) ─────────────────────────────
def cloak_crafter(x, proxy_model, n_iter, eps, lr):
    """
    Full Cloak Crafter pipeline per image.
    x: (1, 3, H, W) tensor on DEVICE, values [0,1]

    Returns: adv_img, query_count, local_success, elapsed_time
    """
    sync()
    t_start = time.time()
    query   = 0

    # ── Step 1: Try Universal Cloak ───────────────────────────
    if global_noise is not None:
        _, _, H, W = x.shape
        noise_resized = F.interpolate(
            global_noise.unsqueeze(0), size=(H, W),
            mode='bilinear', align_corners=False
        ).squeeze(0)                                    # (3, H, W)

        with torch.no_grad():
            eta     = torch.clamp(noise_resized, -eps, eps)
            x_adv   = torch.clamp(x + eta.unsqueeze(0), 0.0, 1.0)
            preds   = proxy_model(x_adv)
            query  += 1
            n_det   = attack_success(preds, conf_thres=args.conf)
            if n_det == 0:
                sync()
                return x_adv, query, 1, time.time() - t_start
            # Universal cloak insufficient → use as init for refinement
            eta = noise_resized.clone()
    else:
        _, _, H, W = x.shape
        eta = torch.empty((1, 3, H, W), device=DEVICE).uniform_(-eps, eps).squeeze(0)

    # ── Step 2: Cloak Refinement (§4.2) ───────────────────────
    noise_model = FindNoise(proxy_model, eps, eta).to(DEVICE)
    optimizer   = torch.optim.Adam([noise_model.noise], lr=lr)

    n_det   = 1   # assume not yet successful
    x_input = x.detach().clone().requires_grad_(True)

    for k in range(n_iter):
        preds = noise_model(x_input)
        query += 1

        n_det = attack_success(preds, conf_thres=args.conf)
        if n_det == 0:
            break

        # BCE loss: suppress all objectness → 0
        # raw[..., 4] is ALREADY a probability (Detect layer applies sigmoid
        # in eval mode) — do NOT apply sigmoid again.
        raw  = preds[0] if isinstance(preds, (list, tuple)) else preds
        obj  = raw[..., 4:5].clamp(1e-6, 1 - 1e-6)
        gt   = torch.zeros_like(obj)
        loss = BCEobj(obj, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proxy_model.zero_grad()

    x_adv   = noise_model.get_adv(x)
    success = 1 if n_det == 0 else 0
    sync()
    return x_adv, query, success, time.time() - t_start


# ─── EVALUATION WITH CHEATING MODEL ──────────────────────────
def evaluate_cheating(cheating_model, img_pil, input_size):
    """Run cheating model on an image → list of person boxes."""
    img_np  = np.array(img_pil)
    results = cheating_model(img_np, size=input_size)
    boxes   = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().tolist():
        if int(cls) == 0:
            boxes.append([int(xyxy[0]), int(xyxy[1]),
                          int(xyxy[2]), int(xyxy[3]), float(conf)])
    return boxes


# ─── GPU WARM-UP (not timed, nothing saved) ──────────────────
def warmup(proxy_model, cheating_model, sample_path, rounds):
    """Run the same code paths the real loop uses (universal forward,
    refinement forward+backward, cheating model) on the first image so that
    CUDA context / cuDNN autotune / kernel loading is paid BEFORE timing."""
    if rounds <= 0:
        return
    t0 = time.time()
    print(f"  Warm-up ({rounds} rounds) …", end=' ', flush=True)
    img_pil = Image.open(sample_path).convert('RGB')
    x = pil_to_tensor(img_pil)
    for _ in range(rounds):
        # universal-cloak style forward (no grad)
        with torch.no_grad():
            _ = proxy_model(x)
        # refinement style forward + backward (with grad)
        noise = torch.zeros_like(x[0]).requires_grad_(True)
        adv   = torch.clamp(x + noise, 0.0, 1.0)
        preds = proxy_model(adv)
        raw   = preds[0] if isinstance(preds, (list, tuple)) else preds
        obj   = raw[..., 4:5].clamp(1e-6, 1 - 1e-6)
        BCEobj(obj, torch.zeros_like(obj)).backward()
        proxy_model.zero_grad()
        # cheating-model evaluation (AutoShape + NMS)
        _ = evaluate_cheating(cheating_model, img_pil, args.input_size)
    sync()
    print(f"done in {time.time() - t0:.1f}s")


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


def side_by_side(left, right, label_l='Original (no cloak)', label_r='Cloaked'):
    """Concatenate two PIL images horizontally with small captions."""
    w, h = left.size
    canvas = Image.new('RGB', (w * 2, h + 24), 'white')
    canvas.paste(left, (0, 24))
    canvas.paste(right, (w, 24))
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), label_l, fill='black')
    d.text((w + 6, 6), label_r, fill='black')
    return canvas


# ─── GT LOADER ────────────────────────────────────────────────
def load_gt_boxes(label_path, img_w, img_h):
    if not os.path.exists(label_path):
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
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


# ─── RUN ONE CHEATING MODEL ──────────────────────────────────
def run_evaluation(cheating_name, proxy_name, img_paths):
    """Full evaluation for one cheating model → returns row dict."""
    print(f"\n{'─'*55}")
    print(f"  Proxy: {proxy_name}  →  Cheating: {cheating_name}")
    print(f"{'─'*55}")

    proxy_model    = load_proxy_model(proxy_name)
    cheating_model = load_cheating_model(cheating_name)

    # Output dirs
    vis_dir   = os.path.join(RESULT_DIR, 'visualization', GAME,
                             f'proxy_{proxy_name}_cheat_{cheating_name}{suffix}')
    orig_dir  = os.path.join(vis_dir, 'orig_boxes')      # before cloak + boxes
    clean_dir = os.path.join(vis_dir, 'attack_clean')    # after cloak, no boxes
    box_dir   = os.path.join(vis_dir, 'attack_boxes')    # after cloak + boxes
    cmp_dir   = os.path.join(vis_dir, 'compare')         # orig_boxes | attack_boxes
    for d in (orig_dir, clean_dir, box_dir, cmp_dir):
        os.makedirs(d, exist_ok=True)

    log_dir = os.path.join(RESULT_DIR, 'log', GAME)
    os.makedirs(log_dir, exist_ok=True)

    # ── Warm-up BEFORE any timing starts ──────────────────────
    warmup(proxy_model, cheating_model, img_paths[0], args.warmup)

    succ_num    = 0
    total_time  = 0.0
    total_query = []
    ssim_vals   = []
    log_rows    = []

    N = len(img_paths)
    verbose_all = N <= 10          # tiny demo set → print every image
    for i, img_path in enumerate(img_paths):
        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size
        fname = os.path.splitext(os.path.basename(img_path))[0]

        img_t = pil_to_tensor(img_pil)

        # ── Cloak Crafter (Universal + Refinement via proxy) ──
        adv_t, query, local_succ, t_cloak = cloak_crafter(
            img_t, proxy_model, args.n_iter, EPS, args.lr
        )
        total_time += t_cloak
        total_query.append(query)

        # ── Convert adversarial to PIL ────────────────────────
        adv_np  = (adv_t[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        adv_pil = Image.fromarray(adv_np)

        # ── Evaluate with cheating model (cloaked + original) ─
        pred_boxes  = evaluate_cheating(cheating_model, adv_pil, args.input_size)
        orig_boxes  = evaluate_cheating(cheating_model, img_pil, args.input_size)
        target_succ = 1 if len(pred_boxes) == 0 else 0
        succ_num   += target_succ

        # ── SSIM ──────────────────────────────────────────────
        ssim_val = ssim_fn(
            (img_t * 255).cpu(), (adv_t * 255).detach().cpu(),
            data_range=255, size_average=False
        ).item()
        ssim_vals.append(ssim_val)

        fps     = (i + 1) / total_time if total_time > 0 else 0
        dsr_now = succ_num / (i + 1)

        if verbose_all or (i + 1) % 50 == 0 or i == 0 or (i + 1) == N:
            print(f"  [{i+1:4d}/{N}] {fname}  Det before/after: "
                  f"{len(orig_boxes)}→{len(pred_boxes)}  "
                  f"LocalSucc:{local_succ}  TargetSucc:{target_succ}  Q:{query}  "
                  f"t:{t_cloak*1000:.0f}ms  DSR:{dsr_now:.3f}  "
                  f"SSIM:{ssim_val:.3f}  FPS:{fps:.1f}")

        # ── Save visualisation ────────────────────────────────
        gt_label = os.path.join(GT_PATH, fname + '.txt')
        gt_boxes = load_gt_boxes(gt_label, orig_w, orig_h)

        vis_orig = draw_boxes(img_pil, orig_boxes, gt_boxes)
        vis_adv  = draw_boxes(adv_pil, pred_boxes, gt_boxes)

        vis_orig.save(os.path.join(orig_dir, f'{fname}.jpg'))
        adv_pil.save(os.path.join(clean_dir, f'{fname}.png'))
        vis_adv.save(os.path.join(box_dir, f'{fname}.jpg'))
        side_by_side(vis_orig, vis_adv).save(os.path.join(cmp_dir, f'{fname}.jpg'))

        log_rows.append({
            'Index':       i,
            'File':        os.path.basename(img_path),
            'GT_boxes':    len(gt_boxes),
            'Det_before':  len(orig_boxes),
            'Det_after':   len(pred_boxes),
            'LocalSucc':   local_succ,
            'TargetSucc':  target_succ,
            'Query':       query,
            'SSIM':        round(ssim_val, 4),
            'Time_s':      round(t_cloak, 4),
        })

    # ── Per-image log ─────────────────────────────────────────
    log_xlsx = os.path.join(log_dir,
        f'{GAME}_{proxy_name}_{cheating_name}_cloak_crafter{suffix}.xlsx')
    pd.DataFrame(log_rows).to_excel(log_xlsx, index=False)
    print(f"  Per-image log → {log_xlsx}")

    # ── Aggregate metrics ─────────────────────────────────────
    final_dsr = succ_num / N
    final_fps = N / total_time if total_time > 0 else 0
    avg_query = np.mean(total_query)
    avg_ssim  = np.mean(ssim_vals)

    print(f"\n  ── Results: {cheating_name} on {GAME.upper()} ──")
    print(f"  DSR      : {final_dsr*100:.2f}%")
    print(f"  FPS      : {final_fps:.2f}")
    print(f"  Avg Query: {avg_query:.2f}")
    print(f"  Avg SSIM : {avg_ssim:.4f}")
    print(f"  Images   : {vis_dir}")

    del proxy_model, cheating_model
    torch.cuda.empty_cache()

    return {
        'Cheating Model': cheating_name,
        'Dataset':        f'AVCA-{GAME.upper()}',
        'DSR(%)':         round(final_dsr * 100, 2),
        'FPS':            round(final_fps, 2),
        '#Query':         round(avg_query, 2),
        'SSIM':           round(avg_ssim, 2),
    }


# ─── SAVE TABLE 2 FORMAT ─────────────────────────────────────
def save_table2(rows):
    """Save / update Excel and chart in Table 2 format."""
    df_new = pd.DataFrame(rows)

    if os.path.exists(EVAL_XLSX):
        df_old = pd.read_excel(EVAL_XLSX)
        for _, r in df_new.iterrows():
            mask = ((df_old['Cheating Model'] == r['Cheating Model']) &
                    (df_old['Dataset']        == r['Dataset']))
            df_old = df_old[~mask]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(EVAL_XLSX, index=False)
    print(f"\n  Table 2 Excel → {EVAL_XLSX}")

    labels  = [f"{r['Cheating Model']}\n{r['Dataset']}" for _, r in df.iterrows()]
    dsrs    = df['DSR(%)'].tolist()
    fpss    = df['FPS'].tolist()
    queries = df['#Query'].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(df) * 2), 5))

    ax = axes[0]
    bars = ax.bar(labels, dsrs, color='steelblue', width=0.55)
    for bar, val in zip(bars, dsrs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('DSR (%)')
    ax.set_ylim(0, 115)
    ax.set_title('Defense Success Rate')

    ax = axes[1]
    bars = ax.bar(labels, fpss, color='darkorange', width=0.55)
    for bar, val in zip(bars, fpss):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(fpss) * 0.02, f'{val:.1f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('FPS')
    ax.set_title('Frames Per Second')

    ax = axes[2]
    bars = ax.bar(labels, queries, color='mediumseagreen', width=0.55)
    for bar, val in zip(bars, queries):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(queries) * 0.02, f'{val:.2f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('#Query')
    ax.set_title('Average Queries')

    plt.suptitle(f'Cloak Crafter — {GAME.upper()}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(EVAL_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart → {EVAL_PNG}")


# ─── MAIN ────────────────────────────────────────────────────
def main():
    print("\n[1] Loading image paths …")
    img_paths = load_image_paths(DATA_PATH)
    print(f"  Found {len(img_paths)} images")

    all_rows = []
    for cheat_name in args.cheating_model:
        proxy_name = args.proxy_model if args.proxy_model else cheat_name
        row = run_evaluation(cheat_name, proxy_name, img_paths)
        all_rows.append(row)

    print(f"\n{'='*65}")
    print(f"  SUMMARY — Cloak Crafter on {GAME.upper()}")
    print(f"{'='*65}")
    print(pd.DataFrame(all_rows).to_string(index=False))
    print()

    save_table2(all_rows)

    print(f"\n{'='*65}")
    print(f"  ALL DONE")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()