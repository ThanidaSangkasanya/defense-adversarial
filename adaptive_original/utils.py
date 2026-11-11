"""
Shared utilities — REWRITTEN to match train_cloak.py's proven pattern:
  DetectMultiBackend (NO AutoShape) used directly, forward pass is
  differentiable, no need for models.yolo.Model + ComputeLoss + .hyp.

This mirrors exactly how Phase 2 trains: load_yolov5n() -> DetectMultiBackend,
model(tensor) -> raw predictions, backprop straight through.
"""
import os
import sys
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.ops import box_iou

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_attack.config import REPO_ROOT, PRETRAINED_YOLOV5N, IMG_SIZE

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ──────────────────────────────────────────────
# YOLO Label I/O
# ──────────────────────────────────────────────

def get_train_test_split(clean_frames_dir, test_ratio=0.2, seed=42):
    """
    Deterministic 80/20 split based on CLEAN frame basenames.
    The same split (by basename) is reused for cloaked images too, since
    cloaked filenames are "<basename>_attack.jpg" — matching by basename
    keeps clean/cloaked versions of the same frame on the same side of
    the split (avoids leaking a frame's cloaked twin into test while its
    clean version was in train, or vice versa).

    Returns: (train_basenames: set, test_basenames: set)
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted([f for f in os.listdir(clean_frames_dir)
                    if f.lower().endswith(exts)])
    basenames = [os.path.splitext(f)[0] for f in files]

    rng = random.Random(seed)
    shuffled = basenames.copy()
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    test_set = set(shuffled[:n_test])
    train_set = set(shuffled[n_test:])

    print(f"[Split] {len(train_set)} train / {len(test_set)} test "
          f"({len(basenames)} total, seed={seed})")
    return train_set, test_set


def save_yolo_label(filepath, detections):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for det in detections:
            cls_id, xc, yc, w, h = det
            f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def load_yolo_label(filepath):
    detections = []
    if not os.path.exists(filepath):
        return detections
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                detections.append([cls_id, xc, yc, w, h])
    return detections


# ──────────────────────────────────────────────
# Visualization: GT + prediction boxes drawn on images
# ──────────────────────────────────────────────

def find_image_path(image_dir, basename, suffix=""):
    """Locate an image file by basename (+ optional suffix like '_attack')."""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    for ext in exts:
        candidate = os.path.join(image_dir, f"{basename}{suffix}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def save_detection_visualizations(model, image_dir, label_dir, output_dir,
                                  basenames, conf_threshold=0.4,
                                  strip_suffix="", max_images=10, prefix="",
                                  img_size=416):
    """
    Runs detection on selected images and saves copies with:
      - GT boxes drawn in GREEN (ground truth, always shown)
      - Predicted boxes drawn in RED, labeled with confidence

    Useful for visually comparing before/after fine-tuning on the same
    held-out images. `model` must be an AutoShape-wrapped model (from
    load_yolov5_for_inference / wrap_raw_model_for_inference).

    Returns list of saved file paths.
    """
    from PIL import Image, ImageDraw
    import torch

    os.makedirs(output_dir, exist_ok=True)
    chosen = sorted(basenames)[:max_images]
    saved = []

    for b in chosen:
        img_path = find_image_path(image_dir, b, suffix=strip_suffix)
        if img_path is None:
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        label_path = os.path.join(label_dir, b + ".txt")
        gt_boxes = []
        for cls_id, xc, yc, bw, bh in load_yolo_label(label_path):
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            gt_boxes.append([x1, y1, x2, y2])

        with torch.no_grad():
            results = model(np.array(img), size=img_size)
        preds = results.xyxy[0].detach().cpu().numpy()

        draw = ImageDraw.Draw(img)
        # GT boxes — green, drawn first (underneath predictions)
        for (x1, y1, x2, y2) in gt_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
        # Predicted boxes — red, with confidence label
        for x1, y1, x2, y2, conf, cls_id in preds:
            if conf < conf_threshold:
                continue
            draw.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=2)
            draw.text((x1, max(0, y1 - 14)), f"{conf:.2f}", fill=(255, 40, 40))

        out_path = os.path.join(output_dir, f"{prefix}{b}.jpg")
        img.save(out_path)
        saved.append(out_path)

    return saved


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class AimbotDataset(Dataset):
    """
    image_dir: clean frames OR get_cloak.py's attack_clean/ output
      (cloaked filenames are "<original>_attack.jpg" — suffix auto-stripped)
    """
    def __init__(self, image_dir, label_dir, img_size=416,
                 subset_ratio=1.0, seed=42, strip_suffix="",
                 allowed_basenames=None):
        """
        allowed_basenames: optional set of basenames (no extension, no
        strip_suffix applied) to restrict this dataset to — used to
        enforce a train/test split so evaluation never sees training frames.
        """
        super().__init__()
        self.img_size = img_size
        self.label_dir = label_dir
        self.strip_suffix = strip_suffix
        self._image_dir = image_dir

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(
                glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
            )
        self.image_paths.sort()

        if allowed_basenames is not None:
            def _basename_of(p):
                rel = os.path.relpath(p, image_dir)
                b = os.path.splitext(rel)[0]
                if strip_suffix and b.endswith(strip_suffix):
                    b = b[: -len(strip_suffix)]
                return b
            self.image_paths = [
                p for p in self.image_paths if _basename_of(p) in allowed_basenames
            ]

        self.image_paths = [
            p for p in self.image_paths if os.path.exists(self._label_path(p))
        ]

        if subset_ratio < 1.0:
            random.seed(seed)
            n = max(1, int(len(self.image_paths) * subset_ratio))
            self.image_paths = random.sample(self.image_paths, n)
            self.image_paths.sort()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        print(f"[AimbotDataset] {len(self.image_paths)} samples "
              f"from {image_dir} (ratio={subset_ratio})")
        if len(self.image_paths) == 0:
            print(f"  [WARNING] 0 matches. Check label basenames vs image "
                  f"basenames (minus '{strip_suffix}' suffix if any).")

    def _label_path(self, image_path):
        rel = os.path.relpath(image_path, self._image_dir)
        basename = os.path.splitext(rel)[0]
        if self.strip_suffix and basename.endswith(self.strip_suffix):
            basename = basename[: -len(self.strip_suffix)]
        return os.path.join(self.label_dir, basename + ".txt")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        detections = load_yolo_label(self._label_path(img_path))
        if len(detections) > 0:
            labels = torch.tensor(detections, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        return img_tensor, labels, img_path


def collate_fn(batch):
    images, labels_list, paths = zip(*batch)
    images = torch.stack(images, dim=0)

    all_labels = []
    for i, labels in enumerate(labels_list):
        if len(labels) > 0:
            batch_idx = torch.full((len(labels), 1), i, dtype=torch.float32)
            all_labels.append(torch.cat([batch_idx, labels], dim=1))

    if all_labels:
        all_labels = torch.cat(all_labels, dim=0)
    else:
        all_labels = torch.zeros((0, 6), dtype=torch.float32)

    return images, all_labels, list(paths)


class CombinedAimbotDataset(Dataset):
    """
    Mixed dataset per your Phase 3 diagram:
      "Generate Adversarial Training Samples" = clean frames + perturbed
      (cloaked) frames, BOTH labeled with the same GT bounding boxes.

    This is what gets fine-tuned on in a SINGLE stage (not two separate
    fine-tune -> AT steps).
    """
    def __init__(self, clean_image_dir, cloaked_image_dir, label_dir,
                 img_size=416, cloaked_subset_ratio=1.0, seed=42,
                 allowed_basenames=None):
        self.clean_ds = AimbotDataset(
            clean_image_dir, label_dir, img_size=img_size,
            subset_ratio=1.0, strip_suffix="",
            allowed_basenames=allowed_basenames,
        )
        self.cloak_ds = AimbotDataset(
            cloaked_image_dir, label_dir, img_size=img_size,
            subset_ratio=cloaked_subset_ratio, seed=seed, strip_suffix="_attack",
            allowed_basenames=allowed_basenames,
        )
        n_total = len(self.clean_ds) + len(self.cloak_ds)
        print(f"[CombinedAimbotDataset] clean={len(self.clean_ds)} + "
              f"cloaked={len(self.cloak_ds)} = {n_total} total samples")

    def __len__(self):
        return len(self.clean_ds) + len(self.cloak_ds)

    def __getitem__(self, idx):
        if idx < len(self.clean_ds):
            return self.clean_ds[idx]
        return self.cloak_ds[idx - len(self.clean_ds)]


# ──────────────────────────────────────────────
# Model loading — SAME pattern as train_cloak.py / get_cloak.py
# ──────────────────────────────────────────────

def load_yolov5n_raw(weights_path=None, device="cuda:0"):
    """
    Identical to load_yolov5n() in train_cloak.py / get_cloak.py:
    DetectMultiBackend, NO AutoShape. Differentiable forward pass.
    """
    from models.common import DetectMultiBackend
    weights_path = weights_path or PRETRAINED_YOLOV5N
    model = DetectMultiBackend(weights_path, device=torch.device(device), fuse=True)
    model.eval()  # matches Phase 2 convention (BN uses running stats)
    return model


def load_finetuned_model(path, device="cuda:0"):
    """Load a model saved by save_full_model() (Step 3 / Step 4 output)."""
    model = torch.load(path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def save_full_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)
    print(f"  -> Saved model checkpoint: {save_path}")


# ──────────────────────────────────────────────
# Loss — IoU-matched objectness BCE
# ──────────────────────────────────────────────
# Rationale: DetectMultiBackend's eval-mode forward returns decoded
# predictions [B, N, 5+nc] where cols 0:4 = cx,cy,w,h (pixel space at
# model input resolution) and col 4 = objectness probability (already
# through sigmoid inside the Detect layer, same as train_cloak.py assumes).
#
# For each GT box we find the single best-IoU prediction and push its
# objectness -> 1 (this is what fine-tuning / adversarial training wants:
# "detect this target"). All other predictions are pushed towards 0 with
# a small weight, standard pos/neg imbalance handling.

def compute_yolov5_loss(model, images, targets, device="cuda:0", img_size=None):
    img_size = img_size or IMG_SIZE
    images = images.to(device)
    targets = targets.to(device)

    preds = model(images)
    raw = preds[0] if isinstance(preds, (list, tuple)) else preds  # [B,N,5+nc]

    obj = raw[..., 4]              # [B, N] probability
    boxes_pred = raw[..., :4]      # [B, N, 4] cx,cy,w,h (pixel space)

    B, N = obj.shape
    obj_target = torch.zeros_like(obj)
    pos_mask = torch.zeros_like(obj, dtype=torch.bool)

    for b in range(B):
        gt_b = targets[targets[:, 0] == b]
        if len(gt_b) == 0:
            continue

        gt_boxes = gt_b[:, 2:6].clone()          # normalized cx,cy,w,h
        gt_boxes[:, [0, 2]] *= img_size
        gt_boxes[:, [1, 3]] *= img_size

        gt_xyxy = torch.zeros_like(gt_boxes)
        gt_xyxy[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
        gt_xyxy[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
        gt_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
        gt_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

        pb = boxes_pred[b]
        pred_xyxy = torch.zeros_like(pb)
        pred_xyxy[:, 0] = pb[:, 0] - pb[:, 2] / 2
        pred_xyxy[:, 1] = pb[:, 1] - pb[:, 3] / 2
        pred_xyxy[:, 2] = pb[:, 0] + pb[:, 2] / 2
        pred_xyxy[:, 3] = pb[:, 1] + pb[:, 3] / 2

        ious = box_iou(gt_xyxy, pred_xyxy)   # [num_gt, N]
        best_idx = ious.argmax(dim=1)        # [num_gt]

        obj_target[b, best_idx] = 1.0
        pos_mask[b, best_idx] = True

    eps = 1e-7
    obj_clamped = obj.clamp(eps, 1 - eps)

    if pos_mask.any():
        pos_loss = F.binary_cross_entropy(obj_clamped[pos_mask], obj_target[pos_mask])
    else:
        pos_loss = torch.tensor(0.0, device=device)

    neg_loss = F.binary_cross_entropy(obj_clamped[~pos_mask], obj_target[~pos_mask])

    # Direct L1 sparsity penalty on negative anchors' raw objectness value
    # (NOT through BCE — BCE saturates/vanishes when values are already near 0,
    # so it fails to stop the model from raising confidence broadly across
    # thousands of anchors as a side-effect of the positive-loss gradient).
    # This term keeps pushing directly on the raw probability value.
    sparsity_loss = obj[~pos_mask].mean() if (~pos_mask).any() else torch.tensor(0.0, device=device)

    loss = pos_loss + 1.0 * neg_loss + 5.0 * sparsity_loss
    return loss


# ──────────────────────────────────────────────
# Detection / DSR (evaluation) — wraps with AutoShape for convenience
# ──────────────────────────────────────────────

def load_yolov5_for_inference(weights_path=None, device="cuda:0", conf=0.4):
    """Inference-mode loader: DetectMultiBackend + AutoShape (like get_cloak.py)."""
    from models.common import DetectMultiBackend, AutoShape
    weights_path = weights_path or PRETRAINED_YOLOV5N
    model = DetectMultiBackend(weights_path, device=torch.device(device), fuse=True)
    model = AutoShape(model)
    model.conf = conf
    model.classes = [0]
    return model


def wrap_raw_model_for_inference(raw_model, conf=0.4):
    """Wrap a raw DetectMultiBackend (e.g. loaded via load_finetuned_model) with AutoShape."""
    from models.common import AutoShape
    for p in raw_model.parameters():
        p.requires_grad_(False)   # inference only — no grad needed, avoids numpy() error
    raw_model.eval()
    model = AutoShape(raw_model)
    model.conf = conf
    model.classes = [0]
    return model


def detect_and_extract(model, image, conf_threshold=0.4, infer_size=416):
    """
    Runs detection matching get_cloak.py's convention EXACTLY:
      results = model(img_np, size=MODEL_INPUT)   # explicit size, no pre-resize

    `image` should be the NATIVE-resolution image (PIL.Image, numpy array,
    or a raw [3,H,W] tensor at native resolution) — do NOT pre-resize to
    a square before calling this. AutoShape does the letterbox resize to
    `infer_size` internally, exactly like Phase 2.
    """
    if isinstance(image, torch.Tensor):
        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    orig_conf = model.conf
    model.conf = conf_threshold
    with torch.no_grad():
        results = model(img_np, size=infer_size)   # <- matches get_cloak.py's size=MODEL_INPUT
    model.conf = orig_conf

    preds = results.xyxy[0].detach().cpu().numpy()
    h, w = img_np.shape[:2]
    detections = []
    for pred in preds:
        x1, y1, x2, y2, conf, cls_id = pred
        if conf < conf_threshold:
            continue
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        detections.append([int(cls_id), xc, yc, bw, bh])
    return detections


def train_detection_model(model, dataset, epochs=10, lr=3e-5, batch_size=16,
                          val_split=0.1, device="cuda:0", verbose=True,
                          seed=42):
    """
    Generic training loop, extracted so it works on ANY dataset
    (AimbotDataset, CombinedAimbotDataset, or a plain subset) —
    used both by step_adaptive_finetune.py and by repeated-split
    evaluation scripts that fine-tune on a fresh half of the data
    each round.

    Returns: (model, history)  — model has best-val-loss weights loaded.
    """
    from torch.utils.data import DataLoader, random_split
    import torch.optim as optim

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    for p in model.parameters():
        p.requires_grad_(True)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.937,
                          weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        epoch_loss, n_batches = 0.0, 0
        for images, targets, paths in train_loader:
            optimizer.zero_grad()
            try:
                loss = compute_yolov5_loss(model, images, targets, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            except Exception as e:
                if verbose:
                    print(f"    [WARNING] batch error: {e}")

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for images, targets, paths in val_loader:
                try:
                    loss = compute_yolov5_loss(model, images, targets, device)
                    val_loss += loss.item()
                    val_batches += 1
                except Exception:
                    continue
        avg_val = val_loss / max(val_batches, 1)

        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})
        if verbose:
            print(f"    Epoch {epoch:3d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, history


def compute_dsr(model, cloak_dir, conf_threshold=0.4, img_size=416, device="cuda:0",
                allowed_basenames=None, strip_suffix="_attack"):
    exts = ("*.png", "*.jpg", "*.jpeg")
    cloaked_paths = []
    for ext in exts:
        cloaked_paths.extend(glob.glob(os.path.join(cloak_dir, "**", ext), recursive=True))
    cloaked_paths.sort()

    if allowed_basenames is not None:
        def _basename_of(p):
            b = os.path.splitext(os.path.basename(p))[0]
            if strip_suffix and b.endswith(strip_suffix):
                b = b[: -len(strip_suffix)]
            return b
        cloaked_paths = [p for p in cloaked_paths if _basename_of(p) in allowed_basenames]

    if not cloaked_paths:
        print(f"[WARNING] No cloaked images in {cloak_dir} (after split filter)")
        return 0.0, {}

    success = 0
    details = {}
    for cp in cloaked_paths:
        img = Image.open(cp).convert("RGB")   # NATIVE resolution — no square pre-resize
        dets = detect_and_extract(model, img, conf_threshold, infer_size=img_size)
        defended = (len(dets) == 0)
        if defended:
            success += 1
        details[os.path.basename(cp)] = {"defended": defended, "num_detections": len(dets)}

    return success / len(cloaked_paths), details