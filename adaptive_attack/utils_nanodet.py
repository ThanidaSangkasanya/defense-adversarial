"""
NanoDet-Plus support for the Adaptive Attack pipeline.
========================================================
Mirrors utils.py's YOLOv5 functions, but using NanoDet's actual API
(taken directly from get_cloak.py's load_nanodet() / detect_nanodet()):

  - Model loading: build_model(cfg.model) + load_model_weight(), NOT
    DetectMultiBackend/AutoShape.
  - Inference: requires a `meta` dict + Pipeline transform + model.inference(meta),
    NOT a simple model(img_np) call.
  - Anchor-free, distribution-regression head (GFL-style) — output format
    is completely different from YOLOv5's [B,N,5+nc] grid predictions.

Kept in a SEPARATE file from utils.py on purpose — the YOLOv5 pipeline
is verified working (DSR matches get_cloak.py exactly); this file must
not risk breaking it.
"""
import os
import sys
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_attack.config import REPO_ROOT, NANODET_ROOT, NANODET_CONFIG_PATH, NANODET_WEIGHTS_PATH

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if NANODET_ROOT not in sys.path:
    sys.path.insert(0, NANODET_ROOT)


# ──────────────────────────────────────────────
# Model loading — mirrors get_cloak.py's load_nanodet()
# ──────────────────────────────────────────────

def load_nanodet_for_inference(device="cuda:0"):
    """
    Loads NanoDet-Plus exactly like get_cloak.py's load_nanodet().
    Returns (model, cfg, pipeline) — all three are needed for inference.
    """
    from nanodet.util import cfg, load_config, Logger as NL
    from nanodet.model.arch import build_model
    from nanodet.util import load_model_weight
    from nanodet.data.transform import Pipeline

    load_config(cfg, NANODET_CONFIG_PATH)
    model = build_model(cfg.model)
    ckpt = torch.load(NANODET_WEIGHTS_PATH, map_location="cpu")
    load_model_weight(model, ckpt, NL(0, use_tensorboard=False))
    model = model.to(device).eval()
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    return model, cfg, pipeline


def load_nanodet_for_training(device="cuda:0"):
    """
    Loads the SAME underlying model/weights as load_nanodet_for_inference,
    but builds a separate TRAIN-time pipeline (cfg.data.train.pipeline),
    not the val pipeline. Reusing the val pipeline for training was a bug
    in an earlier version — val pipeline has no train-time augmentation
    and is built against cfg.data.val settings, mismatched with
    cfg.data.train.input_size used during fine-tuning.

    Returns (model, cfg, train_pipeline).
    Use load_nanodet_for_inference's own val_pipeline (a FRESH one, not
    this train_pipeline) whenever evaluating a model — training and eval
    pipelines should not be mixed.
    """
    from nanodet.util import cfg, load_config, Logger as NL
    from nanodet.model.arch import build_model
    from nanodet.util import load_model_weight
    from nanodet.data.transform import Pipeline

    load_config(cfg, NANODET_CONFIG_PATH)
    model = build_model(cfg.model)
    ckpt = torch.load(NANODET_WEIGHTS_PATH, map_location="cpu")
    load_model_weight(model, ckpt, NL(0, use_tensorboard=False))
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad_(True)

    train_pipeline = Pipeline(cfg.data.train.pipeline, cfg.data.train.keep_ratio)
    return model, cfg, train_pipeline


def build_val_pipeline(cfg):
    """
    Build a fresh VAL pipeline from an already-loaded cfg — use this to
    evaluate a fine-tuned model (AFTER metrics), instead of reusing the
    train_pipeline returned by load_nanodet_for_training.
    """
    from nanodet.data.transform import Pipeline
    return Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)


def save_nanodet_checkpoint(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"  -> Saved NanoDet checkpoint: {save_path}")


def load_nanodet_checkpoint(model, path, device="cuda:0"):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


# ──────────────────────────────────────────────
# Detection — mirrors get_cloak.py's detect_nanodet()
# ──────────────────────────────────────────────

def detect_nanodet(model, nano_cfg, pipeline, image, conf_threshold=0.4, device="cuda:0"):
    """
    image: PIL.Image or numpy array (RGB, native resolution) — NOT pre-resized.
    Returns list of [x1, y1, x2, y2, conf] pixel-space boxes (class 0 = person only,
    matching get_cloak.py's convention of filtering to person class).
    """
    import cv2
    from nanodet.data.collate import naive_collate
    from nanodet.data.batch_process import stack_batch_img

    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    meta = dict(img_info={"id": 0, "file_name": None, "height": h, "width": w},
               raw_img=img_bgr, img=img_bgr)
    meta = pipeline(None, meta, nano_cfg.data.val.input_size)
    meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(device)
    meta = naive_collate([meta])
    meta["img"] = stack_batch_img(meta["img"], divisible=32)

    with torch.no_grad():
        results = model.inference(meta)

    boxes = []
    for d in results[0].get(0, []):   # class 0 = person
        if d[-1] >= conf_threshold:
            boxes.append([int(d[0]), int(d[1]), int(d[2]), int(d[3]), float(d[-1])])
    return boxes


def detect_and_extract_nanodet(model, nano_cfg, pipeline, image, conf_threshold=0.4, device="cuda:0"):
    """Same as detect_nanodet but returns YOLO-normalized [cls, xc, yc, w, h] format."""
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image
    h, w = img_np.shape[:2]

    boxes = detect_nanodet(model, nano_cfg, pipeline, image, conf_threshold, device)
    detections = []
    for x1, y1, x2, y2, conf in boxes:
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        detections.append([0, xc, yc, bw, bh])
    return detections


# ──────────────────────────────────────────────
# DSR / metrics — mirrors utils.py's compute_dsr, but for NanoDet
# ──────────────────────────────────────────────

def compute_dsr_nanodet(model, nano_cfg, pipeline, cloak_dir, conf_threshold=0.4,
                        device="cuda:0", allowed_basenames=None, strip_suffix="_attack"):
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
        img = Image.open(cp).convert("RGB")
        boxes = detect_nanodet(model, nano_cfg, pipeline, img, conf_threshold, device)
        defended = (len(boxes) == 0)
        if defended:
            success += 1
        details[os.path.basename(cp)] = {"defended": defended, "num_detections": len(boxes)}

    return success / len(cloaked_paths), details


# ──────────────────────────────────────────────
# Loss — adapted from get_cloak.py's loss_nanodet(), extended with
# IoU-matching (positive/negative split) same philosophy as YOLOv5's
# compute_yolov5_loss in utils.py.
# ──────────────────────────────────────────────
# NOTE: get_cloak.py's loss_nanodet() only does a global "push confidence
# to 0" BCE across the whole image — fine for the DEFENDER's cloak
# training (suppress everywhere), but WRONG for fine-tuning (would cause
# the same confidence-collapse/false-positive-spam problem seen with
# YOLOv5 if used to push confidence to 1 everywhere instead of 0).
# This version matches GT boxes properly before deciding targets.

def compute_nanodet_loss(model, nano_cfg, pipeline, images, gt_boxes_list,
                         device="cuda:0"):
    """
    images: list of PIL.Image (native resolution, one per sample in the batch)
    gt_boxes_list: list of GT boxes per image, each a list of
                   [x1, y1, x2, y2] pixel-space boxes

    Runs NanoDet's own forward+loss via its built-in Trainer-style API
    (nanodet models implement `forward_train(gt_meta)` that computes its
    native GFL/QFL/DFL loss using its own label assignment) — this is
    the SAFEST option because it reuses NanoDet's own correct anchor-free
    target assignment instead of us hand-rolling one (hand-rolled IoU
    matching caused real problems with YOLOv5 and NanoDet's head format
    is more complex to replicate correctly).
    """
    from nanodet.data.collate import naive_collate
    from nanodet.data.batch_process import stack_batch_img
    import cv2

    metas = []
    for img, gt_boxes in zip(images, gt_boxes_list):
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        gt_bboxes = np.array(gt_boxes, dtype=np.float32) if len(gt_boxes) > 0 \
            else np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.zeros((len(gt_boxes),), dtype=np.int64)  # class 0 = person

        meta = dict(
            img_info={"id": 0, "file_name": None, "height": h, "width": w},
            raw_img=img_bgr, img=img_bgr,
            gt_bboxes=gt_bboxes, gt_labels=gt_labels,
        )
        meta = pipeline(None, meta, nano_cfg.data.train.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(device)
        metas.append(meta)

    batch = naive_collate(metas)
    batch["img"] = stack_batch_img(batch["img"], divisible=32)

    for k in ("gt_bboxes", "gt_labels"):
        if k in batch:
            batch[k] = [torch.as_tensor(x, device=device) if not isinstance(x, torch.Tensor)
                       else x.to(device) for x in batch[k]]

    loss, loss_states = model.forward_train(batch)
    return loss