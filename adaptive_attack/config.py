"""
Shared configuration — matches the ACTUAL get_cloak.py in this repo.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GAME_FOLDER = {"cs2": "CS2", "cf": "CF", "valorant": "Valorant", "overwatch": "Overwatch"}

# ── Training data (frames + GT labels you already have) ──
DATASETS = {
    "cs2": r"C:\Users\tsang\Invisibility-Cloak_Project\result\visualization\cs2\proxy_yolov5m_cheat_yolov5m\attack_clean",
    "cf":  r"C:\Users\tsang\Invisibility-Cloak_Project\result\visualization\cf\proxy_yolov5m_cheat_yolov5m\attack_clean",
}
LABELS_DIR = {
    "cs2": r"C:\Users\tsang\Dataset\dataset_demo\cs2\labels",
    "cf":  r"C:\Users\tsang\Dataset\dataset_demo\cf\labels",
}

# ── Cloaked images: get_cloak.py saves them here (attack_clean = no boxes drawn) ──
CLOAKED_DIR = {
    "cs2": r"C:\Users\tsang\Invisibility-Cloak_Project\result\visualization\cs2\proxy_yolov5m_cheat_yolov5m\attack_clean",
    "cf":  r"C:\Users\tsang\Invisibility-Cloak_Project\result\visualization\cf\proxy_yolov5m_cheat_yolov5m\attack_clean",
}

# ── Pretrained weights ──
PRETRAINED_YOLOV5N = os.path.join(REPO_ROOT, "pretrained_models", "yolov5n.pt")
PRETRAINED_YOLOV5S = os.path.join(REPO_ROOT, "pretrained_models", "yolov5s.pt")
PRETRAINED_YOLOV5M = os.path.join(REPO_ROOT, "pretrained_models", "yolov5m.pt")

PRETRAINED_WEIGHTS = {
    "yolov5n": PRETRAINED_YOLOV5N,
    "yolov5s": PRETRAINED_YOLOV5S,
    "yolov5m": PRETRAINED_YOLOV5M,
}

# ── Output for fine-tuned / AT models ──
FINETUNED_DIR = os.path.join(REPO_ROOT, "adaptive_attack", "finetuned_models")

ATTACK_MODELS = ["yolov5n", "yolov5s", "yolov5m"]

# ── Hyperparameters ──
IMG_SIZE = 416          # yolov5n input size used by get_cloak.py (MODEL_SIZE['yolov5n'])
CONF_THRESHOLD = 0.4    # matches get_cloak.py --conf default
EPSILON_VALUES = [8, 16]  # get_cloak.py --epsilon takes int directly (8 -> 8/255 internally)

FINETUNE_EPOCHS = 15
FINETUNE_LR = 5e-5   # much lower — 1e-3 caused confidence collapse (spam detections)
FINETUNE_BATCH_SIZE = 16

AT_EPOCHS = 10
AT_LR = 5e-5
AT_BATCH_SIZE = 16
AT_RATIO = 0.2

NUM_CLASSES = 1
CLASS_NAMES = ["person"]
DEVICE = "cuda:0"

# ──────────────────────────────────────────────
# NanoDet-Plus specific config (appended — does not affect YOLOv5 config above)
# ──────────────────────────────────────────────
NANODET_ROOT = r"C:\Users\tsang\nanodet"   # matches get_cloak.py's load_nanodet() root
NANODET_CONFIG_PATH = os.path.join(NANODET_ROOT, "config", "nanodet-plus-m_320.yml")
NANODET_WEIGHTS_PATH = os.path.join(NANODET_ROOT, "nanodet", "nanodet-plus-m_320.pth")

NANODET_IMG_SIZE = 320   # MODEL_SIZE['nanodet'] in get_cloak.py

def nanodet_cloaked_dir(game, tag=""):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(REPO_ROOT, "result", "evaluation", game, "nanodet",
                        f"attack_clean{suffix}")

NANODET_CLOAKED_DIR = {
    "cs2": nanodet_cloaked_dir("cs2"),
    "cf":  nanodet_cloaked_dir("cf"),
}

# Fine-tuning hyperparameters for NanoDet (start conservative like YOLOv5's
# lowered LR that avoided confidence collapse)
NANODET_FINETUNE_LR = 3e-5
NANODET_FINETUNE_EPOCHS = 10
NANODET_BATCH_SIZE = 8   # NanoDet forward_train can be more memory-hungry per sample