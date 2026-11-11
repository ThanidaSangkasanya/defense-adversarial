"""
Shared configuration — matches the ACTUAL get_cloak.py in this repo.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GAME_FOLDER = {"cs2": "CS2", "cf": "CF", "valorant": "Valorant", "overwatch": "Overwatch"}

# ── Training data (frames + GT labels you already have) ──
DATASETS = {
    # "cs2": r"C:\Users\tsang\Invisibility-Cloak_Project\result\evaluation\cs2\yolov5n",
    # "cf":  r"C:\Users\tsang\Invisibility-Cloak_Project\result\evaluation\cf\yolov5n",
    # "cs2": r"C:\Users\tsang\Invisibility-Cloak_Project\result\refinement\cs2\yolov5n\attack_clean",
    # "cf":  r"C:\Users\tsang\Invisibility-Cloak_Project\result\refinement\cf\yolov5n\attack_clean",
    "cs2": r"C:\Users\tsang\Invisibility-Cloak_Project\result\visualization\cs2\proxy_yolov5m_cheat_yolov5m\attack_clean",
    "cf":  r"C:\Users\tsang\Invisibility-Cloak_Project\result\visualization\cf\proxy_yolov5m_cheat_yolov5m\attack_clean",
}
LABELS_DIR = {
    "cs2": r"C:\Users\tsang\Dataset\dataset_demo\cs2\labels",
    "cf":  r"C:\Users\tsang\Dataset\dataset_demo\cf\labels",
}

# ── Cloaked images: get_cloak.py saves them here (attack_clean = no boxes drawn) ──
# Pattern: result/evaluation/{game}/{model}/attack_clean{suffix}/{fname}_attack.jpg


#อย่าลืมเอากลับมา
# def cloaked_dir(game, model="yolov5n", tag=""):
#     suffix = f"_{tag}" if tag else ""
#     return os.path.join(REPO_ROOT, "result", "evaluation", game, model,
#                         f"attack_clean{suffix}")

# CLOAKED_DIR = {
#     "cs2": cloaked_dir("cs2"),
#     "cf":  cloaked_dir("cf"),
# }
CLOAKED_DIR = {
    # "cs2": r"C:\Users\tsang\Invisibility-Cloak_Project\result\refinement\cs2\yolov5n\attack_clean",
    # "cf":  r"C:\Users\tsang\Invisibility-Cloak_Project\result\refinement\cf\yolov5n\attack_clean",
    "cs2": r"C:\Users\tsang\Invisibility-Cloak_Project\result\visualization\cs2\proxy_yolov5m_cheat_yolov5m\attack_clean",
    "cf":  r"C:\Users\tsang\Invisibility-Cloak_Project\result\visualization\cf\proxy_yolov5m_cheat_yolov5m\attack_clean",
}

# ── Pretrained weights (same path get_cloak.py / train_cloak.py use) ──
PRETRAINED_YOLOV5N = os.path.join(REPO_ROOT, "pretrained_models", "yolov5n.pt")

# ── Output for fine-tuned / AT models (adaptive_attack — ของเก่า ไม่แตะ) ──
FINETUNED_DIR = os.path.join(REPO_ROOT, "adaptive_attack", "finetuned_models")

# ── Output folder for adaptive_original — ผลลัพธ์ทั้งหมดอยู่ที่นี่ ──
ADAPTIVE_ORIGINAL_DIR = r"C:\Users\tsang\Invisibility-Cloak_Project\adaptive_original"

ATTACK_MODELS = ["yolov5n"]

# ── Hyperparameters ──
IMG_SIZE = 416
CONF_THRESHOLD = 0.4
EPSILON_VALUES = [8, 16]

FINETUNE_EPOCHS = 15
FINETUNE_LR = 5e-5
FINETUNE_BATCH_SIZE = 16

AT_EPOCHS = 10
AT_LR = 5e-5
AT_BATCH_SIZE = 16
AT_RATIO = 0.2

NUM_CLASSES = 1
CLASS_NAMES = ["person"]
DEVICE = "cuda:0"

# ──────────────────────────────────────────────
# NanoDet-Plus specific config
# ──────────────────────────────────────────────
NANODET_ROOT = r"C:\Users\tsang\nanodet"
NANODET_CONFIG_PATH = os.path.join(NANODET_ROOT, "config", "nanodet-plus-m_320.yml")
NANODET_WEIGHTS_PATH = os.path.join(NANODET_ROOT, "nanodet", "nanodet-plus-m_320.pth")

NANODET_IMG_SIZE = 320

def nanodet_cloaked_dir(game, tag=""):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(REPO_ROOT, "result", "evaluation", game, "nanodet",
                        f"attack_clean{suffix}")

NANODET_CLOAKED_DIR = {
    "cs2": nanodet_cloaked_dir("cs2"),
    "cf":  nanodet_cloaked_dir("cf"),
}

NANODET_FINETUNE_LR = 3e-5
NANODET_FINETUNE_EPOCHS = 10
NANODET_BATCH_SIZE = 8