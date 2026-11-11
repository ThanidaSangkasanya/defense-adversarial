"""
Verify DSR — เทียบผลของ compute_dsr (evaluation.py/utils.py) กับ get_cloak.py จริง
==============================================================================
รัน compute_dsr บน "ภาพ cloaked ทั้งหมด" (ไม่ split, ไม่ fine-tune) ด้วยโมเดล
original ล้วนๆ เพื่อแยกว่าปัญหาอยู่ที่ logic การ detect ของเรา หรืออยู่ที่
ผลกระทบจากการวัดแค่ subset (Set A 100 ภาพ)

วิธีใช้:
  1. รันสคริปต์นี้ -> จะได้ DSR จาก compute_dsr() ของเรา บนภาพทั้งหมด
  2. รัน get_cloak.py ตัวจริงสดๆ อีกครั้ง (python get_cloak.py --game cs2 --model yolov5n)
  3. เทียบตัวเลข DSR ทั้งสอง — ถ้าตรงกัน (ห่างกันไม่เกิน 1-2%) แปลว่า
     evaluation logic ของเราถูกต้องแล้ว ปัญหาอยู่ที่ผลกระทบจาก subset/fine-tune
     ถ้ายังต่างกันมาก แปลว่า logic การ detect ยังมีจุดผิดที่ต้องหาต่อ

Usage:
    python adaptive_attack/verify_dsr.py --dataset cs2
"""
import os
import sys
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_attack.config import CLOAKED_DIR, PRETRAINED_YOLOV5N, IMG_SIZE, CONF_THRESHOLD, DEVICE
from adaptive_attack.utils import load_yolov5_for_inference, compute_dsr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cs2", choices=["cs2", "cf"])
    args = parser.parse_args()

    cloaked_dir = CLOAKED_DIR[args.dataset]

    print(f"{'='*70}")
    print(f" Verify DSR — {args.dataset.upper()} (ORIGINAL model, ALL images, no split)")
    print(f" Cloaked dir: {cloaked_dir}")
    print(f" Pretrained:  {PRETRAINED_YOLOV5N}")
    print(f" IMG_SIZE (inference size): {IMG_SIZE}")
    print(f" CONF_THRESHOLD: {CONF_THRESHOLD}")
    print(f"{'='*70}")

    if not os.path.isdir(cloaked_dir):
        print(f"[ERROR] Cloaked dir not found: {cloaked_dir}")
        sys.exit(1)

    exts = ("*.png", "*.jpg", "*.jpeg")
    all_files = []
    for ext in exts:
        all_files.extend(glob.glob(os.path.join(cloaked_dir, "**", ext), recursive=True))
    print(f" Found {len(all_files)} cloaked images total\n")

    model = load_yolov5_for_inference(PRETRAINED_YOLOV5N, device=DEVICE, conf=CONF_THRESHOLD)
    print(f" model.conf    = {model.conf}")
    print(f" model.iou     = {model.iou}")
    print(f" model.classes = {model.classes}\n")

    dsr, details = compute_dsr(
        model, cloaked_dir, CONF_THRESHOLD, img_size=IMG_SIZE,
        device=DEVICE, allowed_basenames=None,  # None = ALL images, no filtering
    )

    n_total = len(details)
    n_defended = sum(1 for d in details.values() if d["defended"])
    n_bypassed = n_total - n_defended

    print(f"{'='*70}")
    print(f" RESULT (our compute_dsr, ALL {n_total} images):")
    print(f"   DSR:       {dsr*100:.2f}%")
    print(f"   Defended:  {n_defended}/{n_total}")
    print(f"   Bypassed:  {n_bypassed}/{n_total}")
    print(f"{'='*70}")
    print(f"\n เทียบกับตัวเลขที่ get_cloak.py รายงานสดๆ (รันแยกต่างหาก):")
    print(f"   python get_cloak.py --game {args.dataset} --model yolov5n")
    print(f"\n ถ้า DSR ตรงกัน (ห่างไม่เกิน 1-2%) -> evaluation logic ถูกแล้ว")
    print(f" ถ้ายังต่างมาก -> ส่งผลนี้กลับมา จะช่วยหาสาเหตุต่อ")


if __name__ == "__main__":
    main()