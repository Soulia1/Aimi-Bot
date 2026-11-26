"""
Train YOLOv8n on the auto-annotated ROB RUN dataset for AIMi.

We use YOLOv8n (nano) instead of YOLOv3-tiny because:
  - YOLOv8n is faster and more accurate
  - It exports to ONNX which OpenCV 5.0 can load
  - It's the same nano-size class (~6MB) suitable for real-time

Usage:
    python training/train.py

After training, run:
    python training/export_model.py
to export and integrate the model into AIMi.
"""

import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run:  pip install ultralytics")

PROJECT = Path(__file__).resolve().parent          # training/
DATASET_YAML = PROJECT / "dataset.yaml"
OUTPUT  = PROJECT / "runs"

# ── Training hyperparameters ───────────────────────────────────────────
EPOCHS     = 100          # usually converges around 50-80 for small datasets
BATCH      = 48           # high GPU utilization in 6GB VRAM for 416×416
IMG_SIZE   = 416          # match AIMi's detection input size
PATIENCE   = 20           # early stop if no improvement for 20 epochs

def main():
    print("=" * 60)
    print("  AIMi Custom Model Training (YOLOv8n)")
    print("=" * 60)
    print(f"  Dataset  : {DATASET_YAML}")
    print(f"  Epochs   : {EPOCHS}")
    print(f"  Batch    : {BATCH}")
    print(f"  ImgSize  : {IMG_SIZE}")
    print(f"  Output   : {OUTPUT}")
    print("=" * 60)

    # Start from pre-trained YOLOv8n (COCO) and fine-tune
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        project=str(OUTPUT),
        name="aimi_robrun",
        device=0,                   # GPU 0 (RTX 3060)
        workers=8,
        cache="disk",
        exist_ok=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        # Save
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

    best_pt = OUTPUT / "aimi_robrun" / "weights" / "best.pt"
    print(f"\n✅ Training complete!")
    print(f"   Best model: {best_pt}")
    print(f"\n   Next step: run  python training/export_model.py")
    print(f"   to convert to ONNX for AIMi.")

if __name__ == "__main__":
    main()
