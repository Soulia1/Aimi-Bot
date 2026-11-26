"""
Export the trained YOLOv8n model to ONNX format for AIMi's OpenCV DNN loader.

After training completes, run:
    python training/export_model.py
"""

import shutil
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run:  pip install ultralytics")

PROJECT  = Path(__file__).resolve().parent
BEST_PT  = PROJECT / "runs" / "aimi_robrun" / "weights" / "best.pt"
MODELS   = PROJECT.parent / "models"

# Backup originals
BACKUP = MODELS / "backup"
BACKUP.mkdir(exist_ok=True)

def main():
    if not BEST_PT.exists():
        sys.exit(f"Best weights not found at {BEST_PT}. Train first!")

    print("Loading best model …")
    model = YOLO(str(BEST_PT))

    # Export to ONNX
    print("Exporting to ONNX …")
    onnx_path = model.export(format="onnx", imgsz=416, simplify=True)
    print(f"   ONNX saved: {onnx_path}")

    # Backup old weights
    for f in ("yolov3-tiny.weights", "yolov3-tiny.cfg", "coco-dataset.labels"):
        src = MODELS / f
        if src.exists() and not (BACKUP / f).exists():
            shutil.copy2(str(src), str(BACKUP / f))
            print(f"   Backed up {f}")

    # Copy PyTorch (.pt) and ONNX to models/
    dst_pt = MODELS / "aimi-custom.pt"
    shutil.copy2(str(BEST_PT), str(dst_pt))
    print(f"   Copied to {dst_pt}")

    dst_onnx = MODELS / "aimi-custom.onnx"
    shutil.copy2(str(onnx_path), str(dst_onnx))
    print(f"   Copied to {dst_onnx}")

    # Update labels file
    labels_path = MODELS / "coco-dataset.labels"
    labels_path.write_text("person\nhead\n")
    print(f"   Updated {labels_path}")

    print(f"\n✅ Export complete!")
    print(f"   ONNX model : {dst_onnx}")
    print(f"   Backup     : {BACKUP}")
    print(f"\n   The detect.py has been updated to load the ONNX model.")

if __name__ == "__main__":
    main()

