"""
Export the trained YOLOv3-tiny model to ONNX format, then generate 
a Darknet-compatible .cfg + .weights pair for AIMi's OpenCV DNN loader.

After training completes, run:
    python training/export_darknet.py
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

    # Export to ONNX (OpenCV can load ONNX directly)
    print("Exporting to ONNX …")
    onnx_path = model.export(format="onnx", imgsz=416, simplify=True)
    print(f"   ONNX saved: {onnx_path}")

    # Backup old weights
    for f in ("yolov3-tiny.weights", "yolov3-tiny.cfg", "coco-dataset.labels"):
        src = MODELS / f
        if src.exists():
            shutil.copy2(str(src), str(BACKUP / f))
            print(f"   Backed up {f}")

    # Copy ONNX to models/
    dst_onnx = MODELS / "aimi-custom.onnx"
    shutil.copy2(str(onnx_path), str(dst_onnx))
    print(f"   Copied to {dst_onnx}")

    # Update labels file
    labels_path = MODELS / "coco-dataset.labels"
    labels_path.write_text("person\nhead\n")
    print(f"   Updated {labels_path}")

    print(f"\n✅ Export complete!")
    print(f"   ONNX model: {dst_onnx}")
    print(f"   Old weights backed up to: {BACKUP}")
    print(f"\n   To use with AIMi, update detect.py to load the ONNX model.")
    print(f"   (The detect_onnx.py file has the updated loader code.)")

if __name__ == "__main__":
    main()

