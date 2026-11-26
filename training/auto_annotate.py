"""
Auto-annotate 1000 ROB RUN training frames using Ultralytics YOLOv8n (COCO).

For each frame we:
  1. Run YOLOv8n COCO inference to find "person" detections (COCO class 0).
  2. For every person box, derive a "head" box from the top ~20% of the body.
  3. Write YOLO-format .txt labels:
       class_id  center_x  center_y  width  height   (all normalised 0-1)
     where class 0 = person, class 1 = head.

Output layout (Ultralytics compatible):
  training/dataset/images/train/   (80% of frames, split by clip)
  training/dataset/images/val/     (20% of frames)
  training/dataset/labels/train/
  training/dataset/labels/val/
"""

import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Install ultralytics first:  pip install ultralytics")

# ── paths ──────────────────────────────────────────────────────────────
PROJECT   = Path(__file__).resolve().parent.parent          # AIMi-main
FRAMES    = Path.home() / "Downloads" / "ROB RUN Training 1000"
DATASET   = PROJECT / "training" / "dataset"

CONFIDENCE = 0.25          # lower than runtime → catch more persons
HEAD_RATIO = 0.20          # top 20% of person box = head
MIN_HEAD_PX = 8            # skip tiny heads
VAL_RATIO  = 0.20          # fraction of clips held out for validation

# ── load pre-trained YOLOv8n (auto-downloads ~6MB) ─────────────────────
print("[1/4] Loading YOLOv8n (COCO) for auto-annotation …")
model = YOLO("yolov8n.pt")

# ── gather images + clip IDs from manifest ─────────────────────────────
print("[2/4] Reading manifest …")
manifest = {}
with open(FRAMES / "manifest.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        manifest[row["image"]] = row["clip_id"]

# group by clip for train/val split
clips = defaultdict(list)
for rel, clip_id in manifest.items():
    full = FRAMES / rel.replace("/", os.sep)
    if full.exists():
        clips[clip_id].append((rel, full))

clip_names = sorted(clips.keys())
n_val = max(1, int(len(clip_names) * VAL_RATIO))
val_clips = set(clip_names[-n_val:])          # last 20% alphabetically
train_clips = set(clip_names) - val_clips

print(f"   {len(clip_names)} clips total → {len(train_clips)} train, {len(val_clips)} val")

# ── prepare output dirs ────────────────────────────────────────────────
for split in ("train", "val"):
    (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

# ── detect + annotate ──────────────────────────────────────────────────
print("[3/4] Running detection on 1000 frames …")
stats = {"frames": 0, "persons": 0, "heads": 0, "empty": 0}

for clip_id in clip_names:
    split = "val" if clip_id in val_clips else "train"
    for rel, full in clips[clip_id]:
        # Run inference — only detect class 0 (person) from COCO
        results = model.predict(
            source=str(full),
            conf=CONFIDENCE,
            classes=[0],          # person only
            verbose=False,
            device=0,             # GPU is much faster for 1000 images
        )

        result = results[0]
        H, W = result.orig_shape
        labels = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # xyxy format
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                bw = x2 - x1
                bh = y2 - y1

                # --- person box (class 0) ---
                pcx = (x1 + bw / 2) / W
                pcy = (y1 + bh / 2) / H
                pw  = bw / W
                ph  = bh / H
                labels.append(f"0 {pcx:.6f} {pcy:.6f} {pw:.6f} {ph:.6f}")
                stats["persons"] += 1

                # --- head box (class 1): top HEAD_RATIO of person ---
                head_h = bh * HEAD_RATIO
                if head_h >= MIN_HEAD_PX:
                    hcx = (x1 + bw / 2) / W
                    hcy = (y1 + head_h / 2) / H
                    hw  = (bw * 0.6) / W          # head narrower than body
                    hh  = head_h / H
                    labels.append(f"1 {hcx:.6f} {hcy:.6f} {hw:.6f} {hh:.6f}")
                    stats["heads"] += 1

        # Always copy the image (empty-label frames act as negatives)
        stem = Path(rel).stem
        dst_img = DATASET / "images" / split / f"{stem}.jpg"
        dst_lbl = DATASET / "labels" / split / f"{stem}.txt"

        shutil.copy2(str(full), str(dst_img))
        with open(dst_lbl, "w") as lf:
            lf.write("\n".join(labels))

        if not labels:
            stats["empty"] += 1
        stats["frames"] += 1

        if stats["frames"] % 100 == 0:
            print(f"  … {stats['frames']}/1000 done")

# ── summary ────────────────────────────────────────────────────────────
print(f"\n[4/4] Done!")
print(f"   Frames  : {stats['frames']}")
print(f"   Persons : {stats['persons']}")
print(f"   Heads   : {stats['heads']}")
print(f"   Empty   : {stats['empty']} (negative examples)")

train_count = len(list((DATASET / "images" / "train").glob("*.jpg")))
val_count   = len(list((DATASET / "images" / "val").glob("*.jpg")))
print(f"   Train   : {train_count} images")
print(f"   Val     : {val_count} images")
print(f"\n   Dataset written to: {DATASET}")
