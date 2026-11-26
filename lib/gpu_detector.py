"""Load AIMi's detector on CUDA for live use and Studio image tests."""

import os
from pathlib import Path

import numpy as np

from lib.target_selection import Detection

DEVICE = 0
IMAGE_SIZE = 320
LIVE_IMAGE_SIZE = 256
RECOVERY_IMAGE_SIZE = 416
ROOT = Path(__file__).resolve().parents[1]


def load_detector():
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("AIMi requires CUDA-enabled PyTorch and Ultralytics.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("AIMi requires CUDA. Start it with the CUDA-enabled Python environment.")

    weights = ROOT / "models" / "aimi-custom.pt"
    if not weights.is_file():
        weights = ROOT / "training" / "runs" / "aimi_robrun" / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError("AIMi could not find the trained PyTorch checkpoint.")

    model = YOLO(str(weights))
    model.predict(
        np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        half=True,
        verbose=False,
    )
    if model.predictor.device.type != "cuda":
        raise RuntimeError("AIMi model initialization did not select CUDA.")
    return model


def detect(model, frame, confidence, iou, image_size=IMAGE_SIZE, offset=(0, 0)):
    result = model.predict(
        frame, imgsz=image_size, conf=confidence, iou=iou, classes=[0, 1],
        device=DEVICE, verbose=False,
    )[0]
    return to_detections(result, offset)


def to_detections(result, offset):
    if result.boxes is None:
        return []
    boxes = result.boxes.data.cpu().tolist()
    left, top = offset
    return [
        Detection((x1 + left, y1 + top, x2 + left, y2 + top), score, int(class_id))
        for x1, y1, x2, y2, score, class_id in boxes
    ]


class LiveDetector:
    """Keep the batch-one inference path warm at the live capture sizes."""

    def __init__(self, model, frame_size, confidence, iou):
        import torch

        self.model = model
        self.predictor = model.predictor
        self.torch = torch
        self.graphs = {}
        width, height = frame_size
        self.image_size = LIVE_IMAGE_SIZE
        self.enabled = os.environ.get("AIMI_CUDA_GRAPHS", "1") != "0"
        if self.enabled:
            try:
                with torch.inference_mode():
                    wide_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    square_frame = np.zeros((192, 192, 3), dtype=np.uint8)
                    for frame, size in ((wide_frame, self.image_size), (square_frame, RECOVERY_IMAGE_SIZE)):
                        self._predict(frame, confidence, iou, size, (0, 0))
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                self.enabled = False
                self.graphs.clear()
                print(f"[GPU] Using standard inference. CUDA Graph setup failed: {exc}")

    def detect(self, frame, confidence, iou, image_size=None, offset=(0, 0)):
        if image_size is None:
            image_size = self.image_size
        if not self.enabled:
            return detect(self.model, frame, confidence, iou, image_size, offset)
        with self.torch.inference_mode():
            return self._predict(frame, confidence, iou, image_size, offset)

    def _predict(self, frame, confidence, iou, image_size, offset):
        predictor = self.predictor
        predictor.imgsz = [image_size, image_size]
        predictor.args.conf = confidence
        predictor.args.iou = iou
        predictor.args.classes = [0, 1]
        predictor.batch = (("capture",),)
        frames = [frame]
        image = predictor.preprocess(frames)
        key = tuple(image.shape)
        if key not in self.graphs:
            self.graphs[key] = self._record_graph(image)
        source, graph, predictions, _ = self.graphs[key]
        source.copy_(image)
        graph.replay()
        result = predictor.postprocess(predictions, image, frames)[0]
        return to_detections(result, offset)

    def _record_graph(self, image):
        cuda = self.torch.cuda
        backend = self.predictor.model
        source = image.clone()
        current = cuda.current_stream(image.device)
        warmup = cuda.Stream(device=image.device)
        warmup.wait_stream(current)
        with cuda.stream(warmup):
            for _ in range(3):
                backend(source)
        current.wait_stream(warmup)

        # The detection head replaces these tensors when the input shape changes.
        # A graph still references the tensors from the shape it was recorded with.
        head = backend.model.model[-1]
        anchors = head.anchors, head.strides
        graph = cuda.CUDAGraph()
        with cuda.graph(graph):
            predictions = backend(source)
        return source, graph, predictions, anchors


class SmallTargetRecovery:
    def __init__(self):
        import cv2

        self.cv2 = cv2
        self.contrast = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        self.reset()

    def reset(self):
        self.last_scan = None

    def due(self, selector, now):
        interval = 0.035 if selector.previous is not None else 0.075
        return self.last_scan is None or now - self.last_scan >= interval

    def select(self, detect_frame, frame, selector, iou, now):
        if not self.due(selector, now):
            return None
        self.last_scan = now
        previous = selector.previous

        height, width = frame.shape[:2]
        # Zoom around the last position even when the target is off center.
        cx, cy = previous.center if previous is not None else (width / 2, height / 2)
        target_size = previous.height if previous is not None else 0
        minimum_side = 128 if 0 < target_size < 35 else 160
        side = min(width, height, max(minimum_side, round(target_size * 2.4)))
        left = min(width - side, max(0, round(cx - side / 2)))
        top = min(height - side, max(0, round(cy - side / 2)))
        crop = self.enhance(frame[top:top + side, left:left + side])
        if target_size < 35:
            blurred = self.cv2.GaussianBlur(crop, (5, 5), 1.0)
            crop = self.cv2.addWeighted(crop, 1.65, blurred, -0.65, 0)
        detections = detect_frame(
            crop, selector.inference_confidence, iou,
            image_size=RECOVERY_IMAGE_SIZE, offset=(left, top),
        )
        return selector.recover(detections, (width, height), now)

    def enhance(self, frame):
        lab = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.contrast.apply(lab[:, :, 0])
        return self.cv2.cvtColor(lab, self.cv2.COLOR_LAB2BGR)

