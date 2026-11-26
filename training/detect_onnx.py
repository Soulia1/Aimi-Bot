"""
Updated detection module that loads the custom-trained ONNX model instead
of the old Darknet .weights/.cfg pair.

Replace the net loading section in lib/detect.py with the code below,
or copy this file over lib/detect.py after training.
"""

# ── USAGE SNIPPET ──────────────────────────────────────────────────────
# Replace lines 54-69 in lib/detect.py with:
#
#     # --- Custom ONNX model ---
#     onnx_path = os.path.sep.join([YOLO_DIRECTORY, "aimi-custom.onnx"])
#     if os.path.exists(onnx_path):
#         net = cv2.dnn.readNetFromONNX(onnx_path)
#         print("\\033[1;32m[AI] Loaded custom-trained ONNX model\\033[0m")
#     else:
#         weightsPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.weights"])
#         configPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.cfg"])
#         net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#         print("\\033[1;33m[AI] Using default COCO weights (no custom model found)\\033[0m")
#
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#     ln = net.getLayerNames()
#     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
#
# And update the labels file to have only:
#     person
#     head
# ───────────────────────────────────────────────────────────────────────

