import time
import timeit
from termcolor import colored
import cv2
import mss
import ctypes
import os
from math import hypot
from threading import Lock
import numpy as np
import pynput
from pynput.mouse import Button, Listener
from lib.aim_target_lock import AimMovement
from lib.gpu_detector import LiveDetector, SmallTargetRecovery, load_detector
from lib.hotkey import windows_hotkey
from lib.target_selection import DEFAULT_CONFIDENCE, TargetSelector

sct = mss.mss()
MONITOR = sct.monitors[1]
Wd, Hd = MONITOR["width"], MONITOR["height"]
SendInput = ctypes.windll.user32.SendInput
GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState
GetAsyncKeyState.argtypes = (ctypes.c_int,)
GetAsyncKeyState.restype = ctypes.c_short
VK_RBUTTON = 0x02
VK_F1 = 0x70
PREVIEW_NAME = "Gui Objects Detector"
PREVIEW_INTERVAL = 1 / 30
DEFAULT_CAPTURE_WIDTH = 800
PREPARED_TARGET_MAX_AGE = 2 / 60
PREPARATION_INTERVAL = 1 / 60


def capture_region(monitor, requested_width):
    width = max(2, min(requested_width, monitor["width"]))
    height = max(2, round(width * monitor["height"] / monitor["width"]))
    return {
        "left": monitor["left"] + (monitor["width"] - width) // 2,
        "top": monitor["top"] + (monitor["height"] - height) // 2,
        "width": width,
        "height": height,
    }


def target_pixel(point, frame_size):
    x, y = point
    width, height = frame_size
    return min(width - 1, max(0, round(x))), min(height - 1, max(0, round(y)))


def position(x, y):
    x = 1 + int(x * 65536./Wd)
    y = 1 + int(y * 65536./Hd)
    send_mouse_move(x, y, 0x0001 | 0x8000)


def send_mouse_move(x, y, flags=0x0001):
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, flags, 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))


def aimbot(ENABLE_AIMBOT):
    confidence = float(os.environ.get("AIMI_CONFIDENCE", DEFAULT_CONFIDENCE))
    iou = float(os.environ.get("AIMI_THRESHOLD", "0.22"))
    selector = TargetSelector(confidence)
    preview_selector = TargetSelector(confidence)
    recovery = SmallTargetRecovery()
    movement = AimMovement()
    paused = True
    input_lock = Lock()
    aim_held = bool(GetAsyncKeyState(VK_RBUTTON) & 0x8000)
    aim_session = 0
    click_at = None
    prepared = None
    first_response = None

    region = capture_region(MONITOR, int(os.environ.get("AIMI_RANGE", DEFAULT_CAPTURE_WIDTH)))
    frame_size = width, height = region["width"], region["height"]
    crosshair = ((width - 1) / 2, (height - 1) / 2)
    next_preview = next_status = 0.0
    next_preparation = 0.0
    preview_open = False

    print("[Status] Loading object detector on CUDA...")
    model = load_detector()
    detector = LiveDetector(model, frame_size, selector.inference_confidence, iou)
    mode = "CUDA Graphs" if detector.enabled else "standard inference"
    print(f"[GPU] Loaded {model.ckpt_path} on {model.predictor.device} ({mode})")
    print(f"[Status] Center capture: {width} x {height} px (inference size {detector.image_size})")
    print(colored("[AI] Aimbot mode" if ENABLE_AIMBOT else "[AI] Detector only",
                  "green" if ENABLE_AIMBOT else "red"))

    def toggle_aimbot():
        nonlocal paused, aim_session, prepared, first_response, click_at
        with input_lock:
            paused = not paused
            aim_session += 1
            prepared = first_response = None
            click_at = None
            movement.reset()

    def use_prepared(observed_at, target, pixel, now):
        nonlocal first_response
        if not 0 <= now - observed_at <= PREPARED_TARGET_MAX_AGE:
            return
        moves = movement.step(pixel, crosshair, (1.0, 1.0), now)
        if any(moves):
            send_mouse_move(*moves)
        first_response = (timeit.default_timer() - click_at, now - observed_at, target, observed_at)

    def on_click(x, y, button, pressed):
        nonlocal aim_held, aim_session, click_at, prepared, first_response
        if button == Button.x2 and pressed:
            toggle_aimbot()
        if button != Button.right or not ENABLE_AIMBOT:
            return
        now = timeit.default_timer()
        with input_lock:
            if pressed == aim_held:
                return
            aim_held = pressed
            aim_session += 1
            click_at = now if pressed and not paused else None
            first_response = None
            movement.reset()
            if click_at is not None and prepared is not None:
                observed_at, target, pixel = prepared
                use_prepared(observed_at, target, pixel, timeit.default_timer())
            prepared = None

    f1_was_down = bool(GetAsyncKeyState(VK_F1) & 0x8000)
    print("[Controls] F1 toggles AIMi. Press 0 in the preview to quit.")
    idle_status = "Ready. Hold the right mouse button to track." if ENABLE_AIMBOT else "Detecting."
    active_status = "Tracking. Release the right mouse button to stop." if ENABLE_AIMBOT else idle_status
    last_status = None
    previous_session = -1
    missed_primary = False
    frame_times = []
    capture_times = []
    detector_times = []
    first_lock_logged = False
    camera = None
    last_capture_timestamp = None
    if os.environ.get("AIMI_CAPTURE", "mss").lower() == "winrt":
        try:
            if len(sct.monitors) != 2:
                raise RuntimeError("WinRT capture currently requires a single display")
            import dxcam

            camera = dxcam.create(output_color="BGR", backend="winrt", max_buffer_len=2)
            left = region["left"] - MONITOR["left"]
            top = region["top"] - MONITOR["top"]
            camera.start(region=(left, top, left + width, top + height),
                         target_fps=120, video_mode=True)
            print("[Capture] WinRT latest-frame mode")
        except Exception as exc:
            if camera is not None:
                camera.release()
                camera = None
            print(f"[Capture] WinRT unavailable. Using MSS: {exc}")
    try:
        with windows_hotkey(VK_F1, toggle_aimbot) as (hotkey_registered, hotkey_error), Listener(on_click=on_click):
            if not hotkey_registered:
                print(f"[Controls] Windows could not register F1 (error {hotkey_error}). Using key polling.")
            while True:
                if not hotkey_registered:
                    f1_down = bool(GetAsyncKeyState(VK_F1) & 0x8000)
                    if f1_down and not f1_was_down:
                        toggle_aimbot()
                    f1_was_down = f1_down
                if preview_open and cv2.pollKey() & 0xFF == ord("0"):
                    break

                started_at = timeit.default_timer()
                with input_lock:
                    session = aim_session
                    enabled = not paused
                    tracking = enabled and (not ENABLE_AIMBOT or aim_held)
                    pressed_at = click_at
                    response = first_response
                status = "Paused. Press F1 to enable." if not enabled else active_status if tracking else idle_status
                if status != last_status:
                    print(f"[Status] {status}")
                    last_status = status
                    next_status = started_at
                if session != previous_session:
                    previous_session = session
                    next_preparation = 0.0
                    selector.reset()
                    recovery.reset()
                    missed_primary = False
                    frame_times.clear()
                    capture_times.clear()
                    detector_times.clear()
                    first_lock_logged = False
                    if tracking and response is not None:
                        delay, age, target, observed_at = response
                        selector.remember(target, observed_at)
                        print(f"[Timing] Click to first lock: {delay * 1000:.2f} ms "
                              f"(prepared detection age {age * 1000:.1f} ms)")
                        first_lock_logged = True

                if not enabled and started_at < next_preview:
                    time.sleep(0.01)
                    continue
                if enabled and not tracking and started_at < next_preparation:
                    time.sleep(0.001)
                    continue

                if camera is None:
                    frame = cv2.cvtColor(np.asarray(sct.grab(region)), cv2.COLOR_BGRA2BGR)
                    observed_at = started_at
                else:
                    sample = camera.get_latest_frame(with_timestamp=True)
                    if sample is None:
                        time.sleep(0.001)
                        continue
                    frame, observed_at = sample
                    if last_capture_timestamp is not None and observed_at <= last_capture_timestamp:
                        continue
                    last_capture_timestamp = observed_at
                captured_at = timeit.default_timer()
                observed_at = min(observed_at, captured_at)
                selected = None
                candidate = None
                aim_point = None
                detections = []
                if tracking:
                    if missed_primary and recovery.due(selector, observed_at):
                        missed_primary = False
                        selected = recovery.select(detector.detect, frame, selector, iou, observed_at)
                        detected_at = timeit.default_timer()
                    else:
                        detections = detector.detect(frame, selector.inference_confidence, iou)
                        detected_at = timeit.default_timer()
                        selected = selector.select(detections, frame_size, observed_at)
                        missed_primary = selected is None
                    if selected is not None:
                        aim_point = target_pixel(selector.aim_point, frame_size)
                        with input_lock:
                            if session != aim_session:
                                continue
                            if ENABLE_AIMBOT:
                                now = timeit.default_timer()
                                if now - observed_at > PREPARED_TARGET_MAX_AGE:
                                    continue
                                moves = movement.step(aim_point, crosshair, (1.0, 1.0), now)
                                if any(moves):
                                    send_mouse_move(*moves)
                elif enabled:
                    next_preparation = started_at + PREPARATION_INTERVAL
                    detections = detector.detect(frame, selector.inference_confidence, iou)
                    preview_selector.reset()
                    candidate = preview_selector.select(detections, frame_size, observed_at)
                    if candidate is not None:
                        x1, y1, x2, y2 = candidate.body.box
                        cx, cy = crosshair
                        on_target = x1 <= cx <= x2 and y1 <= cy <= y2
                        if not on_target and hypot(candidate.point[0] - cx, candidate.point[1] - cy) > 32:
                            candidate = None
                    if candidate is not None:
                        aim_point = target_pixel(candidate.point, frame_size)

                processed_at = timeit.default_timer()
                if tracking and selected is not None and not first_lock_logged:
                    if pressed_at is not None:
                        print(f"[Timing] Click to first lock: {(processed_at - pressed_at) * 1000:.1f} ms "
                              "(fresh inference)")
                    first_lock_logged = True
                if (enabled and not tracking) or processed_at >= next_preview:
                    if not preview_open:
                        open_preview(region)
                        preview_open = True
                    draw_preview(frame, () if ENABLE_AIMBOT else detections, aim_point, confidence)
                    if enabled and not tracking:
                        with input_lock:
                            if session == aim_session:
                                prepared = (observed_at, candidate, aim_point) if candidate is not None else None
                    next_preview = processed_at + (PREVIEW_INTERVAL if tracking else 1 / 15)

                if tracking:
                    frame_times.append((processed_at - observed_at) * 1000)
                    capture_times.append((captured_at - started_at) * 1000)
                    detector_times.append((detected_at - captured_at) * 1000)
                    if processed_at >= next_status and len(frame_times) >= 10:
                        ordered = sorted(frame_times)
                        median_ms = ordered[len(ordered) // 2]
                        p95_ms = ordered[(95 * len(ordered) - 1) // 100]
                        capture_p95 = sorted(capture_times)[(95 * len(ordered) - 1) // 100]
                        detector_p95 = sorted(detector_times)[(95 * len(ordered) - 1) // 100]
                        print(f"[Timing] Capture to aim: median {median_ms:.1f} ms, p95 {p95_ms:.1f} ms "
                              f"({len(ordered)} frames). Capture p95 {capture_p95:.1f} ms, "
                              f"detection and recovery p95 {detector_p95:.1f} ms")
                        frame_times.clear()
                        capture_times.clear()
                        detector_times.clear()
                        next_status = processed_at + 1.0
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.release()
        sct.close()
        cv2.destroyAllWindows()
        print("\n[Exit] AIMi closed")


def open_preview(region):
    width, height = region["width"], region["height"]
    right = region["left"] + width
    available_width = MONITOR["left"] + Wd - right - 32
    preview_width = min(560, max(1, available_width))
    preview_height = min(Hd - 100, round(preview_width * height / width))
    cv2.namedWindow(PREVIEW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(PREVIEW_NAME, preview_width, max(1, preview_height))
    cv2.moveWindow(PREVIEW_NAME, min(right + 12, MONITOR["left"] + Wd - preview_width - 20),
                   MONITOR["top"] + 48)


def draw_preview(frame, detections, aim_point, confidence):
    for detection in detections:
        if detection.confidence < confidence:
            continue
        x1, y1, x2, y2 = map(int, detection.box)
        is_head = detection.class_id == 1
        color = (0, 165, 255) if is_head else (0, 255, 0)
        label = "Head" if is_head else "Person"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {detection.confidence:.0%}"
        cv2.putText(frame, caption, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if aim_point is not None:
        aim_x, aim_y = aim_point
        frame[aim_y, aim_x] = (0, 0, 255)
    cv2.imshow(PREVIEW_NAME, frame)
