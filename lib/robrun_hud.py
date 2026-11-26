"""Read the armed reticle and directional damage arc from ROB RUN's HUD."""

import cv2
import numpy as np


REFERENCE_WIDTH = 256
REFERENCE_HEIGHT = 144
# Coordinates are normalized against the game viewport, then offset into the capture.
CAPTURE = (64, 23, 192, 121)
CAPTURE_SIZE = (CAPTURE[2] - CAPTURE[0], CAPTURE[3] - CAPTURE[1])
HIT_ORIGIN = (128 - CAPTURE[0], 70 - CAPTURE[1])
CROSSHAIR = (141 - CAPTURE[0], 72 - CAPTURE[1])

_angles = np.deg2rad(np.arange(0, 360, 5))
_radii = np.arange(25, 40)[:, None]
_arc_x = np.rint(HIT_ORIGIN[0] + _radii * np.cos(_angles)).astype(int)
_arc_y = np.rint(HIT_ORIGIN[1] + _radii * np.sin(_angles)).astype(int)


def has_armed_reticle(normalized):
    x, y = CROSSHAIR
    # The aiming reticle is a thin, bright stroke centered vertically on the HUD.
    gray = cv2.cvtColor(normalized[y - 4:y + 4, x - 3:x + 4], cv2.COLOR_BGR2GRAY)
    sides = np.maximum(gray[:, :2].mean(axis=1), gray[:, -2:].mean(axis=1))
    for column in (2, 3, 4):
        contrast = gray[:, column] - sides
        if np.count_nonzero(contrast >= 20) >= 5 and contrast[2:6].mean() >= 18:
            return True
    return False


def incoming_hit(normalized):
    samples = normalized[_arc_y, _arc_x].astype(np.int16)
    blue, green, red = samples.transpose(2, 0, 1)
    crimson = (red >= 152) & (red - green >= 52) & (red - blue >= 30)
    # A HUD arc traces a long, thin band, unlike most isolated red effects.
    thin_arc = (crimson.sum(axis=0) >= 2) & (crimson.sum(axis=0) <= 8)
    score = sum(np.roll(thin_arc, shift) for shift in range(-4, 5))
    peak = int(np.argmax(score))
    if score[peak] < 7:
        return None
    if np.any(score[(peak + np.arange(15, 58)) % len(_angles)] >= 7):
        return None
    neighborhood = (peak + np.arange(-6, 7)) % len(_angles)
    valid = neighborhood[thin_arc[neighborhood]]
    angle = np.arctan2(np.sin(_angles[valid]).sum(), np.cos(_angles[valid]).sum())
    return float(angle)


def read_hud(frame):
    normalized = cv2.resize(frame, CAPTURE_SIZE, interpolation=cv2.INTER_AREA)
    armed = has_armed_reticle(normalized)
    return armed, incoming_hit(normalized) if armed else None
