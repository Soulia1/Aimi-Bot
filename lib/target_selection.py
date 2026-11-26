from dataclasses import dataclass
from math import hypot, tau


DEFAULT_CONFIDENCE = 0.36
MIN_CONTINUATION_CONFIDENCE = 0.20
HEAD_CONFIDENCE = 0.30
TRACK_MEMORY_SECONDS = 0.22


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float
    class_id: int

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def height(self):
        return self.box[3] - self.box[1]

    def contains_head(self, head):
        x, y = head.center
        x1, y1, x2, _ = self.box
        return x1 <= x <= x2 and y1 - 0.10 * self.height <= y <= y1 + 0.40 * self.height

    def matches(self, other):
        return (
            0.55 <= self.height / other.height <= 1.8
            and hypot(self.center[0] - other.center[0], self.center[1] - other.center[1])
            <= max(6, 0.45 * max(self.height, other.height))
        )

    def is_foreground_avatar(self, frame_size):
        width, height = frame_size
        x1, _, x2, y2 = self.box
        # The player's shoulder can shift toward center as the camera leaves aim mode.
        return y2 >= height - 5 and x2 < 0.52 * width and x2 - x1 >= 0.22 * height


@dataclass(frozen=True)
class Selection:
    body: Detection
    head: Detection | None

    @property
    def point(self):
        if self.head is not None:
            return self.head.center
        return self.body.center[0], self.body.box[1] + 0.28 * self.body.height


class TargetSelector:
    def __init__(self, confidence):
        self.confidence = confidence
        self.continuation_confidence = min(confidence, max(MIN_CONTINUATION_CONFIDENCE, confidence * 0.65))
        self.reset()

    @property
    def inference_confidence(self):
        return min(self.continuation_confidence, HEAD_CONFIDENCE)

    def reset(self):
        self.previous = None
        self.seen_at = None
        self.pending = None
        self.velocity = (0.0, 0.0)
        self.aim_offset = None
        self.aim_point = None
        self.head_sample = None
        self.head_velocity = (0.0, 0.0)

    def remember(self, selected, now):
        x, y = selected.body.center
        aim_x, aim_y = selected.point
        height = selected.body.height
        offset = ((aim_x - x) / height, (aim_y - y) / height)
        if selected.head is not None:
            sample = offset
            if self.head_sample is not None and now > self.head_sample[0]:
                sampled_at, previous = self.head_sample
                elapsed = now - sampled_at
                derivative_blend = (tau * 4 * elapsed) / (1 + tau * 4 * elapsed)
                self.head_velocity = tuple(
                    speed + derivative_blend * ((value - last) * height / elapsed - speed)
                    for speed, value, last in zip(self.head_velocity, sample, previous)
                )
            self.head_sample = now, sample
            if self.aim_offset is not None:
                elapsed = max(0, now - self.seen_at)
                cutoff = 4 + 0.09 * min(400.0, hypot(*self.head_velocity))
                blend = (tau * cutoff * elapsed) / (1 + tau * cutoff * elapsed)
                offset = tuple(old + blend * (new - old) for old, new in zip(self.aim_offset, sample))
        elif self.aim_offset is not None:
            offset = self.aim_offset
        self.aim_offset = offset
        self.aim_point = (x + offset[0] * height, y + offset[1] * height)
        if self.previous is not None and now > self.seen_at:
            elapsed = now - self.seen_at
            self.velocity = tuple(
                (current - previous) / elapsed
                for current, previous in zip(selected.body.center, self.previous.center)
            )
        self.previous = selected.body
        self.pending = None
        self.seen_at = now
        return selected

    def continue_lock(self, detections, frame_size, now):
        if now - self.seen_at > TRACK_MEMORY_SECONDS:
            return None
        height = self.previous.height
        horizon = min(now - self.seen_at, 0.10)
        px, py = self.previous.center
        vx, vy = self.velocity
        px += max(-height, min(height, vx * horizon))
        py += max(-height, min(height, vy * horizon))
        heads = [d for d in detections if d.class_id == 1 and d.confidence >= HEAD_CONFIDENCE]
        candidates = []
        for body in detections:
            if (body.class_id != 0 or body.confidence < self.continuation_confidence
                    or not 0.6 <= body.height / height <= 1.65
                    or body.is_foreground_avatar(frame_size)):
                continue
            x, y = body.center
            distance = hypot(x - px, y - py)
            if distance <= max(7, 0.65 * height):
                cost = distance + 0.5 * abs(body.height - height)
                candidates.append((cost, body))
        candidates.sort(key=lambda candidate: candidate[0])
        if not candidates:
            return None
        cost, body = candidates[0]
        if len(candidates) > 1 and candidates[1][0] - cost < max(3, 0.12 * height):
            return None
        head = max((h for h in heads if body.contains_head(h)), key=lambda h: h.confidence, default=None)
        return self.remember(Selection(body, head), now)

    def recover(self, detections, frame_size, now):
        if self.previous is not None:
            return self.continue_lock(detections, frame_size, now)
        width, height = frame_size
        center_x, center_y = (width - 1) / 2, (height - 1) / 2
        limit = 0.30 * min(width, height)
        heads = [d for d in detections if d.class_id == 1 and d.confidence >= HEAD_CONFIDENCE]
        choices = []
        for body in detections:
            if body.class_id != 0 or body.height <= 0 or body.is_foreground_avatar(frame_size):
                continue
            head = max((h for h in heads if body.contains_head(h)), key=lambda h: h.confidence, default=None)
            support = head.confidence if head is not None else 0.0
            if body.confidence < self.confidence or body.height > limit:
                continue
            x1, y1, x2, y2 = body.box
            x, y = body.center
            distance = hypot(x - center_x, y - center_y)
            if distance > limit / 2:
                continue
            if head is not None or (x1 <= center_x <= x2 and y1 <= center_y <= y2):
                selection = Selection(body, head)
                offset = hypot(selection.point[0] - center_x, selection.point[1] - center_y)
                choices.append(((-offset, body.confidence + support), selection))
        if not choices:
            return None
        _, selected = max(choices, key=lambda item: item[0])
        self.pending = None
        return self.remember(selected, now)

    def select(self, detections, frame_size, now):
        if self.seen_at is not None and now - self.seen_at > TRACK_MEMORY_SECONDS:
            self.reset()
        if self.previous is not None:
            return self.continue_lock(detections, frame_size, now)

        heads = [d for d in detections if d.class_id == 1 and d.confidence >= HEAD_CONFIDENCE]
        width, height = frame_size
        center_x, center_y = width / 2, height / 2
        choices = []
        pending_choices = []
        for body in detections:
            if (body.class_id != 0 or body.height <= 0 or body.confidence < self.continuation_confidence
                    or body.is_foreground_avatar(frame_size)):
                continue
            head = max((h for h in heads if body.contains_head(h)), key=lambda h: h.confidence, default=None)
            x1, y1, x2, y2 = body.box
            under_crosshair = x1 <= center_x <= x2 and y1 <= center_y <= y2
            # A body cut off by a side of the crop is weaker evidence for a new lock.
            inside_sides = x1 > 1 and x2 < width - 1
            eligible = body.confidence >= self.confidence and (
                inside_sides or (under_crosshair and head is not None)
            )
            selection = Selection(body, head)
            distance = hypot(selection.point[0] - center_x, selection.point[1] - center_y)
            score = (under_crosshair, -distance, body.confidence)
            if eligible and (head is not None or y1 <= 1 or under_crosshair):
                choices.append((score, selection))
            elif eligible:
                pending_choices.append((score, selection))
                if self.pending is not None:
                    previous_body, previous_time = self.pending
                    if now - previous_time <= TRACK_MEMORY_SECONDS and body.matches(previous_body):
                        choices.append((score, selection))

        pending = max(pending_choices, key=lambda item: item[0], default=None)
        self.pending = (pending[1].body, now) if pending is not None else None
        if not choices:
            return None
        _, selected = max(choices, key=lambda item: item[0])
        return self.remember(selected, now)
