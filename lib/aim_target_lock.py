from dataclasses import dataclass
from math import cos, exp, hypot, radians, sin


@dataclass(frozen=True)
class Target:
    x: float
    y: float
    width: float
    height: float
    head_visible: bool

    @property
    def body(self):
        return self.x + self.width / 2, self.y + self.height * 0.48

    @property
    def head(self):
        return self.x + self.width / 2, self.y + self.height * 0.16


class AimTargetLock:
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = None
        self.point = None
        self.updated_at = None
        self.awaiting_new_hit = False
        self.hit_angle = None

    def update(self, candidates, hit_angle, hit_origin, now):
        # The top of the arc indicates a shooter in front of the camera.
        if hit_angle is None or sin(hit_angle) > -0.35:
            self.reset()
            return None

        if self.hit_angle is not None and cos(hit_angle - self.hit_angle) < cos(radians(55)):
            self.reset()
        self.hit_angle = hit_angle

        if self.target is None:
            # The arc encodes bearing, so the target can be below its center.
            if self.awaiting_new_hit:
                return None
            shooter_x = hit_origin[0] + 32 * cos(hit_angle)
            eligible = [candidate for candidate in candidates if abs(candidate.body[0] - shooter_x) <= 17]
            if len(eligible) != 1:
                return None
            self.target = eligible[0]
            self.updated_at = now
            return None

        previous_x, previous_y = self.target.body
        nearby = []
        for candidate in candidates:
            x, y = candidate.body
            distance = hypot(x - previous_x, y - previous_y)
            if (distance <= max(5, self.target.height * 0.32)
                    and 0.65 <= candidate.height / self.target.height <= 1.55):
                nearby.append(candidate)

        if len(nearby) != 1:
            # Never send movement on a missed frame or transfer the lock to a crowd.
            if now - self.updated_at > 0.10:
                self.reset()
                self.awaiting_new_hit = True
                self.hit_angle = hit_angle
            return None

        self.target, = nearby
        x, body_y = self.target.body
        _, head_y = self.target.head
        head_y = head_y if self.target.head_visible else body_y
        if self.point is None:
            self.point = (x, body_y)
        elapsed = now - self.updated_at
        self.updated_at = now
        previous_x, previous_y = self.point
        follow = 1 - exp(-elapsed / 0.045)
        lift = 1 - exp(-elapsed / 0.11) if head_y < previous_y else follow
        self.point = (previous_x + follow * (x - previous_x),
                      previous_y + lift * (head_y - previous_y))
        return self.point


class AimMovement:
    def __init__(self):
        self.reset()

    def reset(self):
        self.updated_at = None
        self.remainder = (0.0, 0.0)
        self.filtered_error = (0.0, 0.0)

    def step(self, point, crosshair, pixel_scale, now):
        if self.updated_at is not None and now - self.updated_at > 0.12:
            self.reset()
        elapsed = 1 / 120 if self.updated_at is None else max(0.0, min(now - self.updated_at, 0.025))
        self.updated_at = now
        dx = (point[0] - crosshair[0]) * pixel_scale[0]
        dy = (point[1] - crosshair[1]) * pixel_scale[1]
        error = hypot(dx, dy)
        if error <= 1.5:
            self.remainder = (0.0, 0.0)
            self.filtered_error = (0.0, 0.0)
            return 0, 0

        previous_x, previous_y = self.filtered_error
        # Damp small detection changes near center and respond faster to larger corrections.
        correction = min(1.0, (error - 1.5) / 6)
        follow = 1 - exp(-elapsed / (0.016 - 0.012 * correction))
        filtered_x = previous_x + follow * (dx - previous_x)
        filtered_y = previous_y + follow * (dy - previous_y)
        # A delayed estimate must not drive the camera past the latest observation.
        filtered_x = min(max(filtered_x, min(0.0, dx)), max(0.0, dx))
        filtered_y = min(max(filtered_y, min(0.0, dy)), max(0.0, dy))
        self.filtered_error = (filtered_x, filtered_y)
        distance = hypot(filtered_x, filtered_y)
        gain = min(0.75, 1 - exp(-elapsed / 0.010), 4800 * elapsed / max(1.0, distance))
        gain *= 0.25 + 0.75 * correction
        remainder_x, remainder_y = self.remainder
        x = filtered_x * gain + (remainder_x if filtered_x * previous_x > 0 else 0.0)
        y = filtered_y * gain + (remainder_y if filtered_y * previous_y > 0 else 0.0)
        move_x, move_y = round(x), round(y)
        self.remainder = (x - move_x, y - move_y)
        return move_x, move_y
