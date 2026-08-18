from __future__ import annotations

import cv2
import numpy as np

from .detector import HandPair


def build_reveal_quad(
    hands: HandPair,
    min_gap: float,
    min_area: float,
) -> np.ndarray | None:
    screen_left, screen_right = hands
    if len(screen_left) <= 8 or len(screen_right) <= 8:
        return None

    left_index = np.asarray(screen_left[8], dtype=np.float32)
    right_index = np.asarray(screen_right[8], dtype=np.float32)
    right_thumb = np.asarray(screen_right[4], dtype=np.float32)
    left_thumb = np.asarray(screen_left[4], dtype=np.float32)
    points = np.asarray(
        [left_index, right_index, right_thumb, left_thumb], dtype=np.float32
    )

    if not np.isfinite(points).all():
        return None
    if float(np.linalg.norm(right_index - left_index)) <= min_gap:
        return None

    hull = cv2.convexHull(points, clockwise=False).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < min_area:
        return None

    # Keep corner correspondence stable for the temporal smoother. OpenCV may
    # choose a different first hull vertex as the hands move.
    first_corner = int(np.argmin(hull[:, 0] + hull[:, 1]))
    hull = np.roll(hull, -first_corner, axis=0)
    return hull.astype(np.float32, copy=False)


def is_plausible_quad(
    quad: np.ndarray,
    previous: np.ndarray | None,
    frame_width: int,
    frame_height: int,
    max_jump_fraction: float,
) -> bool:
    """Reject one-frame landmark explosions without blocking normal hand motion."""
    if previous is None:
        return True

    diagonal = float(np.hypot(frame_width, frame_height))
    center_jump = float(np.linalg.norm(quad.mean(axis=0) - previous.mean(axis=0)))
    if center_jump > diagonal * max_jump_fraction:
        return False

    area = abs(float(cv2.contourArea(quad)))
    previous_area = abs(float(cv2.contourArea(previous)))
    if previous_area <= 1.0:
        return True
    area_ratio = area / previous_area
    return 0.2 <= area_ratio <= 5.0
