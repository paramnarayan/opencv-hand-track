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
