from __future__ import annotations

import cv2
import numpy as np

from .detector import HandPair


def _projected_fingertip(
    hand: list[tuple[float, float]], tip_index: int, neighbor_index: int
) -> np.ndarray:
    """Place the corner at the visible end of the finger, not inside the tip."""
    tip = np.asarray(hand[tip_index], dtype=np.float32)
    neighbor = np.asarray(hand[neighbor_index], dtype=np.float32)
    return tip + (tip - neighbor) * 0.06


def build_reveal_quad(
    hands: HandPair,
    min_gap: float,
    min_area: float,
) -> np.ndarray | None:
    screen_left, screen_right = hands
    if len(screen_left) <= 8 or len(screen_right) <= 8:
        return None

    # Use the adjacent joint to estimate the finger direction, then extend a
    # few percent beyond MediaPipe's center-of-tip landmark to reach the
    # visible endpoint seen in the camera image.
    left_index = _projected_fingertip(screen_left, 8, 7)
    right_index = _projected_fingertip(screen_right, 8, 7)
    right_thumb = _projected_fingertip(screen_right, 4, 3)
    left_thumb = _projected_fingertip(screen_left, 4, 3)
    points = np.asarray(
        [left_index, right_index, right_thumb, left_thumb], dtype=np.float32
    )

    if not np.isfinite(points).all():
        return None
    if float(np.linalg.norm(right_index - left_index)) <= min_gap:
        return None

    contour = points.astype(np.int32)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(points)) < min_area:
        return None
    return points


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
