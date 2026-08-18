from __future__ import annotations

import cv2
import numpy as np

from .detector import HandPair


def _stable_fingertip(
    hand: list[tuple[float, float]], tip_index: int, neighbor_index: int
) -> np.ndarray:
    """Use the landmark next to a tip to reduce single-point tracking jitter."""
    tip = np.asarray(hand[tip_index], dtype=np.float32)
    neighbor = np.asarray(hand[neighbor_index], dtype=np.float32)
    return tip * 0.85 + neighbor * 0.15


def build_reveal_quad(
    hands: HandPair,
    min_gap: float,
    min_area: float,
) -> np.ndarray | None:
    screen_left, screen_right = hands
    if len(screen_left) <= 8 or len(screen_right) <= 8:
        return None

    # Blend each fingertip with the landmark immediately behind it. This keeps
    # the corner near the visible tip while suppressing endpoint flicker.
    left_index = _stable_fingertip(screen_left, 8, 7)
    right_index = _stable_fingertip(screen_right, 8, 7)
    right_thumb = _stable_fingertip(screen_right, 4, 3)
    left_thumb = _stable_fingertip(screen_left, 4, 3)
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
