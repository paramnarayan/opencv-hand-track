from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    image_path: Path
    model_path: Path
    camera_index: int = 0
    rotation: str = "none"
    min_gap: float = 80.0
    min_quad_area: float = 2_500.0
    image_display_fraction: float = 0.70
    feather_radius: int = 0
    one_euro_min_cutoff: float = 1.5
    one_euro_beta: float = 0.015
    inference_max_dimension: int = 640
    dropout_hold_frames: int = 4
    max_quad_jump_fraction: float = 0.35
    requested_fps: int = 60
    list_cameras: bool = False
    max_camera_index: int = 4
