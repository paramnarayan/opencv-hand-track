from __future__ import annotations

import argparse
from pathlib import Path

from .app import run
from .config import AppConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hand-reveal",
        description="Reveal a fixed image through a frame formed by two hands.",
    )
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "testimage.jpg")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--flip-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mirror the camera horizontally.",
    )
    parser.add_argument("--rotate", choices=("none", "cw", "ccw", "180"), default="none")
    parser.add_argument("--min-gap", type=float, default=80.0)
    parser.add_argument("--min-area", type=float, default=2_500.0)
    parser.add_argument("--image-size", type=float, default=0.35)
    parser.add_argument("--orange-strength", type=float, default=0.22)
    parser.add_argument("--feather-radius", type=int, default=5)
    parser.add_argument("--min-cutoff", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--max-camera-index", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AppConfig(
        image_path=args.image.expanduser().resolve(),
        model_path=args.model.expanduser().resolve(),
        camera_index=args.camera,
        flip_camera=args.flip_camera,
        rotation=args.rotate,
        min_gap=args.min_gap,
        min_quad_area=args.min_area,
        image_display_fraction=args.image_size,
        orange_strength=args.orange_strength,
        feather_radius=args.feather_radius,
        one_euro_min_cutoff=args.min_cutoff,
        one_euro_beta=args.beta,
        requested_fps=args.fps,
        list_cameras=args.list_cameras,
        max_camera_index=args.max_camera_index,
    )
    return run(config)
