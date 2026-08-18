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
    parser.add_argument("--rotate", choices=("none", "cw", "ccw", "180"), default="none")
    parser.add_argument("--min-gap", type=float, default=80.0)
    parser.add_argument("--min-area", type=float, default=2_500.0)
    parser.add_argument("--image-size", type=float, default=0.80)
    parser.add_argument("--feather-radius", type=int, default=0)
    parser.add_argument("--min-cutoff", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=0.015)
    parser.add_argument(
        "--inference-size",
        type=int,
        default=640,
        help="Maximum detector input dimension; 0 uses the full camera frame.",
    )
    parser.add_argument("--dropout-hold", type=int, default=4)
    parser.add_argument("--max-quad-jump", type=float, default=0.35)
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
        rotation=args.rotate,
        min_gap=args.min_gap,
        min_quad_area=args.min_area,
        image_display_fraction=args.image_size,
        feather_radius=args.feather_radius,
        one_euro_min_cutoff=args.min_cutoff,
        one_euro_beta=args.beta,
        inference_max_dimension=args.inference_size,
        dropout_hold_frames=args.dropout_hold,
        max_quad_jump_fraction=args.max_quad_jump,
        requested_fps=args.fps,
        list_cameras=args.list_cameras,
        max_camera_index=args.max_camera_index,
    )
    return run(config)
