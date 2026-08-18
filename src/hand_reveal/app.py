from __future__ import annotations

import time

import cv2

from .config import AppConfig
from .detector import HandTracker, create_hand_tracker
from .filters import OneEuroFilter
from .geometry import build_reveal_quad, is_plausible_quad
from .renderer import Projector, load_overlay


ROTATIONS = {
    "none": None,
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def transform_frame(frame, config: AppConfig):
    rotation = ROTATIONS[config.rotation]
    if rotation is not None:
        frame = cv2.rotate(frame, rotation)
    # This camera backend supplies mirrored frames. Flip once so the displayed
    # view has normal, non-mirrored orientation.
    return cv2.flip(frame, 1)


def available_cameras(max_index: int) -> list[int]:
    available: list[int] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                available.append(index)
        finally:
            capture.release()
    return available


def _create_projector(overlay, frame, config: AppConfig) -> Projector:
    frame_height, frame_width = frame.shape[:2]
    return Projector(
        overlay,
        frame_width,
        frame_height,
        config.image_display_fraction,
        config.feather_radius,
    )


def run(config: AppConfig) -> int:
    if config.dropout_hold_frames < 0:
        raise ValueError("Dropout hold frames cannot be negative")
    if not 0 < config.max_quad_jump_fraction <= 1:
        raise ValueError("Maximum quad jump must be between 0 and 1")

    if config.list_cameras:
        cameras = available_cameras(config.max_camera_index)
        if cameras:
            print("Available camera indexes:", ", ".join(map(str, cameras)))
            return 0
        print("No cameras detected.")
        return 1

    capture = None
    tracker: HandTracker | None = None
    window_name = "Hand Projection"

    try:
        print(f"[INFO] Opening camera index {config.camera_index}...")
        capture = cv2.VideoCapture(config.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {config.camera_index}")

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FPS, config.requested_fps)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read the first frame from the camera")
        frame = transform_frame(frame, config)

        overlay = load_overlay(config.image_path, config.image_path.parent)
        projector = _create_projector(overlay, frame, config)
        tracker = create_hand_tracker(config.model_path, config.inference_max_dimension)
        print(f"[INFO] Detector: {tracker.name}")

        smoother = OneEuroFilter(
            min_cutoff=config.one_euro_min_cutoff,
            beta=config.one_euro_beta,
        )
        start_ns = time.monotonic_ns()
        last_video_timestamp_ms = -1
        fps_started = time.monotonic()
        fps_count = 0
        fps_value = 0.0
        read_failures = 0
        detection_error_last_reported = 0.0
        first_frame = True
        last_raw_quad = None
        last_smoothed_quad = None
        missed_quad_frames = 0
        print("Controls: Q or ESC quits.")

        while True:
            if not first_frame:
                ok, next_frame = capture.read()
                if not ok or next_frame is None:
                    read_failures += 1
                    if read_failures >= 30:
                        raise RuntimeError("Camera stopped returning frames")
                    continue
                read_failures = 0
                frame = transform_frame(next_frame, config)
            first_frame = False

            if frame.shape[:2] != (projector.frame_height, projector.frame_width):
                projector = _create_projector(overlay, frame, config)

            elapsed_ms = (time.monotonic_ns() - start_ns) // 1_000_000
            video_timestamp_ms = max(last_video_timestamp_ms + 1, elapsed_ms)
            last_video_timestamp_ms = video_timestamp_ms

            hands = None
            try:
                hands = tracker.detect(frame, video_timestamp_ms)
            except Exception as exc:
                now = time.monotonic()
                if now - detection_error_last_reported >= 1.0:
                    print(f"[WARNING] Hand detection failed: {exc}")
                    detection_error_last_reported = now

            quad = (
                build_reveal_quad(hands, config.min_gap, config.min_quad_area)
                if hands is not None
                else None
            )
            if quad is not None and is_plausible_quad(
                quad,
                last_raw_quad,
                projector.frame_width,
                projector.frame_height,
                config.max_quad_jump_fraction,
            ):
                last_raw_quad = quad
                last_smoothed_quad = smoother.smooth(quad, video_timestamp_ms / 1000.0)
                missed_quad_frames = 0
                frame = projector.render(frame, last_smoothed_quad)
            elif (
                last_smoothed_quad is not None
                and missed_quad_frames < config.dropout_hold_frames
            ):
                missed_quad_frames += 1
                frame = projector.render(frame, last_smoothed_quad)
            else:
                smoother.reset()
                last_raw_quad = None
                last_smoothed_quad = None
                missed_quad_frames = 0

            fps_count += 1
            now = time.monotonic()
            if now - fps_started >= 1.0:
                fps_value = fps_count / (now - fps_started)
                fps_count = 0
                fps_started = now
            cv2.putText(
                frame,
                f"FPS: {fps_value:.0f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 120),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        return 0
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
        return 0
    except (RuntimeError, ValueError, cv2.error) as exc:
        print(f"[ERROR] {exc}")
        return 1
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception as exc:
                print(f"[WARNING] Detector cleanup failed: {exc}")
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
