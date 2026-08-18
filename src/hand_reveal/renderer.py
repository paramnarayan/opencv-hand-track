from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_overlay(image_path: Path, search_directory: Path) -> np.ndarray:
    candidates = [image_path]
    if not image_path.is_file() and search_directory.is_dir():
        candidates.extend(
            path
            for path in sorted(search_directory.iterdir())
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            and not path.name.startswith(".")
        )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        image = cv2.imread(str(candidate))
        if image is not None and image.size:
            print(
                f"[INFO] Loaded image: {candidate} "
                f"({image.shape[1]}x{image.shape[0]} px)"
            )
            return image

    print("[WARNING] No readable image found; using a generated placeholder.")
    card = np.full((1080, 1920, 3), (20, 20, 20), dtype=np.uint8)
    cv2.rectangle(card, (20, 20), (1900, 1060), (0, 180, 255), 8)
    cv2.putText(
        card,
        "PUT YOUR IMAGE IN THIS FOLDER",
        (380, 560),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (255, 255, 255),
        5,
        cv2.LINE_AA,
    )
    return card


class Projector:
    """Composites a fixed image through a hand-controlled reveal mask."""

    def __init__(
        self,
        overlay: np.ndarray,
        frame_width: int,
        frame_height: int,
        image_display_fraction: float,
        feather_radius: int,
    ):
        if not 0 < image_display_fraction <= 1:
            raise ValueError("Image display fraction must be between 0 and 1")
        if feather_radius < 0:
            raise ValueError("Feather radius cannot be negative")

        self.frame_width = frame_width
        self.frame_height = frame_height

        overlay_height, overlay_width = overlay.shape[:2]
        max_width = max(1, int(frame_width * image_display_fraction))
        max_height = max(1, int(frame_height * image_display_fraction))
        scale = min(max_width / overlay_width, max_height / overlay_height)
        display_width = max(1, round(overlay_width * scale))
        display_height = max(1, round(overlay_height * scale))

        resized = cv2.resize(
            overlay, (display_width, display_height), interpolation=cv2.INTER_AREA
        )
        center_x, center_y = frame_width // 2, frame_height // 2
        image_x0 = center_x - display_width // 2
        image_y0 = center_y - display_height // 2
        image_x1 = image_x0 + display_width
        image_y1 = image_y0 + display_height

        self.canvas_fixed = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        self.image_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        source_x0 = max(0, -image_x0)
        source_y0 = max(0, -image_y0)
        source_x1 = min(display_width, frame_width - image_x0)
        source_y1 = min(display_height, frame_height - image_y0)
        dest_x0 = max(0, image_x0)
        dest_y0 = max(0, image_y0)
        dest_x1 = min(frame_width, image_x1)
        dest_y1 = min(frame_height, image_y1)
        if source_x1 > source_x0 and source_y1 > source_y0:
            self.canvas_fixed[dest_y0:dest_y1, dest_x0:dest_x1] = resized[
                source_y0:source_y1, source_x0:source_x1
            ]
            self.image_mask[dest_y0:dest_y1, dest_x0:dest_x1] = 255

        shape = (frame_height, frame_width)
        self.quad_mask = np.zeros(shape, dtype=np.uint8)
        self.reveal_mask = np.zeros(shape, dtype=np.uint8)

        self.feather_radius = feather_radius
        if feather_radius:
            self.erode_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (feather_radius, feather_radius)
            )
            self.blur_size = feather_radius * 2 + 1
            self.blur_sigma = feather_radius * 0.5

        print(f"[INFO] Projector ready: display size {display_width}x{display_height} px")

    def render(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        self.quad_mask.fill(0)
        cv2.fillConvexPoly(self.quad_mask, quad.astype(np.int32), 255)
        cv2.bitwise_and(self.quad_mask, self.image_mask, dst=self.reveal_mask)

        if self.feather_radius:
            cv2.erode(
                self.reveal_mask,
                self.erode_kernel,
                dst=self.reveal_mask,
                iterations=1,
            )
            cv2.GaussianBlur(
                self.reveal_mask,
                (self.blur_size, self.blur_size),
                self.blur_sigma,
                dst=self.reveal_mask,
            )

        if not self.feather_radius:
            cv2.copyTo(self.canvas_fixed, self.reveal_mask, frame)
            return frame

        x, y, width, height = cv2.boundingRect(quad.astype(np.int32))
        padding = self.feather_radius * 2
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(self.frame_width, x + width + padding)
        y1 = min(self.frame_height, y + height + padding)
        mask = self.reveal_mask[y0:y1, x0:x1]
        alpha = mask.astype(np.float32)[:, :, None] * (1.0 / 255.0)
        camera = frame[y0:y1, x0:x1]
        image = self.canvas_fixed[y0:y1, x0:x1]
        image_part = np.empty_like(image, dtype=np.float32)
        camera_part = np.empty_like(camera, dtype=np.float32)
        np.multiply(image, alpha, out=image_part)
        np.multiply(camera, 1.0 - alpha, out=camera_part)
        np.add(image_part, camera_part, out=image_part)
        np.copyto(camera, image_part, casting="unsafe")
        return frame
