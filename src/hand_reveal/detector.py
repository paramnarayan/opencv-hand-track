from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2


Landmarks = list[tuple[float, float]]
HandPair = tuple[Landmarks, Landmarks]


class HandTracker(Protocol):
    name: str

    def detect(self, frame, timestamp_ms: int) -> HandPair | None: ...

    def close(self) -> None: ...


def _sort_hands_by_screen_position(hands: list[Landmarks]) -> HandPair:
    hands.sort(key=lambda landmarks: sum(point[0] for point in landmarks) / len(landmarks))
    return hands[0], hands[1]


class MediaPipeTracker:
    name = "MediaPipe HandLandmarker"

    def __init__(self, model_path: Path, inference_max_dimension: int = 640):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        self._inference_max_dimension = max(0, inference_max_dimension)
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.70,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, timestamp_ms: int) -> HandPair | None:
        frame_height, frame_width = frame.shape[:2]
        largest_dimension = max(frame_width, frame_height)
        if self._inference_max_dimension and largest_dimension > self._inference_max_dimension:
            scale = self._inference_max_dimension / largest_dimension
            inference_width = max(1, round(frame_width * scale))
            inference_height = max(1, round(frame_height * scale))
            inference_frame = cv2.resize(
                frame,
                (inference_width, inference_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            inference_frame = frame

        rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks or len(result.hand_landmarks) != 2:
            return None

        hands: list[Landmarks] = [
            [(landmark.x * frame_width, landmark.y * frame_height) for landmark in hand]
            for hand in result.hand_landmarks
        ]
        return _sort_hands_by_screen_position(hands)

    def close(self) -> None:
        self._landmarker.close()


class CvZoneTracker:
    name = "CVZone HandDetector"

    def __init__(self):
        from cvzone.HandTrackingModule import HandDetector

        self._detector = HandDetector(detectionCon=0.75, maxHands=2)

    def detect(self, frame, timestamp_ms: int) -> HandPair | None:
        del timestamp_ms
        hands, _ = self._detector.findHands(frame, draw=False, flipType=False)
        if len(hands) != 2:
            return None
        landmarks: list[Landmarks] = [
            [(float(point[0]), float(point[1])) for point in hand["lmList"]]
            for hand in hands
        ]
        return _sort_hands_by_screen_position(landmarks)

    def close(self) -> None:
        return None


def create_hand_tracker(
    model_path: Path, inference_max_dimension: int = 640
) -> HandTracker:
    errors: list[str] = []

    if model_path.is_file():
        try:
            return MediaPipeTracker(model_path, inference_max_dimension)
        except (ImportError, RuntimeError, ValueError) as exc:
            errors.append(f"MediaPipe: {exc}")
    else:
        errors.append(f"MediaPipe model not found: {model_path}")

    try:
        return CvZoneTracker()
    except (ImportError, RuntimeError, ValueError) as exc:
        errors.append(f"CVZone: {exc}")

    details = "; ".join(errors)
    raise RuntimeError(
        "No hand detector could be initialized. Install the project dependencies "
        f"and verify the model path. Details: {details}"
    )
