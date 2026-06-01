from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisionObservation:
    width: int
    height: int
    brightness: float
    contrast: float
    dominant_color: str
    motion_level: float = 0.0

    def explain(self) -> str:
        scene = "dark" if self.brightness < 70 else "bright" if self.brightness > 180 else "moderately lit"
        motion = "no clear motion" if self.motion_level < 8 else "some motion" if self.motion_level < 25 else "strong motion"
        return (
            f"Camera sees a {self.width}x{self.height} frame that is {scene}, "
            f"mostly {self.dominant_color}, with {motion}. "
            f"Brightness={self.brightness:.1f}, contrast={self.contrast:.1f}."
        )


class VisionProcessor:
    def __init__(self, camera_index: int = 0) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for camera processing. Install opencv-python.") from exc

        self.cv2 = cv2
        self.np = np
        self.camera_index = camera_index
        self.previous_gray = None

    def observe_once(self) -> VisionObservation:
        cap = self.cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}.")
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            raise RuntimeError("Could not read a frame from the camera.")
        return self.analyze_frame(frame)

    def analyze_frame(self, frame) -> VisionObservation:
        height, width = frame.shape[:2]
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        average_bgr = frame.reshape(-1, 3).mean(axis=0)
        dominant = self._dominant_color(average_bgr)
        motion = 0.0
        if self.previous_gray is not None and self.previous_gray.shape == gray.shape:
            diff = self.cv2.absdiff(gray, self.previous_gray)
            motion = float(diff.mean())
        self.previous_gray = gray
        return VisionObservation(width, height, brightness, contrast, dominant, motion)

    @staticmethod
    def _dominant_color(average_bgr) -> str:
        blue, green, red = average_bgr
        if max(red, green, blue) - min(red, green, blue) < 18:
            return "gray/neutral"
        if red >= green and red >= blue:
            return "red/warm"
        if green >= red and green >= blue:
            return "green"
        return "blue/cool"
