"""
utils.py — Timing, scaling, and helper utilities for DogDayDiffuser.
"""

import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FPSCounter:
    """Simple exponential-moving-average FPS counter."""

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Smoothing factor (0 < alpha < 1).  Smaller = smoother.
        """
        self.alpha = alpha
        self._fps: float = 0.0
        self._last_time: float = time.perf_counter()

    def tick(self) -> float:
        """Call once per rendered frame.  Returns current smoothed FPS."""
        now = time.perf_counter()
        elapsed = now - self._last_time
        self._last_time = now

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            if self._fps == 0.0:
                self._fps = instant_fps
            else:
                self._fps = self.alpha * instant_fps + (1.0 - self.alpha) * self._fps

        return self._fps

    @property
    def fps(self) -> float:
        """Return the latest smoothed FPS value."""
        return self._fps


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize *frame* to (*width* × *height*) using fast nearest-neighbour interpolation."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)


def scale_to_fit(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Scale *frame* up to fill (*max_width* × *max_height*) while keeping aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS text in the top-left corner of *frame* (in-place, returns frame)."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_effect_name(frame: np.ndarray, name: str) -> np.ndarray:
    """Overlay current effect name in the bottom-left corner (in-place, returns frame)."""
    h = frame.shape[0]
    cv2.putText(
        frame,
        name,
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return frame


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* between *low* and *high*."""
    return max(low, min(high, value))


def smooth(current: float, target: float, alpha: float = 0.15) -> float:
    """One-pole low-pass smoother: moves *current* towards *target*."""
    return current + alpha * (target - current)
