"""
effects/kaleidoscope.py — Mirrored kaleidoscope / symmetry effect.

The frame is divided into radial wedge slices; one wedge is mirrored and
tiled around a configurable centre point.  Face position shifts the centre;
audio can modulate the number of slices.

Performance notes:
- Uses polar-to-Cartesian remapping with cv2.remap (runs in C++).
- A pair of remap tables is pre-computed once per resolution change.
- Runs comfortably at 320×240 on Pi 3 class hardware.
"""

import math
from typing import Optional

import cv2
import numpy as np

from .base import BaseEffect
from face_detection import FaceInfo
from audio_reactivity import AudioFeatures


class KaleidoscopeEffect(BaseEffect):
    """Radial kaleidoscope / mirror symmetry effect."""

    name = "Kaleidoscope"

    def __init__(self, slices: int = 8):
        super().__init__()
        self._slices = slices
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._last_shape: Optional[tuple] = None
        self._last_slices: int = 0
        self._last_cx: float = -1.0
        self._last_cy: float = -1.0

    # ------------------------------------------------------------------

    def _rebuild_maps(self, h: int, w: int,
                      cx: float, cy: float, slices: int) -> None:
        """Pre-compute the polar-remap lookup tables for the given parameters."""
        # Pixel coordinate grids
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

        # Vector from the symmetry centre to each pixel
        dx = xs - cx
        dy = ys - cy

        # Polar coordinates
        angle = np.arctan2(dy, dx)
        radius = np.hypot(dx, dy)

        # Fold angle into one wedge and mirror
        sector_angle = 2.0 * math.pi / slices
        angle_folded = np.mod(angle, sector_angle)
        # Mirror every other fold so seams blend smoothly
        mirror_mask = angle_folded > sector_angle / 2.0
        angle_folded[mirror_mask] = sector_angle - angle_folded[mirror_mask]

        # Back to Cartesian source coordinates
        src_x = cx + radius * np.cos(angle_folded)
        src_y = cy + radius * np.sin(angle_folded)

        self._map_x = src_x.astype(np.float32)
        self._map_y = src_y.astype(np.float32)
        self._last_shape = (h, w)
        self._last_slices = slices
        self._last_cx = cx
        self._last_cy = cy

    # ------------------------------------------------------------------

    def apply(
        self,
        frame: np.ndarray,
        face: Optional[FaceInfo] = None,
        audio: Optional[AudioFeatures] = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Face drives the symmetry centre (normalised 0–1 → pixels)
        if face is not None and face.detected:
            cx = face.cx_norm * w
            cy = face.cy_norm * h
        else:
            cx = w / 2.0
            cy = h / 2.0

        # Audio can bump the slice count (bass energy adds up to 4 extra)
        bass = self._audio_value(audio, "bass")
        slices = self._slices + int(bass * 4)
        slices = max(2, slices)

        # Rebuild remap tables only when parameters change meaningfully
        rebuild = (
            self._map_x is None
            or self._last_shape != (h, w)
            or self._last_slices != slices
            or abs(self._last_cx - cx) > 2.0
            or abs(self._last_cy - cy) > 2.0
        )
        if rebuild:
            self._rebuild_maps(h, w, cx, cy, slices)

        # Apply the remap
        result = cv2.remap(
            frame,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        return result
