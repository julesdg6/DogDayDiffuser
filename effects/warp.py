"""
effects/warp.py — Face-centred local warp / swirl effect.

Distorts the area around the detected face using a bulge or swirl
displacement field computed with NumPy / OpenCV remap.

Modes:
  "bulge"  — radial outward / inward bulge (magnifying-lens feel)
  "swirl"  — angular twist that increases towards the face centre
  "ripple" — concentric sine-wave ripple outward from face centre

Performance notes:
- Displacement maps are recomputed only when face position moves noticeably.
- At 320×240 this runs fast enough on Pi 3.
- Audio (bass) boosts the warp amount for extra drama on beats.
"""

import math
from typing import Optional

import cv2
import numpy as np

from .base import BaseEffect
from face_detection import FaceInfo
from audio_reactivity import AudioFeatures

# How many pixels the face centre must move before maps are rebuilt
_REBUILD_THRESHOLD = 4.0


class WarpEffect(BaseEffect):
    """Face-centred local warp / swirl distortion."""

    name = "Warp"

    def __init__(self, mode: str = "swirl"):
        super().__init__()
        self.mode = mode          # "bulge" | "swirl" | "ripple"
        self.warp_amount = 0.6

        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._last_params: Optional[tuple] = None

    # ------------------------------------------------------------------

    def _rebuild_maps(self, h: int, w: int,
                      cx: float, cy: float,
                      radius: float, strength: float) -> None:
        """Compute displacement maps for the current face/warp parameters."""
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xs - cx
        dy = ys - cy
        dist = np.hypot(dx, dy)
        norm_dist = dist / max(radius, 1.0)

        # Smooth falloff mask — only distort within the face radius
        mask = np.clip(1.0 - norm_dist, 0.0, 1.0) ** 2

        if self.mode == "bulge":
            # Bulge: push pixels outward proportional to mask × strength
            offset_x = dx * mask * strength
            offset_y = dy * mask * strength
            src_x = xs - offset_x
            src_y = ys - offset_y

        elif self.mode == "swirl":
            # Swirl: rotate each pixel by angle proportional to mask × strength
            angle = mask * strength * math.pi
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            # Rotate (dx, dy) by angle, then recentre
            src_x = cx + dx * cos_a - dy * sin_a
            src_y = cy + dx * sin_a + dy * cos_a

        else:  # ripple
            # Concentric sine ripple
            ripple = np.sin(norm_dist * math.pi * 6) * mask * strength * radius * 0.15
            src_x = xs + ripple * (dx / (dist + 1e-6))
            src_y = ys + ripple * (dy / (dist + 1e-6))

        self._map_x = src_x.astype(np.float32)
        self._map_y = src_y.astype(np.float32)
        self._last_params = (h, w, cx, cy, radius, strength, self.mode)

    # ------------------------------------------------------------------

    def apply(
        self,
        frame: np.ndarray,
        face: Optional[FaceInfo] = None,
        audio: Optional[AudioFeatures] = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Face centre and radius
        if face is not None and face.detected:
            cx = face.cx_norm * w
            cy = face.cy_norm * h
            radius = max(face.w, face.h) * 0.8
        else:
            cx, cy = w / 2.0, h / 2.0
            radius = min(w, h) * 0.35

        # Audio boosts strength
        bass = self._audio_value(audio, "bass")
        strength = self.warp_amount + bass * 0.4
        strength = min(strength, 1.5)

        # Rebuild maps only when parameters change enough
        need_rebuild = self._last_params is None
        if not need_rebuild:
            ph, pw, pcx, pcy, pr, ps, pm = self._last_params
            need_rebuild = (
                (h, w) != (ph, pw)
                or abs(cx - pcx) > _REBUILD_THRESHOLD
                or abs(cy - pcy) > _REBUILD_THRESHOLD
                or abs(radius - pr) > 2.0
                or abs(strength - ps) > 0.05
                or self.mode != pm
            )

        if need_rebuild:
            self._rebuild_maps(h, w, cx, cy, radius, strength)

        result = cv2.remap(
            frame,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return result
