"""
effects/feedback.py — Zoom feedback / recursive trail effect.

Blends the current frame with a zoomed, optionally-rotated and slightly
shifted copy of the previous output.  Creates recursive trail / tunnel
visuals characteristic of analogue video feedback.

Parameters tuned via instance attributes:
  trail_decay    — how much of the previous buffer survives (0–1)
  zoom           — zoom factor per frame (1.0 = no zoom; >1 zooms in)
  rotation_deg   — degrees of rotation added each frame
  drift_x/y      — pixel offset drift per frame

Audio mapping (when audio is available):
  bass  → boosts zoom
  beat  → pulses trail_decay
  treble → adds a little rotation
"""

from typing import Optional

import cv2
import numpy as np

from .base import BaseEffect
from face_detection import FaceInfo
from audio_reactivity import AudioFeatures


class FeedbackEffect(BaseEffect):
    """Zoom feedback / recursive trail effect."""

    name = "Feedback"

    def __init__(
        self,
        zoom: float = 1.02,
        rotation_deg: float = 0.3,
        drift_x: float = 0.5,
        drift_y: float = 0.3,
    ):
        super().__init__()
        self.zoom = zoom
        self.rotation_deg = rotation_deg
        self.drift_x = drift_x
        self.drift_y = drift_y
        self.trail_decay = 0.88

        self._buffer: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def apply(
        self,
        frame: np.ndarray,
        face: Optional[FaceInfo] = None,
        audio: Optional[AudioFeatures] = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Audio modulation
        bass   = self._audio_value(audio, "bass")
        beat   = self._audio_value(audio, "beat")
        treble = self._audio_value(audio, "treble")

        zoom   = self.zoom + bass * 0.03
        decay  = self.trail_decay - beat * 0.15
        decay  = max(0.3, min(0.98, decay))
        rot_deg = self.rotation_deg + treble * 0.5

        # Face adjusts drift direction
        if face is not None and face.detected:
            dx = (face.cx_norm - 0.5) * self.drift_x * 4
            dy = (face.cy_norm - 0.5) * self.drift_y * 4
        else:
            dx, dy = self.drift_x, self.drift_y

        # Initialise buffer from first frame
        if self._buffer is None or self._buffer.shape[:2] != (h, w):
            self._buffer = frame.astype(np.float32)

        # Build affine transform: zoom + rotation + drift
        M = cv2.getRotationMatrix2D((cx, cy), rot_deg, zoom)
        M[0, 2] += dx
        M[1, 2] += dy

        # Warp the previous buffer
        warped = cv2.warpAffine(
            self._buffer, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # Blend: new frame + decayed warped trail
        frame_f = frame.astype(np.float32)
        blended = frame_f * (1.0 - decay) + warped * decay

        blended = np.clip(blended, 0, 255)
        self._buffer = blended
        return blended.astype(np.uint8)
