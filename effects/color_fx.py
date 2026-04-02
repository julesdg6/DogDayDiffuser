"""
effects/color_fx.py — Fake-AI colour styling effect.

Applies a chain of cheap colour transforms to give frames a strange,
"processed" AI aesthetic:

  1. Posterisation  — quantises colours to a small number of levels
  2. Edge enhancement — emphasise outlines
  3. Channel shift  — laterally shift the red and blue channels
  4. Glow / bloom   — soft-threshold glow on bright areas
  5. Contrast pump  — stretch histogram for extra punch

All transforms use NumPy / OpenCV operations only (no per-pixel Python loops).

Audio mapping:
  volume → contrast pump amount
  bass   → posterisation levels (fewer levels on strong bass)
  treble → channel shift amount
  beat   → momentary brightness flash
"""

from typing import Optional

import cv2
import numpy as np

from .base import BaseEffect
from face_detection import FaceInfo
from audio_reactivity import AudioFeatures


class ColorFXEffect(BaseEffect):
    """Fake-AI colour processing chain."""

    name = "Color FX"

    def __init__(
        self,
        posterize_levels: int = 6,
        channel_shift: int = 4,
        glow_threshold: int = 180,
        contrast: float = 1.3,
    ):
        super().__init__()
        self.posterize_levels = posterize_levels
        self.channel_shift = channel_shift
        self.glow_threshold = glow_threshold
        self.contrast = contrast

    # ------------------------------------------------------------------
    # Individual transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _posterize(frame: np.ndarray, levels: int) -> np.ndarray:
        """Quantise each channel to *levels* distinct values."""
        levels = max(2, levels)
        step = 256 // levels
        # Integer division then multiply — cheap numpy op
        return ((frame // step) * step).astype(np.uint8)

    @staticmethod
    def _edge_enhance(frame: np.ndarray, weight: float = 0.4) -> np.ndarray:
        """Add sharpened edges back into the frame."""
        blurred = cv2.GaussianBlur(frame, (0, 0), 2)
        # unsharp mask
        sharpened = cv2.addWeighted(frame, 1.0 + weight, blurred, -weight, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def _channel_shift(frame: np.ndarray, shift: int) -> np.ndarray:
        """Shift red channel right and blue channel left by *shift* pixels."""
        if shift == 0:
            return frame
        b, g, r = cv2.split(frame)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        return cv2.merge([b, g, r])

    @staticmethod
    def _glow(frame: np.ndarray, threshold: int, intensity: float = 0.5) -> np.ndarray:
        """Add a soft bloom / glow around bright pixels."""
        # Mask of bright regions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        # Blur the mask to create bloom
        bloom = cv2.GaussianBlur(
            cv2.bitwise_and(frame, frame, mask=mask), (0, 0), 8
        )
        return cv2.addWeighted(frame, 1.0, bloom, intensity, 0)

    @staticmethod
    def _contrast_pump(frame: np.ndarray, factor: float) -> np.ndarray:
        """Scale pixel values around mid-grey for contrast."""
        f = frame.astype(np.float32)
        f = (f - 128.0) * factor + 128.0
        return np.clip(f, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------

    def apply(
        self,
        frame: np.ndarray,
        face: Optional[FaceInfo] = None,
        audio: Optional[AudioFeatures] = None,
    ) -> np.ndarray:
        # Audio-driven parameter modulation
        volume = self._audio_value(audio, "volume")
        bass   = self._audio_value(audio, "bass")
        treble = self._audio_value(audio, "treble")
        beat   = self._audio_value(audio, "beat")

        levels = max(2, self.posterize_levels - int(bass * 3))
        shift  = self.channel_shift + int(treble * 6)
        contrast = self.contrast + volume * 0.4

        # 1. Posterise
        result = self._posterize(frame, levels)

        # 2. Edge enhance
        result = self._edge_enhance(result, weight=0.3 + bass * 0.3)

        # 3. Channel shift
        result = self._channel_shift(result, shift)

        # 4. Glow
        threshold = max(100, self.glow_threshold - int(beat * 60))
        result = self._glow(result, threshold, intensity=0.4 + beat * 0.4)

        # 5. Contrast pump
        result = self._contrast_pump(result, contrast)

        return result
