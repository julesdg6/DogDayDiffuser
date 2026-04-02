"""
effects/base.py — Abstract base class for DogDayDiffuser visual effects.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from face_detection import FaceInfo
from audio_reactivity import AudioFeatures


class BaseEffect(ABC):
    """Abstract base for all visual effects.

    Subclasses must implement :meth:`apply`.

    Shared tuneable parameters (all effects should respect these where
    applicable):

    - ``strength`` — general intensity / mix amount (0.0 – 1.0)
    - ``warp_amount`` — local distortion strength (0.0 – 1.0)
    - ``trail_decay`` — feedback trail persistence (0.0 – 1.0)
    """

    #: Human-readable display name
    name: str = "Base Effect"

    def __init__(self):
        self.strength: float = 0.8
        self.warp_amount: float = 0.5
        self.trail_decay: float = 0.85

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(
        self,
        frame: np.ndarray,
        face: Optional[FaceInfo] = None,
        audio: Optional[AudioFeatures] = None,
    ) -> np.ndarray:
        """Apply the effect to *frame* and return the result.

        Args:
            frame: BGR uint8 input frame at the internal processing resolution.
            face:  Latest face detection result, or None if no face detected.
            audio: Latest audio features, or None if audio is disabled.

        Returns:
            Processed BGR uint8 frame.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _audio_value(self, audio: Optional[AudioFeatures],
                     feature: str, default: float = 0.0) -> float:
        """Safely read an audio feature value."""
        if audio is None:
            return default
        return getattr(audio, feature, default)
