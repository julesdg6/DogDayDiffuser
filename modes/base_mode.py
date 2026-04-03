"""
modes/base_mode.py — Abstract base class for DogDayDiffuser visual modes.

A VisualMode is a higher-level concept than an effect.  Modes own their own
internal state machines, preset systems, and signal routing logic.

Signal dictionary keys (all optional — absent keys default safely):
    audio_level     float 0-1   overall audio volume
    audio_bass      float 0-1   low-frequency energy
    audio_mid       float 0-1   mid-frequency energy
    audio_treble    float 0-1   high-frequency energy
    beat_pulse      float 0-1   onset / beat detection pulse
    motion          float 0-1   inter-frame motion magnitude
    face_count      int         number of detected faces (0 or 1)
    face_center     (x, y)      face centre pixel coordinates
    face_size       float       approximate face diameter in pixels
    fps             float       current smoothed frame rate
    source_name     str         human-readable input source label
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class VisualMode(ABC):
    """Abstract base for all visual modes.

    Subclasses must implement :meth:`render`.

    Lifecycle:
        - Instantiated once at startup.
        - :meth:`reset` is called when the mode is (re-)activated.
        - :meth:`update` receives the signal dictionary every frame (before
          render), allowing modes to maintain internal timers and state.
        - :meth:`render` receives the source frame and returns a processed
          BGR uint8 frame at the same resolution.
    """

    #: Human-readable display name shown in the on-screen overlay
    name: str = "Base Mode"

    #: Subtitle shown alongside name (e.g. current preset)
    subtitle: str = ""

    def reset(self) -> None:
        """Reset internal state.  Called when mode is activated."""

    def update(self, dt: float, signals: Dict[str, Any]) -> None:
        """Advance internal timers / state machines.

        Args:
            dt:      Elapsed seconds since the last frame.
            signals: Signal dictionary (see module docstring).
        """

    @abstractmethod
    def render(self, frame: np.ndarray, signals: Dict[str, Any]) -> np.ndarray:
        """Apply the mode to *frame* and return the result.

        Args:
            frame:   BGR uint8 source frame at internal processing resolution.
            signals: Signal dictionary (see module docstring).

        Returns:
            Processed BGR uint8 frame (same resolution as input).
        """

    # ------------------------------------------------------------------
    # Convenience signal accessors
    # ------------------------------------------------------------------

    @staticmethod
    def _sig(signals: Dict[str, Any], key: str, default: Any = 0.0) -> Any:
        """Safely read a value from the signal dict."""
        return signals.get(key, default)
