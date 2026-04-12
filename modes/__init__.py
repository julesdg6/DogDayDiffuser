"""
modes/__init__.py — Mode registry for DogDayDiffuser.
"""

from __future__ import annotations

from typing import Dict, Type, Optional, Any

from .base_mode import VisualMode
from .geiss_mode import GeissMode
from .milkdrop_mode import MilkDropMode, PRESET_NAMES
from .audio_tunnel_mode import AudioTunnelMode

# Mapping from CLI/config name to mode class
MODES: Dict[str, Type[VisualMode]] = {
    "geiss": GeissMode,
    "milkdrop": MilkDropMode,
    "audio_tunnel": AudioTunnelMode,
}

MODE_NAMES = list(MODES.keys())


def build_signals(
    audio=None,
    face=None,
    fps: float = 0.0,
    source_name: str = "",
    frame_w: int = 320,
    frame_h: int = 240,
    motion: float = 0.0,
) -> Dict[str, Any]:
    """Build the canonical signals dictionary from available app state.

    All values default safely when the corresponding subsystem is unavailable.

    Args:
        audio:       AudioFeatures snapshot or None.
        face:        FaceInfo snapshot or None.
        fps:         Current smoothed frame rate.
        source_name: Human-readable input source label.
        frame_w:     Internal frame width in pixels.
        frame_h:     Internal frame height in pixels.
        motion:      Inter-frame motion magnitude 0–1.

    Returns:
        Dictionary with the keys described in base_mode module docstring.
    """
    cx_default = frame_w / 2.0
    cy_default = frame_h / 2.0

    signals: Dict[str, Any] = {
        "audio_level":  0.0,
        "audio_bass":   0.0,
        "audio_mid":    0.0,
        "audio_treble": 0.0,
        "beat_pulse":   0.0,
        "motion":       motion,
        "face_count":   0,
        "face_center":  (cx_default, cy_default),
        "face_size":    0.0,
        "fps":          fps,
        "source_name":  source_name,
    }

    if audio is not None:
        signals["audio_level"]  = float(getattr(audio, "volume", 0.0))
        signals["audio_bass"]   = float(getattr(audio, "bass", 0.0))
        signals["audio_mid"]    = float(getattr(audio, "mid", 0.0))
        signals["audio_treble"] = float(getattr(audio, "treble", 0.0))
        signals["beat_pulse"]   = float(getattr(audio, "beat", 0.0))

    if face is not None and getattr(face, "detected", False):
        signals["face_count"]  = 1
        signals["face_center"] = (
            float(face.cx_norm) * frame_w,
            float(face.cy_norm) * frame_h,
        )
        signals["face_size"] = float(max(
            getattr(face, "w", 0),
            getattr(face, "h", 0),
        ))

    return signals


__all__ = [
    "VisualMode",
    "GeissMode",
    "MilkDropMode",
    "AudioTunnelMode",
    "MODES",
    "MODE_NAMES",
    "PRESET_NAMES",
    "build_signals",
]
