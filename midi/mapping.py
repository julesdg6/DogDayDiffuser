"""
midi/mapping.py — CC/Note-to-parameter mapping and live MIDI state.

MidiState holds a flat dict of named float parameters (0.0–1.0) that
the rest of the application can read without knowing anything about MIDI.

DEFAULT_MAPPING maps each CC number (or transport action name) to a
parameter name in MidiState.  This corresponds to the "default_vj" profile
described in the project issue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from . import nanokontrol2 as nk2


# ---------------------------------------------------------------------------
# Live parameter state
# ---------------------------------------------------------------------------


@dataclass
class MidiState:
    """Current values of all MIDI-controlled parameters (0.0 – 1.0).

    Parameters are named after the visual effect controls they drive.
    Any parameter not actively controlled by a CC stays at its default.
    """

    # Fader-controlled
    master_intensity: float = 0.8
    warp_amount: float = 0.5
    feedback_decay: float = 0.85
    zoom: float = 0.5
    symmetry: float = 0.5
    color_shift: float = 0.0
    glow: float = 0.0
    mix: float = 0.5

    # Knob-controlled
    warp_speed: float = 0.5
    rotation: float = 0.0
    noise_amount: float = 0.0
    edge_gain: float = 0.5
    audio_sensitivity: float = 0.5
    palette_shift: float = 0.0
    trail_length: float = 0.5
    brightness: float = 1.0

    # Button toggles (True = active)
    solo_active: Dict[int, bool] = field(default_factory=dict)   # strip 0–7
    mute_active: Dict[int, bool] = field(default_factory=dict)   # strip 0–7
    rec_active: Dict[int, bool] = field(default_factory=dict)    # strip 0–7

    # Transport
    playing: bool = False
    recording: bool = False

    def as_dict(self) -> dict:
        """Return a plain-dict snapshot of all scalar parameters."""
        return {
            "master_intensity": self.master_intensity,
            "warp_amount": self.warp_amount,
            "feedback_decay": self.feedback_decay,
            "zoom": self.zoom,
            "symmetry": self.symmetry,
            "color_shift": self.color_shift,
            "glow": self.glow,
            "mix": self.mix,
            "warp_speed": self.warp_speed,
            "rotation": self.rotation,
            "noise_amount": self.noise_amount,
            "edge_gain": self.edge_gain,
            "audio_sensitivity": self.audio_sensitivity,
            "palette_shift": self.palette_shift,
            "trail_length": self.trail_length,
            "brightness": self.brightness,
            "playing": self.playing,
            "recording": self.recording,
        }


# ---------------------------------------------------------------------------
# Default CC → parameter mapping  (profile "default_vj")
# ---------------------------------------------------------------------------

#: Maps CC number → parameter name on MidiState (continuous, 0–127 → 0.0–1.0)
_FADER_PARAMS = [
    "master_intensity",
    "warp_amount",
    "feedback_decay",
    "zoom",
    "symmetry",
    "color_shift",
    "glow",
    "mix",
]

_KNOB_PARAMS = [
    "warp_speed",
    "rotation",
    "noise_amount",
    "edge_gain",
    "audio_sensitivity",
    "palette_shift",
    "trail_length",
    "brightness",
]


def build_cc_map() -> Dict[int, str]:
    """Return a dict mapping CC numbers to MidiState attribute names."""
    cc_map: Dict[int, str] = {}

    for i, param in enumerate(_FADER_PARAMS):
        cc_map[nk2.FADER_CC[i]] = param

    for i, param in enumerate(_KNOB_PARAMS):
        cc_map[nk2.KNOB_CC[i]] = param

    return cc_map


#: Default CC → param mapping (built once at import time)
DEFAULT_CC_MAP: Dict[int, str] = build_cc_map()

#: Transport CC → (attribute, action) pairs.
#: action is "set_true", "set_false", or "toggle".
TRANSPORT_CC_ACTIONS: Dict[int, tuple] = {
    nk2.TRANSPORT_PLAY_CC: ("playing", "set_true"),
    nk2.TRANSPORT_STOP_CC: ("playing", "set_false"),
    nk2.TRANSPORT_REC_CC:  ("recording", "toggle"),
}
