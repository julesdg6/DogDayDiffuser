"""
midi — Korg nanoKONTROL2 MIDI support for DogDayDiffuser.

Public API::

    from midi import MidiManager, MidiState, DEFAULT_CC_MAP

Example::

    mgr = MidiManager()
    try:
        mgr.open()          # auto-detects nanoKONTROL2
    except (ImportError, RuntimeError) as exc:
        print(f"MIDI unavailable: {exc}")
    else:
        state = mgr.get_state()
        print(state.warp_amount)
        mgr.close()
"""

from .midi_manager import MidiManager
from .mapping import MidiState, DEFAULT_CC_MAP

__all__ = [
    "MidiManager",
    "MidiState",
    "DEFAULT_CC_MAP",
]
