"""
midi/nanokontrol2.py — Korg nanoKONTROL2 default CC and Note constants.

All values match the factory default MIDI mapping (Scene 1, Channel 1).
Reference: Korg nanoKONTROL2 MIDI Implementation Chart.

Channel strips 1–8 share the same layout with incrementing CC numbers.
"""

# ---------------------------------------------------------------------------
# Device identification
# ---------------------------------------------------------------------------

#: Substring to match against MIDI port names for auto-detection.
DEVICE_NAME_SUBSTRING = "nanoKONTROL2"

# ---------------------------------------------------------------------------
# Continuous controllers — value range 0–127
# ---------------------------------------------------------------------------

# Faders (sliders) — one per channel strip
FADER_CC = [0, 1, 2, 3, 4, 5, 6, 7]

# Knobs — one per channel strip
KNOB_CC = [16, 17, 18, 19, 20, 21, 22, 23]

# Buttons — CC mode (factory default)
SOLO_CC = [32, 33, 34, 35, 36, 37, 38, 39]
MUTE_CC = [48, 49, 50, 51, 52, 53, 54, 55]
REC_CC  = [64, 65, 66, 67, 68, 69, 70, 71]

# ---------------------------------------------------------------------------
# Transport controls — CC mode (factory default)
# ---------------------------------------------------------------------------

TRANSPORT_BACK_CC  = 43
TRANSPORT_FWD_CC   = 44
TRANSPORT_STOP_CC  = 42
TRANSPORT_PLAY_CC  = 41
TRANSPORT_LOOP_CC  = 45
TRANSPORT_REC_CC   = 46

# ---------------------------------------------------------------------------
# Convenience sets for quick membership tests
# ---------------------------------------------------------------------------

ALL_CC: set = (
    set(FADER_CC)
    | set(KNOB_CC)
    | set(SOLO_CC)
    | set(MUTE_CC)
    | set(REC_CC)
    | {
        TRANSPORT_BACK_CC,
        TRANSPORT_FWD_CC,
        TRANSPORT_STOP_CC,
        TRANSPORT_PLAY_CC,
        TRANSPORT_LOOP_CC,
        TRANSPORT_REC_CC,
    }
)
