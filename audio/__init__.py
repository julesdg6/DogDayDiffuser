"""
audio — USB audio auto-detection and HDMI passthrough for DogDayDiffuser.

Exports the top-level AudioManager which replaces AudioReactor when advanced
device selection or passthrough is needed.
"""

from audio.audio_manager import AudioManager
from audio.device_detection import (
    find_usb_input_device,
    find_hdmi_output_device,
    get_device_label,
)

__all__ = [
    "AudioManager",
    "find_usb_input_device",
    "find_hdmi_output_device",
    "get_device_label",
]
