"""
audio/device_detection.py — Enumerate and identify audio input/output devices.

Provides helpers to:
  - Detect USB audio input devices (sound cards, USB microphones, interfaces).
  - Detect HDMI audio output devices.
  - Return human-readable labels for detected devices.

Device detection is done via ``sounddevice.query_devices()``.  The *sd*
module object is passed in as a parameter so callers (and tests) can
inject a mock without patching global imports.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Keywords that identify USB audio input devices (case-insensitive match on
# the device name reported by sounddevice / ALSA / PortAudio).
_USB_INPUT_KEYWORDS = (
    "usb",
    "focusrite",
    "scarlett",
    "behringer",
    "presonus",
    "steinberg",
    "m-audio",
    "motu",
    "arturia",
    "roland",
    "native instruments",
    "sound blaster",
    "c-media",
    "cmedia",
    "blue yeti",
    "blue snowball",
    "rode",
    "shure",
)

# Keywords that identify HDMI audio output devices.
_HDMI_OUTPUT_KEYWORDS = (
    "hdmi",
    "displayport",
    "dp audio",
)


def find_usb_input_device(sd) -> Optional[int]:
    """Return the index of the first USB audio input device, or ``None``.

    Iterates over all devices reported by *sd* (the ``sounddevice`` module)
    and returns the index of the first device whose name matches one of the
    known USB audio keywords and has at least one input channel.

    Args:
        sd: The ``sounddevice`` module (or a compatible mock).

    Returns:
        Integer device index, or ``None`` if no USB input is found.
    """
    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.warning("Could not query audio devices: %s", exc)
        return None

    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) < 1:
            continue
        name = device.get("name", "").lower()
        if any(kw in name for kw in _USB_INPUT_KEYWORDS):
            logger.info(
                "Found USB audio input device: [%d] %s", i, device.get("name", "")
            )
            return i

    logger.debug("No USB audio input device found among %d device(s)", len(devices))
    return None


def find_hdmi_output_device(sd) -> Optional[int]:
    """Return the index of the first HDMI audio output device, or ``None``.

    Iterates over all devices reported by *sd* and returns the index of the
    first device whose name contains an HDMI-related keyword and has at least
    one output channel.

    Args:
        sd: The ``sounddevice`` module (or a compatible mock).

    Returns:
        Integer device index, or ``None`` if no HDMI output is found.
    """
    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.warning("Could not query audio devices: %s", exc)
        return None

    for i, device in enumerate(devices):
        if device.get("max_output_channels", 0) < 1:
            continue
        name = device.get("name", "").lower()
        if any(kw in name for kw in _HDMI_OUTPUT_KEYWORDS):
            logger.info(
                "Found HDMI audio output device: [%d] %s", i, device.get("name", "")
            )
            return i

    logger.debug("No HDMI audio output device found")
    return None


def get_device_label(sd, device_index: Optional[int], default: str = "default") -> str:
    """Return a human-readable name for *device_index*.

    Args:
        sd:           The ``sounddevice`` module (or a compatible mock).
        device_index: Integer device index, or ``None`` for system default.
        default:      Label to return when *device_index* is ``None``.

    Returns:
        The device name string, or *default* if the index cannot be resolved.
    """
    if device_index is None:
        return default
    try:
        devices = sd.query_devices()
        if 0 <= device_index < len(devices):
            return str(devices[device_index].get("name", f"device {device_index}"))
    except Exception:
        pass
    return f"device {device_index}"
