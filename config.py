"""
config.py — Defaults and command-line argument parsing for DogDayDiffuser.

Loads settings from (in ascending priority order):
  1. Built-in defaults
  2. JSON config file (if --config is provided)
  3. Command-line flags
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """All tuneable parameters for the application."""

    # Camera
    camera: int = 0

    # Internal processing resolution
    width: int = 320
    height: int = 240

    # Display
    fullscreen: bool = False

    # Effect selection
    effect: str = "kaleidoscope"  # kaleidoscope | feedback | warp | color

    # Face detection
    no_face: bool = False
    use_openvino: bool = False
    detection_interval: int = 5  # run detector every N frames

    # Audio reactivity
    audio: bool = False
    audio_device: Optional[int] = None

    # USB audio auto-detection and HDMI passthrough
    audio_auto_usb: bool = True            # Auto-detect USB sound cards as input
    audio_prefer_usb: bool = True          # Prefer USB over explicit device index
    audio_output_prefer_hdmi: bool = True  # Prefer HDMI output for passthrough
    audio_buffer_size: int = 512           # Buffer size in frames
    audio_sample_rate: int = 44100         # Sampling rate in Hz
    audio_enable_passthrough: bool = False  # Route input to HDMI output

    # Input source selection
    prefer_camera: bool = True
    usb_mount_roots: List[str] = field(
        default_factory=lambda: ["/media", "/mnt", "/run/media"]
    )
    video_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
    )
    usb_video_loop: bool = True
    usb_video_shuffle: bool = False
    rescan_on_source_failure: bool = True

    # Config file path (not a runtime parameter)
    config: Optional[str] = None

    # Visual mode (overrides effect when set)
    default_mode: Optional[str] = None  # "geiss" | "milkdrop" | None

    # MilkDrop-specific
    milkdrop_auto_cycle: bool = True
    milkdrop_cycle_seconds: float = 15.0
    milkdrop_beat_transition: bool = True

    # Geiss-specific
    geiss_use_symmetry: bool = True
    geiss_plasma_overlay: bool = True

    # Shared mode settings
    mode_allow_face_modulation: bool = True
    mode_allow_audio_modulation: bool = True


EFFECT_NAMES = ("kaleidoscope", "feedback", "warp", "color")

MODE_NAMES = ("geiss", "milkdrop")


def load_json_config(path: str) -> dict:
    """Load a JSON settings file and return it as a plain dict."""
    if not os.path.isfile(path):
        logger.warning("Config file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded config from %s", path)
    return data


def parse_args(argv=None) -> AppConfig:
    """Parse command-line arguments and return an AppConfig instance.

    JSON config values are applied first so that CLI flags can still
    override individual settings.
    """
    parser = argparse.ArgumentParser(
        description="DogDayDiffuser — real-time psychedelic webcam effects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--camera", type=int, default=None,
                        help="Camera device index")
    parser.add_argument("--width", type=int, default=None,
                        help="Internal processing width in pixels")
    parser.add_argument("--height", type=int, default=None,
                        help="Internal processing height in pixels")
    parser.add_argument("--fullscreen", action="store_true", default=None,
                        help="Start in fullscreen mode")
    parser.add_argument("--effect", choices=EFFECT_NAMES, default=None,
                        help="Starting visual effect")
    parser.add_argument("--no-face", action="store_true", default=None,
                        dest="no_face", help="Disable face detection")
    parser.add_argument("--audio", action="store_true", default=None,
                        help="Enable audio reactivity")
    parser.add_argument("--audio-device", type=int, default=None,
                        dest="audio_device",
                        help="Audio input device index (uses system default if omitted)")
    parser.add_argument("--no-audio-auto-usb", action="store_false", default=None,
                        dest="audio_auto_usb",
                        help="Disable automatic USB sound card detection")
    parser.add_argument("--no-audio-prefer-usb", action="store_false", default=None,
                        dest="audio_prefer_usb",
                        help="Do not prefer USB device over explicit --audio-device")
    parser.add_argument("--no-audio-hdmi-out", action="store_false", default=None,
                        dest="audio_output_prefer_hdmi",
                        help="Disable automatic HDMI output selection for passthrough")
    parser.add_argument("--audio-buffer-size", type=int, default=None,
                        dest="audio_buffer_size",
                        help="Audio buffer size in frames (default: 512)")
    parser.add_argument("--audio-sample-rate", type=int, default=None,
                        dest="audio_sample_rate",
                        help="Audio sampling rate in Hz (default: 44100)")
    parser.add_argument("--audio-passthrough", action="store_true", default=None,
                        dest="audio_enable_passthrough",
                        help="Route USB audio input to HDMI output in real time")
    parser.add_argument("--use-openvino", action="store_true", default=None,
                        dest="use_openvino",
                        help="Use Intel OpenVINO / NCS2 for face detection")
    parser.add_argument("--detection-interval", type=int, default=None,
                        dest="detection_interval",
                        help="Run face detection every N frames")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON settings file")
    parser.add_argument("--no-prefer-camera", action="store_false", default=None,
                        dest="prefer_camera",
                        help="Skip webcam and go straight to USB video scan")
    parser.add_argument("--usb-mount-roots", type=str, nargs="+", default=None,
                        dest="usb_mount_roots",
                        help="Directories to scan for USB video files "
                             "(default: /media /mnt /run/media)")
    parser.add_argument("--video-extensions", type=str, nargs="+", default=None,
                        dest="video_extensions",
                        help="Supported video file extensions "
                             "(default: .mp4 .mov .avi .mkv .webm .m4v)")
    parser.add_argument("--no-usb-loop", action="store_false", default=None,
                        dest="usb_video_loop",
                        help="Do not loop USB video files when playback ends")
    parser.add_argument("--usb-shuffle", action="store_true", default=None,
                        dest="usb_video_shuffle",
                        help="Shuffle the USB video playlist")
    parser.add_argument("--no-rescan", action="store_false", default=None,
                        dest="rescan_on_source_failure",
                        help="Disable automatic source fallback on read failure")
    parser.add_argument("--mode", choices=MODE_NAMES, default=None,
                        help="Start in a visual mode: geiss or milkdrop")
    parser.add_argument("--no-milkdrop-auto-cycle", action="store_false", default=None,
                        dest="milkdrop_auto_cycle",
                        help="Disable automatic MilkDrop preset cycling")
    parser.add_argument("--milkdrop-cycle-seconds", type=float, default=None,
                        dest="milkdrop_cycle_seconds",
                        help="Seconds between automatic MilkDrop preset changes")
    parser.add_argument("--no-milkdrop-beat-transition", action="store_false",
                        default=None, dest="milkdrop_beat_transition",
                        help="Disable beat-triggered MilkDrop preset transitions")
    parser.add_argument("--no-geiss-symmetry", action="store_false", default=None,
                        dest="geiss_use_symmetry",
                        help="Disable rotational symmetry in Geiss mode")
    parser.add_argument("--no-geiss-plasma", action="store_false", default=None,
                        dest="geiss_plasma_overlay",
                        help="Disable plasma colour overlay in Geiss mode")
    parser.add_argument("--no-face-modulation", action="store_false", default=None,
                        dest="mode_allow_face_modulation",
                        help="Disable face-driven modulation in visual modes")
    parser.add_argument("--no-audio-modulation", action="store_false", default=None,
                        dest="mode_allow_audio_modulation",
                        help="Disable audio-driven modulation in visual modes")

    args = parser.parse_args(argv)

    # Start with built-in defaults
    cfg = AppConfig()

    # Apply JSON config overrides
    if args.config:
        json_data = load_json_config(args.config)
        for key, value in json_data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                logger.warning("Unknown config key ignored: %s", key)

    # Apply CLI overrides (only when explicitly provided)
    cli_map = {
        "camera": args.camera,
        "width": args.width,
        "height": args.height,
        "fullscreen": args.fullscreen,
        "effect": args.effect,
        "no_face": args.no_face,
        "audio": args.audio,
        "audio_device": args.audio_device,
        "audio_auto_usb": args.audio_auto_usb,
        "audio_prefer_usb": args.audio_prefer_usb,
        "audio_output_prefer_hdmi": args.audio_output_prefer_hdmi,
        "audio_buffer_size": args.audio_buffer_size,
        "audio_sample_rate": args.audio_sample_rate,
        "audio_enable_passthrough": args.audio_enable_passthrough,
        "use_openvino": args.use_openvino,
        "detection_interval": args.detection_interval,
        "config": args.config,
        "prefer_camera": args.prefer_camera,
        "usb_mount_roots": args.usb_mount_roots,
        "video_extensions": args.video_extensions,
        "usb_video_loop": args.usb_video_loop,
        "usb_video_shuffle": args.usb_video_shuffle,
        "rescan_on_source_failure": args.rescan_on_source_failure,
        "default_mode": args.mode,
        "milkdrop_auto_cycle": args.milkdrop_auto_cycle,
        "milkdrop_cycle_seconds": args.milkdrop_cycle_seconds,
        "milkdrop_beat_transition": args.milkdrop_beat_transition,
        "geiss_use_symmetry": args.geiss_use_symmetry,
        "geiss_plasma_overlay": args.geiss_plasma_overlay,
        "mode_allow_face_modulation": args.mode_allow_face_modulation,
        "mode_allow_audio_modulation": args.mode_allow_audio_modulation,
    }
    for key, value in cli_map.items():
        if value is not None:
            setattr(cfg, key, value)

    return cfg
