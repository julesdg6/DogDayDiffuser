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


EFFECT_NAMES = ("kaleidoscope", "feedback", "warp", "color")


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
        "use_openvino": args.use_openvino,
        "detection_interval": args.detection_interval,
        "config": args.config,
        "prefer_camera": args.prefer_camera,
        "usb_mount_roots": args.usb_mount_roots,
        "video_extensions": args.video_extensions,
        "usb_video_loop": args.usb_video_loop,
        "usb_video_shuffle": args.usb_video_shuffle,
        "rescan_on_source_failure": args.rescan_on_source_failure,
    }
    for key, value in cli_map.items():
        if value is not None:
            setattr(cfg, key, value)

    return cfg
