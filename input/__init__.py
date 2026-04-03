"""
input/ — Unified frame-source abstraction for DogDayDiffuser.

Public surface:

    from input import FrameSource, CameraSource, VideoFileSource, SourceManager
    from input.usb_media import find_usb_video_files
"""

from .base import FrameSource
from .camera_source import CameraSource
from .video_file_source import VideoFileSource
from .source_manager import SourceManager

__all__ = [
    "FrameSource",
    "CameraSource",
    "VideoFileSource",
    "SourceManager",
]
