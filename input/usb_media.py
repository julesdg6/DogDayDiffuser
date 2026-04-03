"""
input/usb_media.py — USB storage device discovery for DogDayDiffuser.

Scans common Linux mount roots for video files on removable/USB storage
and returns a sorted list of candidate paths.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Default mount roots to search on Linux / Raspberry Pi OS.
DEFAULT_MOUNT_ROOTS: List[str] = ["/media", "/mnt", "/run/media"]

# Supported video file extensions (lower-case, with leading dot).
DEFAULT_VIDEO_EXTENSIONS: List[str] = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]


def find_usb_video_files(
    mount_roots: List[str] | None = None,
    extensions: List[str] | None = None,
) -> List[Path]:
    """Discover video files on mounted USB storage devices.

    Walks each directory in *mount_roots*, skipping hidden entries, and
    collects all files whose suffix matches *extensions*.  Results are
    returned sorted by full path so the order is deterministic.

    Args:
        mount_roots: List of top-level mount directories to search.
                     Defaults to :data:`DEFAULT_MOUNT_ROOTS`.
        extensions:  Allowed file extensions (lower-case, with leading
                     dot).  Defaults to :data:`DEFAULT_VIDEO_EXTENSIONS`.

    Returns:
        Sorted list of :class:`pathlib.Path` objects for every matching
        video file found.  An empty list means nothing was found.
    """
    if mount_roots is None:
        mount_roots = DEFAULT_MOUNT_ROOTS
    if extensions is None:
        extensions = DEFAULT_VIDEO_EXTENSIONS

    ext_set = {e.lower() for e in extensions}
    found: List[Path] = []

    for root_str in mount_roots:
        root = Path(root_str)
        if not root.is_dir():
            logger.debug("Mount root does not exist, skipping: %s", root)
            continue

        logger.info("Scanning %s for video files", root)

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden directories (e.g. .Spotlight-V100 on macOS USB sticks).
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            for filename in filenames:
                if filename.startswith("."):
                    continue
                if Path(filename).suffix.lower() in ext_set:
                    found.append(Path(dirpath) / filename)

    found.sort()
    logger.info("Found %d USB video file(s)", len(found))
    return found
