"""
input/video_file_source.py — Video-file playback FrameSource for DogDayDiffuser.

Opens a local video file via cv2.VideoCapture and exposes it through the
unified FrameSource interface.  Supports optional looping (default) and
optional advancing to the next file when the current one ends.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import FrameSource

logger = logging.getLogger(__name__)


class VideoFileSource(FrameSource):
    """Reads frames from one or more video files.

    When *playlist* contains more than one path the source advances to the
    next file when the current one ends (if *loop* is ``True`` the whole
    playlist restarts after the last entry).

    Args:
        playlist: Ordered list of video file paths to play.
        width:    Target frame width.  ``0`` keeps the native file width.
        height:   Target frame height.  ``0`` keeps the native file height.
        loop:     Restart playback from the beginning when the playlist ends.
    """

    def __init__(
        self,
        playlist: List[Path],
        width: int = 320,
        height: int = 240,
        loop: bool = True,
    ) -> None:
        if not playlist:
            raise ValueError("VideoFileSource requires at least one video path")

        self._playlist = list(playlist)
        self.width = width
        self.height = height
        self.loop = loop

        self._index: int = 0  # index into _playlist
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _current_path(self) -> Path:
        return self._playlist[self._index]

    def _open_current(self) -> bool:
        """Open the video file at ``self._index``.  Returns True on success."""
        path = self._current_path
        logger.info("Opening video file: %s", path)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.warning("Cannot open video file: %s", path)
            cap.release()
            return False

        ret, _ = cap.read()
        if not ret:
            logger.warning("Video file opened but produced no frames: %s", path)
            cap.release()
            return False

        # Rewind so the first frame is available via read().
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._cap = cap
        logger.info("Selected USB video source: %s", path)
        return True

    def _advance(self) -> bool:
        """Move to the next file in the playlist.

        Returns ``True`` if a new file was successfully opened, ``False``
        if all options are exhausted (or loop is disabled).
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # Try each remaining file.
        for _ in range(len(self._playlist)):
            self._index = (self._index + 1) % len(self._playlist)

            if self._index == 0 and not self.loop:
                logger.info("Playlist ended and loop=False; no more video files")
                return False

            if self._open_current():
                return True

            logger.warning("Skipping unreadable video: %s", self._current_path)

        logger.warning("No readable video files remain in the playlist")
        return False

    # ------------------------------------------------------------------
    # FrameSource interface
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the first file in the playlist.

        Returns:
            ``True`` if at least one file could be opened, ``False``
            otherwise.
        """
        # Try each file in order until one works.
        for attempt in range(len(self._playlist)):
            self._index = attempt
            if self._open_current():
                return True
            logger.warning("Skipping unreadable video: %s", self._current_path)

        logger.error("No readable video files found in playlist")
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the next video frame.

        When the current file ends, the source either loops, advances to
        the next file, or signals failure (``False, None``) if no further
        files are available.

        Returns:
            ``(True, frame)`` on success, ``(False, None)`` on failure.
        """
        if self._cap is None:
            return False, None

        ret, frame = self._cap.read()

        if not ret or frame is None:
            logger.info("USB video reached end: %s", self._current_path)
            if len(self._playlist) == 1:
                # Single-file: rewind if looping is enabled.
                if not self.loop:
                    logger.info("USB video reached end (loop=False), stopping")
                    return False, None
                logger.info("USB video reached end, looping")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return False, None
            else:
                # Multi-file: advance.
                if not self._advance():
                    return False, None
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return False, None

        # Resize if needed.
        if self.width > 0 and self.height > 0:
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(
                    frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST
                )

        return True, frame

    def release(self) -> None:
        """Release the video capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("VideoFileSource released: %s", self._current_path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return f"USB VIDEO - {self._current_path.name}"

    @property
    def is_live(self) -> bool:
        return False
