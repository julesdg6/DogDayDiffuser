"""
input/source_manager.py — Unified input-source orchestrator for DogDayDiffuser.

Implements the startup selection logic (webcam → USB video → failure) and
runtime fallback when the active source stops delivering frames.

The rest of the application interacts only with SourceManager.read() and
SourceManager.release(), without knowing whether frames come from a webcam
or a video file.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .base import FrameSource
from .camera_source import CameraSource
from .usb_media import find_usb_video_files
from .video_file_source import VideoFileSource

logger = logging.getLogger(__name__)


class SourceManager:
    """Manages selection and failover between available frame sources.

    Selection priority:
        1. Webcam (if ``prefer_camera`` is ``True`` in config, default).
        2. USB video files (scanned from ``usb_mount_roots``).
        3. Clean failure if neither is available.

    Runtime behaviour:
        - If the active webcam source begins failing, the manager attempts
          USB video fallback (when ``rescan_on_source_failure`` is enabled).
        - If USB video playback reaches the end and the source returns no
          further frames, the manager logs the event and signals failure so
          the main loop can react (e.g. show a status screen).

    Args:
        config: AppConfig instance supplying camera index, resolution,
                USB mount roots, and feature flags.
    """

    # How many consecutive failed reads to tolerate before triggering
    # a source-failure event.
    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(self, config) -> None:
        self._cfg = config
        self._source: Optional[FrameSource] = None
        self._consecutive_failures: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Select the best available source and open it.

        Returns:
            ``True`` if a usable source was found, ``False`` if startup
            should be aborted (no valid source at all).
        """
        if getattr(self._cfg, "prefer_camera", True):
            if self._try_camera():
                return True
            logger.info("Webcam unavailable, falling back to USB scan")
        else:
            logger.info("Camera preference disabled; going straight to USB scan")

        if self._try_usb_video():
            return True

        logger.error("No valid input source available")
        return False

    def _try_camera(self) -> bool:
        """Attempt to open the configured webcam.  Returns True on success."""
        source = CameraSource(
            index=getattr(self._cfg, "camera", 0),
            width=getattr(self._cfg, "width", 320),
            height=getattr(self._cfg, "height", 240),
        )
        if source.open():
            self._source = source
            logger.info("Active source: %s", source.source_name)
            return True
        return False

    def _try_usb_video(self) -> bool:
        """Scan USB mounts for video files and open the first valid one.

        Returns True on success.
        """
        mount_roots = getattr(self._cfg, "usb_mount_roots", None)
        extensions = getattr(self._cfg, "video_extensions", None)

        video_files = find_usb_video_files(
            mount_roots=mount_roots,
            extensions=extensions,
        )

        if not video_files:
            logger.warning("No USB video files found")
            return False

        source = VideoFileSource(
            playlist=video_files,
            width=getattr(self._cfg, "width", 320),
            height=getattr(self._cfg, "height", 240),
            loop=getattr(self._cfg, "usb_video_loop", True),
        )

        if source.open():
            self._source = source
            logger.info("Active source: %s", source.source_name)
            return True

        logger.error("USB video files were found but none could be opened")
        return False

    # ------------------------------------------------------------------
    # Frame delivery
    # ------------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the next frame from the active source.

        When the active source starts failing consistently and
        ``rescan_on_source_failure`` is enabled, the manager attempts to
        fall back to a USB video source automatically.

        Returns:
            ``(True, frame)`` on success, ``(False, None)`` on failure.
        """
        if self._source is None:
            return False, None

        ok, frame = self._source.read()

        if ok:
            self._consecutive_failures = 0
            return True, frame

        self._consecutive_failures += 1
        logger.warning(
            "Active source failed: %s read returned false (consecutive=%d)",
            self._source.source_name,
            self._consecutive_failures,
        )

        # Attempt fallback only when we've exceeded the tolerance threshold
        # and the config allows rescanning.
        if (
            self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES
            and getattr(self._cfg, "rescan_on_source_failure", True)
        ):
            logger.info("Attempting source fallback after %d consecutive failures",
                        self._consecutive_failures)

            # Only attempt USB video fallback if the current source is a webcam
            # (avoids an infinite loop if a video file itself is broken).
            if isinstance(self._source, CameraSource):
                old_source = self._source
                if self._try_usb_video():
                    old_source.release()
                    self._consecutive_failures = 0
                    logger.info("Switched to USB video after webcam failure")
                else:
                    # Reset counter so we don't spam logs on every subsequent read.
                    self._consecutive_failures = 0
                    logger.warning("Webcam failed and no USB video fallback available")
            else:
                self._consecutive_failures = 0

        return False, None

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a status snapshot for diagnostics / overlay display.

        Returns:
            A plain dictionary with keys ``source_name``, ``is_live``, and
            ``active`` (bool indicating whether a source is open).
        """
        if self._source is None:
            return {
                "source_name": "NO INPUT",
                "is_live": False,
                "active": False,
            }
        return {
            "source_name": self._source.source_name,
            "is_live": self._source.is_live,
            "active": True,
        }

    @property
    def source_name(self) -> str:
        """Human-readable name of the currently active source."""
        if self._source is None:
            return "NO INPUT"
        return self._source.source_name

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release the active source and clean up resources."""
        if self._source is not None:
            self._source.release()
            self._source = None
            logger.info("SourceManager released")
