"""
input/camera_source.py — Webcam FrameSource implementation.

Wraps cv2.VideoCapture and exposes it through the unified FrameSource
interface.  A camera is only considered valid when it can actually deliver
a test frame — simply opening the capture device is not sufficient.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import FrameSource

logger = logging.getLogger(__name__)


class CameraSource(FrameSource):
    """Reads frames from a local webcam.

    Args:
        index:  Camera device index (0 = system default).
        width:  Desired frame width in pixels.
        height: Desired frame height in pixels.
    """

    def __init__(self, index: int = 0, width: int = 320, height: int = 240) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # FrameSource interface
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the webcam and validate that at least one frame can be read.

        Returns:
            ``True`` on success, ``False`` if the camera cannot be opened or
            does not deliver frames.
        """
        logger.info("Trying webcam at index %d", self.index)
        cap = cv2.VideoCapture(self.index)

        if not cap.isOpened():
            logger.warning("Webcam %d could not be opened", self.index)
            cap.release()
            return False

        # Validate with a test frame — some drivers open but produce no data.
        ret, _ = cap.read()
        if not ret:
            logger.warning(
                "Webcam %d opened but failed to deliver a test frame", self.index
            )
            cap.release()
            return False

        # Request the desired resolution.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Webcam opened successfully (index=%d, actual=%dx%d)",
            self.index,
            actual_w,
            actual_h,
        )

        self._cap = cap
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture the next webcam frame.

        Returns:
            ``(True, frame)`` on success, ``(False, None)`` otherwise.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Webcam read returned False (index=%d)", self.index)
            return False, None

        # Resize if the driver did not honour our resolution request.
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )

        return True, frame

    def release(self) -> None:
        """Release the webcam capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Webcam released (index=%d)", self.index)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return f"WEBCAM (index={self.index})"

    @property
    def is_live(self) -> bool:
        return True
