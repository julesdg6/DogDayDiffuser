"""
camera.py — Webcam capture abstraction for DogDayDiffuser.

Wraps cv2.VideoCapture with error handling and optional frame downscaling.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Manages webcam capture.

    Usage::

        cam = Camera(index=0, width=320, height=240)
        cam.open()
        frame = cam.read()   # returns None on failure
        cam.release()

    Can also be used as a context manager::

        with Camera(0, 320, 240) as cam:
            frame = cam.read()
    """

    def __init__(self, index: int = 0, width: int = 320, height: int = 240):
        """
        Args:
            index:  Camera device index (0 = default camera).
            width:  Target capture width in pixels.
            height: Target capture height in pixels.
        """
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the camera.  Returns True on success, raises RuntimeError on failure."""
        logger.info("Opening camera %d at %dx%d", self.index, self.width, self.height)
        self._cap = cv2.VideoCapture(self.index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.index}. "
                "Check that a webcam is connected and not in use by another process."
            )

        # Request the desired resolution from the driver; it may not honour it.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera opened — actual resolution: %dx%d", actual_w, actual_h)
        return True

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray | None:
        """Capture and return the next frame, or None on failure.

        The frame is returned as a BGR uint8 NumPy array.  If the camera
        delivers a resolution different from the requested one the frame is
        resized before being returned.
        """
        if self._cap is None or not self._cap.isOpened():
            logger.warning("Camera.read() called on closed capture")
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Failed to read frame from camera %d", self.index)
            return None

        # Resize if the driver didn't honour our resolution request
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )

        return frame

    @property
    def is_open(self) -> bool:
        """True if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()
