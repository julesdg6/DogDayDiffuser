"""
renderer.py — Compositing and final output display for DogDayDiffuser.

Handles:
  - Upscaling the low-resolution processed frame to the display size
  - Overlaying debug info (FPS, effect name)
  - Managing the OpenCV display window (normal and fullscreen)
  - Drawing optional face bounding boxes
"""

import logging
from typing import Optional

import cv2
import numpy as np

from face_detection import FaceInfo
from utils import draw_fps, draw_effect_name, scale_to_fit

logger = logging.getLogger(__name__)

WINDOW_NAME = "DogDayDiffuser"


class Renderer:
    """Manages the output display window.

    Args:
        fullscreen: Start in fullscreen mode.
        show_fps:   Overlay FPS counter.
        show_face:  Draw face bounding box in debug mode.
    """

    def __init__(
        self,
        fullscreen: bool = False,
        show_fps: bool = True,
        show_face: bool = False,
    ):
        self.fullscreen = fullscreen
        self.show_fps = show_fps
        self.show_face = show_face

        # Create window
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        logger.info("Renderer initialised (fullscreen=%s)", fullscreen)

    # ------------------------------------------------------------------

    def show(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        effect_name: str = "",
        face: Optional[FaceInfo] = None,
    ) -> None:
        """Display *frame* in the output window.

        The frame is scaled up to fill the window while preserving aspect
        ratio.  Overlays are applied after scaling so text is always readable.

        Args:
            frame:       Processed BGR uint8 frame (at internal resolution).
            fps:         Current smoothed FPS value.
            effect_name: Name of the active effect (shown in corner).
            face:        Face detection result for optional bounding box.
        """
        # Upscale for display
        display = cv2.resize(frame, (0, 0), fx=2, fy=2,
                             interpolation=cv2.INTER_LINEAR)

        # Optional face bounding box (scaled up by 2×)
        if self.show_face and face is not None and face.detected:
            x, y, w, h = face.x * 2, face.y * 2, face.w * 2, face.h * 2
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Overlays
        if self.show_fps:
            draw_fps(display, fps)
        if effect_name:
            draw_effect_name(display, effect_name)

        cv2.imshow(WINDOW_NAME, display)

    def toggle_fullscreen(self) -> None:
        """Toggle between fullscreen and windowed mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
            )
        logger.info("Fullscreen toggled → %s", self.fullscreen)

    def destroy(self) -> None:
        """Close the display window."""
        cv2.destroyWindow(WINDOW_NAME)
        logger.info("Renderer destroyed")
