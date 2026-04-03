"""
renderer.py — Compositing and final output display for DogDayDiffuser.

Handles:
  - Upscaling the low-resolution processed frame to the display size
  - Overlaying debug info (FPS, effect name)
  - Managing the OpenCV display window (normal and fullscreen)
  - Drawing optional face bounding boxes
"""

import logging
import os
import re
import subprocess
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
        self._headless = False
        self._screen_size: Optional[tuple[int, int]] = None
        self._backend = "cv2"
        self._pygame = None
        self._pg_screen = None

        # On Linux servers/containers there may be no display available.
        if os.name != "nt" and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            self._headless = True
            logger.warning("No display detected; renderer running in headless mode")
            return

        # Prefer pygame for fullscreen stability on Raspberry Pi/VNC sessions.
        if self._init_pygame():
            logger.info("Renderer initialised via pygame (fullscreen=%s)", fullscreen)
            return

        # Fallback to OpenCV window backend.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        self._screen_size = self._detect_screen_size()
        if fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            if self._screen_size is not None:
                sw, sh = self._screen_size
                cv2.resizeWindow(WINDOW_NAME, sw, sh)
        logger.info("Renderer initialised via cv2 (fullscreen=%s)", fullscreen)

    def _init_pygame(self) -> bool:
        """Try to initialize pygame display backend."""
        try:
            import pygame  # type: ignore
        except Exception:
            return False

        try:
            pygame.init()
            self._pygame = pygame
            flags = pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
            if self.fullscreen:
                self._pg_screen = pygame.display.set_mode((0, 0), flags)
            else:
                self._pg_screen = pygame.display.set_mode((640, 480), flags)

            pygame.display.set_caption(WINDOW_NAME)
            size = self._pg_screen.get_size()
            self._screen_size = (int(size[0]), int(size[1]))
            self._backend = "pygame"
            return True
        except Exception as exc:
            logger.warning("Pygame renderer unavailable (%s); using cv2", exc)
            try:
                pygame.quit()
            except Exception:
                pass
            self._pygame = None
            self._pg_screen = None
            return False

    def _detect_screen_size(self) -> Optional[tuple[int, int]]:
        """Try to detect active display resolution (Linux/X11)."""
        if os.name == "nt":
            return None
        if not os.environ.get("DISPLAY"):
            return None

        try:
            proc = subprocess.run(
                ["xrandr", "--current"],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                return None

            # Prefer the line marked as current mode with '*', e.g. "1920x1080 60.00*+"
            for line in proc.stdout.splitlines():
                m = re.search(r"\b(\d+)x(\d+)\b.*\*", line)
                if m:
                    return int(m.group(1)), int(m.group(2))
        except Exception:
            return None

        return None

    @property
    def is_headless(self) -> bool:
        """True when no GUI display is available."""
        return self._headless

    def poll_key(self) -> int:
        """Return key code compatible with main loop controls, or -1."""
        if self._headless:
            return -1

        if self._backend == "pygame" and self._pygame is not None:
            keymap = {
                self._pygame.K_ESCAPE: 27,
                self._pygame.K_q: ord("q"),
                self._pygame.K_f: ord("f"),
                self._pygame.K_SPACE: ord(" "),
                self._pygame.K_1: ord("1"),
                self._pygame.K_2: ord("2"),
                self._pygame.K_3: ord("3"),
                self._pygame.K_4: ord("4"),
                self._pygame.K_a: ord("a"),
                self._pygame.K_d: ord("d"),
                self._pygame.K_EQUALS: ord("="),
                self._pygame.K_MINUS: ord("-"),
                self._pygame.K_w: ord("w"),
                self._pygame.K_s: ord("s"),
            }

            k_plus = getattr(self._pygame, "K_PLUS", None)
            if k_plus is not None:
                keymap[k_plus] = ord("+")
            k_underscore = getattr(self._pygame, "K_UNDERSCORE", None)
            if k_underscore is not None:
                keymap[k_underscore] = ord("_")

            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    return ord("q")
                if event.type == self._pygame.KEYDOWN:
                    return keymap.get(event.key, -1)
            return -1

        return cv2.waitKey(1) & 0xFF

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
        if self._headless:
            return

        # Determine target display size.
        target_w, target_h = 0, 0

        # In fullscreen, trust detected physical screen size first.
        if self._backend == "pygame" and self._pg_screen is not None:
            target_w, target_h = self._pg_screen.get_size()
        elif self.fullscreen and self._screen_size is not None:
            target_w, target_h = self._screen_size
        else:
            # Read current window drawable size when available.
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
                if win_w > 0 and win_h > 0:
                    target_w, target_h = win_w, win_h
            except Exception:
                pass

        # Fallback to 2x internal resolution if size is still unknown.
        fallback_w = frame.shape[1] * 2
        fallback_h = frame.shape[0] * 2
        if target_w <= 0 or target_h <= 0:
            target_w, target_h = fallback_w, fallback_h

        if self.fullscreen:
            # In fullscreen mode, fill the entire screen (no letterboxing).
            display = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            # In windowed mode, preserve aspect ratio.
            display = scale_to_fit(frame, target_w, target_h)

        scale_x = display.shape[1] / max(frame.shape[1], 1)
        scale_y = display.shape[0] / max(frame.shape[0], 1)

        # Optional face bounding box
        if self.show_face and face is not None and face.detected:
            x = int(face.x * scale_x)
            y = int(face.y * scale_y)
            w = int(face.w * scale_x)
            h = int(face.h * scale_y)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Overlays
        if self.show_fps:
            draw_fps(display, fps)
        if effect_name:
            draw_effect_name(display, effect_name)

        # Letterbox/pillarbox only in windowed mode.
        if (not self.fullscreen) and (
            display.shape[1] != target_w or display.shape[0] != target_h
        ):
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y0 = (target_h - display.shape[0]) // 2
            x0 = (target_w - display.shape[1]) // 2
            canvas[y0:y0 + display.shape[0], x0:x0 + display.shape[1]] = display
            cv2.imshow(WINDOW_NAME, canvas)
            return

        if self._backend == "pygame" and self._pygame is not None and self._pg_screen is not None:
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            surface = self._pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            self._pg_screen.blit(surface, (0, 0))
            self._pygame.display.flip()
            return

        cv2.imshow(WINDOW_NAME, display)

    def toggle_fullscreen(self) -> None:
        """Toggle between fullscreen and windowed mode."""
        if self._headless:
            return

        if self._backend == "pygame" and self._pygame is not None:
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                self._pg_screen = self._pygame.display.set_mode((0, 0), self._pygame.FULLSCREEN)
            else:
                self._pg_screen = self._pygame.display.set_mode((640, 480), self._pygame.RESIZABLE)
            size = self._pg_screen.get_size()
            self._screen_size = (int(size[0]), int(size[1]))
            logger.info("Fullscreen toggled → %s", self.fullscreen)
            return

        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            if self._screen_size is not None:
                sw, sh = self._screen_size
                cv2.resizeWindow(WINDOW_NAME, sw, sh)
        else:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
            )
        logger.info("Fullscreen toggled → %s", self.fullscreen)

    def destroy(self) -> None:
        """Close the display window."""
        if self._headless:
            return

        if self._backend == "pygame" and self._pygame is not None:
            self._pygame.quit()
            logger.info("Renderer destroyed")
            return

        cv2.destroyWindow(WINDOW_NAME)
        logger.info("Renderer destroyed")
