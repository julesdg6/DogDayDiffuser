"""
main.py — DogDayDiffuser entry point.

Startup sequence:
  1. Parse config (CLI flags + optional JSON file)
  2. Open webcam
  3. Initialise face detector (optional)
  4. Initialise audio reactor (optional)
  5. Instantiate effect objects
  6. Run main loop
  7. Clean up on exit

Keyboard shortcuts:
  q / Esc   — quit
  f         — toggle fullscreen
  1–4       — select effect directly
  space     — cycle to next effect
  a         — toggle audio reactivity
  d         — toggle face detection
  + / =     — increase trail strength
  - / _     — decrease trail strength
  w         — increase warp amount
  s         — decrease warp amount
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, Optional

import cv2
import numpy as np

from config import parse_args, AppConfig, EFFECT_NAMES
from camera import Camera
from face_detection import FaceInfo, build_detector
from audio_reactivity import AudioFeatures
from effects import EFFECTS, EFFECT_ORDER
from effects.base import BaseEffect
from renderer import Renderer
from utils import FPSCounter, smooth

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------


class DogDayDiffuser:
    """Main application controller."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        # Active effect
        self._effect_name: str = cfg.effect
        self._effects: Dict[str, BaseEffect] = {
            name: cls() for name, cls in EFFECTS.items()
        }

        # State
        self._face: Optional[FaceInfo] = None
        self._face_frame_count: int = 0
        self._audio: Optional[AudioFeatures] = None
        self._audio_enabled: bool = cfg.audio
        self._face_enabled: bool = not cfg.no_face
        self._running: bool = False
        self._dropped_frames: int = 0

        # Sub-systems (initialised in run())
        self._camera: Optional[Camera] = None
        self._detector = None
        self._audio_reactor = None
        self._renderer: Optional[Renderer] = None
        self._fps_counter = FPSCounter()

    # ------------------------------------------------------------------
    # Property helpers
    # ------------------------------------------------------------------

    @property
    def _current_effect(self) -> BaseEffect:
        return self._effects[self._effect_name]

    def _next_effect(self) -> None:
        idx = EFFECT_ORDER.index(self._effect_name)
        self._effect_name = EFFECT_ORDER[(idx + 1) % len(EFFECT_ORDER)]
        logger.info("Effect → %s", self._effect_name)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_camera(self) -> None:
        self._camera = Camera(
            index=self.cfg.camera,
            width=self.cfg.width,
            height=self.cfg.height,
        )
        self._camera.open()

    def _init_face_detector(self) -> None:
        if not self._face_enabled:
            logger.info("Face detection disabled by config")
            return
        try:
            self._detector = build_detector(
                use_openvino=self.cfg.use_openvino,
            )
        except Exception as exc:
            logger.warning("Face detector failed to initialise: %s", exc)
            self._face_enabled = False

    def _init_audio(self) -> None:
        if not self._audio_enabled:
            return
        try:
            from audio_reactivity import AudioReactor
            self._audio_reactor = AudioReactor(device=self.cfg.audio_device)
        except Exception as exc:
            logger.warning(
                "Audio reactivity unavailable: %s. Continuing without audio.", exc
            )
            self._audio_enabled = False
            self._audio_reactor = None

    def _init_renderer(self) -> None:
        self._renderer = Renderer(fullscreen=self.cfg.fullscreen)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _status_frame(self, message: str) -> np.ndarray:
        """Build a simple status frame at internal resolution."""
        frame = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "DogDayDiffuser",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            message,
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Tip: verify camera cable/index and rpicam sees sensor",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return frame

    def run(self) -> None:
        """Open all resources and run the main processing loop."""
        logger.info("DogDayDiffuser starting up")
        logger.info("Config: camera=%d, resolution=%dx%d, effect=%s",
                    self.cfg.camera, self.cfg.width, self.cfg.height, self.cfg.effect)

        # Initialise sub-systems
        try:
            self._init_camera()
        except RuntimeError as exc:
            logger.error("%s", exc)
            sys.exit(1)

        self._init_face_detector()
        self._init_audio()
        self._init_renderer()

        logger.info(
            "Face detection: %s | Audio: %s",
            "ON" if self._face_enabled else "OFF",
            "ON" if self._audio_enabled else "OFF",
        )

        self._running = True

        try:
            self._loop()
        finally:
            self._cleanup()

    def _loop(self) -> None:
        """Inner main loop."""
        frame_idx = 0

        while self._running:
            # 1. Capture frame
            frame = self._camera.read()
            if frame is None:
                self._dropped_frames += 1
                logger.warning("Dropped frame from camera")

                # Keep GUI alive with a visible diagnostic instead of black screen.
                if self._renderer is not None and self._dropped_frames >= 1:
                    status = self._status_frame("No camera frames received")
                    fps = self._fps_counter.tick()
                    self._renderer.show(
                        status,
                        fps=fps,
                        effect_name="camera error",
                        face=None,
                    )
                    key = self._renderer.poll_key()
                    if not self._handle_key(key):
                        break
                continue
            self._dropped_frames = 0

            # 2. Face detection (every N frames)
            if self._face_enabled and self._detector is not None:
                if frame_idx % self.cfg.detection_interval == 0:
                    detected = self._detector.detect(frame)
                    if detected is not None:
                        self._face = detected
                    # Keep last known face (no reset on miss — smoother UX)

            # 3. Audio features
            if self._audio_enabled and self._audio_reactor is not None:
                self._audio = self._audio_reactor.get_features()

            # 4. Apply effect
            processed = self._current_effect.apply(
                frame, face=self._face, audio=self._audio
            )

            # 5. Render
            fps = self._fps_counter.tick()
            self._renderer.show(
                processed,
                fps=fps,
                effect_name=self._current_effect.name,
                face=self._face,
            )

            # 6. Handle keyboard
            key = -1
            if self._renderer is not None:
                key = self._renderer.poll_key()
            if not self._handle_key(key):
                break

            frame_idx += 1

    def _handle_key(self, key: int) -> bool:
        """Process a keypress.  Returns False to signal quit."""
        if key in (ord("q"), 27):  # q or Esc
            logger.info("Quit requested")
            return False

        elif key == ord("f"):
            self._renderer.toggle_fullscreen()

        elif key == ord(" "):
            self._next_effect()

        elif key == ord("1"):
            self._effect_name = "kaleidoscope"

        elif key == ord("2"):
            self._effect_name = "feedback"

        elif key == ord("3"):
            self._effect_name = "warp"

        elif key == ord("4"):
            self._effect_name = "color"

        elif key == ord("a"):
            if self._audio_reactor is not None:
                self._audio_enabled = not self._audio_enabled
                logger.info("Audio reactivity: %s",
                            "ON" if self._audio_enabled else "OFF")
            else:
                logger.info("Audio reactor not available")

        elif key == ord("d"):
            self._face_enabled = not self._face_enabled
            logger.info("Face detection: %s",
                        "ON" if self._face_enabled else "OFF")

        elif key in (ord("+"), ord("=")):
            for e in self._effects.values():
                e.trail_decay = min(e.trail_decay + 0.03, 0.98)
            logger.info("Trail decay → %.2f", self._current_effect.trail_decay)

        elif key in (ord("-"), ord("_")):
            for e in self._effects.values():
                e.trail_decay = max(e.trail_decay - 0.03, 0.0)
            logger.info("Trail decay → %.2f", self._current_effect.trail_decay)

        elif key == ord("w"):
            for e in self._effects.values():
                e.warp_amount = min(e.warp_amount + 0.05, 2.0)
            logger.info("Warp amount → %.2f", self._current_effect.warp_amount)

        elif key == ord("s"):
            for e in self._effects.values():
                e.warp_amount = max(e.warp_amount - 0.05, 0.0)
            logger.info("Warp amount → %.2f", self._current_effect.warp_amount)

        return True

    # ------------------------------------------------------------------
    # Clean-up
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        logger.info("Shutting down…")
        if self._camera:
            self._camera.release()
        if self._audio_reactor:
            self._audio_reactor.close()
        if self._detector:
            self._detector.close()
        if self._renderer:
            self._renderer.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    app = DogDayDiffuser(cfg)
    app.run()


if __name__ == "__main__":
    main()
