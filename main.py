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
from input import SourceManager
from face_detection import FaceInfo, build_detector
from audio_reactivity import AudioFeatures
from effects import EFFECTS, EFFECT_ORDER
from effects.base import BaseEffect
from renderer import Renderer
from utils import FPSCounter, smooth
from midi.mapping import MidiState

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
        self._midi: Optional[MidiState] = None
        self._audio_enabled: bool = cfg.audio
        self._face_enabled: bool = not cfg.no_face
        self._midi_enabled: bool = cfg.midi_enabled
        self._running: bool = False
        self._dropped_frames: int = 0

        # Sub-systems (initialised in run())
        self._source_manager: Optional[SourceManager] = None
        self._detector = None
        self._audio_reactor = None
        self._midi_manager = None
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

    def _apply_midi_params(self, state: MidiState) -> None:
        """Apply MIDI-controlled parameters to all loaded effects."""
        for effect in self._effects.values():
            effect.strength = state.master_intensity
            effect.warp_amount = state.warp_amount
            effect.trail_decay = state.feedback_decay

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_source(self) -> None:
        self._source_manager = SourceManager(self.cfg)
        if not self._source_manager.initialize():
            raise RuntimeError("No valid input source available. "
                               "Connect a webcam or insert a USB drive with video files.")

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

    def _init_midi(self) -> None:
        if not self._midi_enabled:
            return
        try:
            from midi import MidiManager
            self._midi_manager = MidiManager(device_name=self.cfg.midi_device)
            self._midi_manager.open()
            logger.info("MIDI controller connected: %s", self.cfg.midi_device)
        except ImportError:
            logger.warning(
                "mido / python-rtmidi not installed. "
                "Install with: pip install mido python-rtmidi. "
                "Continuing without MIDI."
            )
            self._midi_enabled = False
            self._midi_manager = None
        except RuntimeError as exc:
            logger.warning(
                "MIDI unavailable: %s. Continuing without MIDI.", exc
            )
            self._midi_enabled = False
            self._midi_manager = None

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
        logger.info("Config: resolution=%dx%d, effect=%s, prefer_camera=%s",
                    self.cfg.width, self.cfg.height, self.cfg.effect,
                    self.cfg.prefer_camera)

        # Initialise sub-systems
        try:
            self._init_source()
        except RuntimeError as exc:
            logger.error("%s", exc)
            sys.exit(1)

        self._init_face_detector()
        self._init_audio()
        self._init_midi()
        self._init_renderer()

        logger.info(
            "Face detection: %s | Audio: %s | MIDI: %s",
            "ON" if self._face_enabled else "OFF",
            "ON" if self._audio_enabled else "OFF",
            "ON" if self._midi_enabled else "OFF",
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
            ok, frame = self._source_manager.read()
            if not ok or frame is None:
                self._dropped_frames += 1
                logger.warning("Dropped frame from source: %s",
                               self._source_manager.source_name)

                # Keep GUI alive with a visible diagnostic instead of black screen.
                if self._renderer is not None and self._dropped_frames >= 1:
                    status = self._status_frame(
                        f"No frames — SOURCE: {self._source_manager.source_name}"
                    )
                    fps = self._fps_counter.tick()
                    self._renderer.show(
                        status,
                        fps=fps,
                        effect_name="source error",
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

            # 4. MIDI parameter state
            if self._midi_enabled and self._midi_manager is not None:
                self._midi = self._midi_manager.get_state()
                self._apply_midi_params(self._midi)

            # 4. Apply effect
            processed = self._current_effect.apply(
                frame, face=self._face, audio=self._audio
            )

            # 5. Render
            fps = self._fps_counter.tick()
            source_label = f"SOURCE: {self._source_manager.source_name}"
            self._renderer.show(
                processed,
                fps=fps,
                effect_name=f"{self._current_effect.name} | {source_label}",
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
        if self._source_manager:
            self._source_manager.release()
        if self._audio_reactor:
            self._audio_reactor.close()
        if self._midi_manager:
            self._midi_manager.close()
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
