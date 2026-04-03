"""
main.py — DogDayDiffuser entry point.

Startup sequence:
  1. Parse config (CLI flags + optional JSON file)
  2. Open webcam
  3. Initialise face detector (optional)
  4. Initialise audio reactor (optional)
  5. Instantiate effect and mode objects
  6. Run main loop
  7. Clean up on exit

Keyboard shortcuts:
  q / Esc   — quit
  f         — toggle fullscreen
  1–4       — select effect directly
  space     — cycle to next effect
  g         — switch to Geiss mode
  m         — switch to MilkDrop mode
  e         — exit visual mode (return to effects)
  [         — previous MilkDrop preset
  ]         — next MilkDrop preset
  r         — reset current mode
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
import time
from typing import Dict, Optional

import cv2
import numpy as np

from config import parse_args, AppConfig, EFFECT_NAMES
from input import SourceManager
from face_detection import FaceInfo, build_detector
from audio_reactivity import AudioFeatures
from effects import EFFECTS, EFFECT_ORDER
from effects.base import BaseEffect
from modes import MODES, build_signals
from modes.base_mode import VisualMode
from modes.milkdrop_mode import MilkDropMode
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

        # Active visual mode (None = use effects instead)
        self._mode: Optional[VisualMode] = None
        self._modes: Dict[str, VisualMode] = self._build_modes(cfg)
        if cfg.default_mode and cfg.default_mode in self._modes:
            self._mode = self._modes[cfg.default_mode]
            self._mode.reset()
            logger.info("Starting in visual mode: %s", cfg.default_mode)

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
        self._last_frame_time: float = time.monotonic()
        self._prev_gray: Optional[np.ndarray] = None   # for motion estimation

        # Sub-systems (initialised in run())
        self._source_manager: Optional[SourceManager] = None
        self._detector = None
        self._audio_reactor = None
        self._midi_manager = None
        self._renderer: Optional[Renderer] = None
        self._fps_counter = FPSCounter()

    @staticmethod
    def _build_modes(cfg: AppConfig) -> Dict[str, VisualMode]:
        """Instantiate all visual modes with config-driven parameters."""
        geiss = MODES["geiss"](
            use_symmetry=cfg.geiss_use_symmetry,
            plasma_overlay=cfg.geiss_plasma_overlay,
        )
        milkdrop = MODES["milkdrop"](
            auto_cycle=cfg.milkdrop_auto_cycle,
            cycle_seconds=cfg.milkdrop_cycle_seconds,
            beat_transition=cfg.milkdrop_beat_transition,
            allow_face_modulation=cfg.mode_allow_face_modulation,
            allow_audio_modulation=cfg.mode_allow_audio_modulation,
        )
        return {"geiss": geiss, "milkdrop": milkdrop}

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
            from audio.audio_manager import AudioManager
            self._audio_reactor = AudioManager(
                audio_device=self.cfg.audio_device,
                auto_usb=self.cfg.audio_auto_usb,
                prefer_usb=self.cfg.audio_prefer_usb,
                output_prefer_hdmi=self.cfg.audio_output_prefer_hdmi,
                enable_passthrough=self.cfg.audio_enable_passthrough,
                sample_rate=self.cfg.audio_sample_rate,
                buffer_size=self.cfg.audio_buffer_size,
            )
            logger.info(
                "Audio IN: %s | Audio OUT: %s",
                self._audio_reactor.input_label,
                self._audio_reactor.output_label,
            )
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
            self._show_splash()
            if self._running:
                self._loop()
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Splash screen
    # ------------------------------------------------------------------

    def _collect_hw_info(self) -> list:
        """Build hardware detection rows for the splash console.

        Called after all sub-systems have been initialised so that actual
        device names and states can be reported accurately.
        """
        import platform as _platform
        rows = []

        # System
        try:
            rows.append(("OS", f"{_platform.system()} {_platform.release()}"[:42], "val"))
        except Exception:
            rows.append(("OS", "unknown", "warn"))
        try:
            cpu = (_platform.processor() or _platform.machine() or "unknown")[:42]
            rows.append(("CPU", cpu, "val"))
        except Exception:
            rows.append(("CPU", "unknown", "warn"))

        # Input source
        if self._source_manager is not None:
            rows.append(("Input", self._source_manager.source_name[:42], "ok"))
        else:
            rows.append(("Input", "not available", "err"))

        # Face detection
        if self._face_enabled and self._detector is not None:
            det_type = type(self._detector).__name__
            rows.append(("Face Det.", det_type, "ok"))
        elif not self._face_enabled:
            rows.append(("Face Det.", "disabled", "warn"))
        else:
            rows.append(("Face Det.", "init failed", "err"))

        # Audio
        if self._audio_enabled and self._audio_reactor is not None:
            in_lbl = getattr(self._audio_reactor, "input_label", "active")
            rows.append(("Audio IN", in_lbl[:42], "ok"))
            out_lbl = getattr(self._audio_reactor, "output_label", "")
            if out_lbl:
                rows.append(("Audio OUT", out_lbl[:42], "ok"))
        else:
            rows.append(("Audio", "disabled", "warn"))

        # MIDI
        if self._midi_enabled and self._midi_manager is not None:
            rows.append(("MIDI", self.cfg.midi_device[:42], "ok"))
        else:
            rows.append(("MIDI", "disabled", "warn"))

        # OpenCV version
        rows.append(("OpenCV", cv2.__version__, "val"))

        # Starting effect
        rows.append(("Effect", self._effect_name, "val"))

        return rows

    def _show_splash(self) -> None:
        """Display the animated splash screen before the main loop.

        Shows the King Charles Cavalier palette-animated fractal in the top
        two thirds of the frame and a colourful hardware-detection console in
        the bottom third.  Runs for ``SplashScreen.DURATION`` seconds then
        returns automatically.  The user can skip with SPACE or quit with Q /
        Esc (which also sets ``_running = False`` so the main loop is skipped).
        """
        from splash import SplashScreen

        if self._renderer is None or self._renderer.is_headless:
            # No display — just log and continue (no visual delay).
            logger.info("Splash: headless mode, skipping visual splash")
            return

        hw_entries = self._collect_hw_info()
        splash = SplashScreen(
            width=self.cfg.width,
            height=self.cfg.height,
            hw_entries=hw_entries,
        )

        logger.info("Splash: starting (%.0f s)", SplashScreen.DURATION)
        start = time.monotonic()
        while self._running:
            t = time.monotonic() - start
            if t >= SplashScreen.DURATION:
                break
            frame = splash.render(t)
            self._renderer.show(frame, fps=0.0, effect_name="")
            key = self._renderer.poll_key()
            if key in (ord("q"), 27):   # quit entirely
                self._running = False
                break
            if key == ord(" "):         # skip to main loop
                break
        logger.info("Splash: done")

    def _loop(self) -> None:
        """Inner main loop."""
        frame_idx = 0

        while self._running:
            # Track frame delta time
            now = time.monotonic()
            dt = now - self._last_frame_time
            self._last_frame_time = now

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

            # 5. Estimate inter-frame motion
            motion = self._estimate_motion(frame)

            # 6. Build signals dict for visual modes
            h, w = frame.shape[:2]
            fps_now = self._fps_counter.fps
            signals = build_signals(
                audio=self._audio,
                face=self._face,
                fps=fps_now,
                source_name=self._source_manager.source_name,
                frame_w=w,
                frame_h=h,
                motion=motion,
            )

            # 7. Apply visual mode or effect
            if self._mode is not None:
                self._mode.update(dt, signals)
                processed = self._mode.render(frame, signals)
                overlay_label = f"{self._mode.name}"
                if hasattr(self._mode, "subtitle") and self._mode.subtitle:
                    overlay_label += f" / {self._mode.subtitle}"
                source_label = f"SOURCE: {self._source_manager.source_name}"
                overlay_label += f" | {source_label}"
            else:
                processed = self._current_effect.apply(
                    frame, face=self._face, audio=self._audio
                )
                source_label = f"SOURCE: {self._source_manager.source_name}"
                overlay_label = f"{self._current_effect.name} | {source_label}"

            # 8. Render
            fps = self._fps_counter.tick()
            self._renderer.show(
                processed,
                fps=fps,
                effect_name=overlay_label,
                face=self._face,
            )

            # 9. Handle keyboard
            key = -1
            if self._renderer is not None:
                key = self._renderer.poll_key()
            if not self._handle_key(key):
                break

            frame_idx += 1

    def _estimate_motion(self, frame: np.ndarray) -> float:
        """Return normalised inter-frame motion magnitude 0–1."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        motion = float(np.mean(diff)) / 255.0
        return min(motion * 6.0, 1.0)  # scale up for typical scenes

    def _ensure_milkdrop_mode(self) -> None:
        """Switch to MilkDrop mode if not already active."""
        if self._mode is not self._modes.get("milkdrop"):
            self._mode = self._modes["milkdrop"]
            logger.info("Switched to MilkDrop mode for preset cycling")


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
            self._mode = None

        elif key == ord("2"):
            self._effect_name = "feedback"
            self._mode = None

        elif key == ord("3"):
            self._effect_name = "warp"
            self._mode = None

        elif key == ord("4"):
            self._effect_name = "color"
            self._mode = None

        elif key == ord("g"):
            self._mode = self._modes["geiss"]
            self._mode.reset()
            logger.info("Visual mode → GEISS")

        elif key == ord("m"):
            self._mode = self._modes["milkdrop"]
            self._mode.reset()
            logger.info("Visual mode → MILKDROP")

        elif key == ord("e"):
            self._mode = None
            logger.info("Visual mode → OFF (effect: %s)", self._effect_name)

        elif key == ord("r"):
            if self._mode is not None:
                self._mode.reset()
                logger.info("Mode reset: %s", self._mode.name)

        elif key == ord("]"):
            md = self._modes.get("milkdrop")
            if md is not None and isinstance(md, MilkDropMode):
                md.cycle_preset(direction=1)
                logger.info("MilkDrop preset → next")
            self._ensure_milkdrop_mode()

        elif key == ord("["):
            md = self._modes.get("milkdrop")
            if md is not None and isinstance(md, MilkDropMode):
                md.cycle_preset(direction=-1)
                logger.info("MilkDrop preset → previous")
            self._ensure_milkdrop_mode()

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
