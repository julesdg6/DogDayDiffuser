"""
modes/milkdrop_mode.py — MilkDrop-inspired visual mode for DogDayDiffuser.

Inspired by Geiss' later GPU-oriented successor MilkDrop, which was famous
for its preset-driven, audio-reactive rendered scenes and smooth transitions.

This is an *inspired-by* implementation — not a port of the actual MilkDrop
scripting engine or preset format.

Features:
  - Internal preset library (Python-defined parameter sets)
  - Smooth parameter interpolation between presets
  - Feedback compositor with configurable decay/zoom/rotation
  - Waveform overlays (ring, spokes, horizontal)
  - Shape overlays (circles, star bursts, orbiting dots)
  - Beat-triggered and timer-based preset transitions
  - Named colour palettes

Performance:
  All operations use NumPy / OpenCV only — no per-pixel Python loops.
  Target: usable on Raspberry Pi 3 at 320×240.

Signal mapping:
    audio_level   → general animation strength
    audio_bass    → large-scale pulsing and zoom
    audio_mid     → waveform amplitude
    audio_treble  → sparkle, edge brightness, detail motion
    beat_pulse    → preset transition trigger or strobe accent
    motion        → transition energy multiplier
    face_center   → symmetry / centre offset
    face_size     → bloom radius or shape scale
"""

from __future__ import annotations

import math
import time
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

from .base_mode import VisualMode


# ---------------------------------------------------------------------------
# Named colour palettes  (BGR tuples for OpenCV)
# ---------------------------------------------------------------------------

PALETTES: Dict[str, List[Tuple[int, int, int]]] = {
    "neon_fire":      [(0, 0, 255), (0, 64, 255), (0, 128, 255), (0, 200, 200), (200, 200, 0)],
    "acid_green":     [(0, 255, 0), (0, 220, 100), (0, 180, 200), (0, 100, 220), (50, 255, 50)],
    "cyber_blue":     [(255, 100, 0), (200, 200, 0), (0, 255, 255), (0, 180, 255), (100, 100, 255)],
    "sunset_glow":    [(0, 50, 255), (0, 150, 255), (0, 200, 150), (50, 100, 200), (100, 0, 150)],
    "monochrome_ice": [(200, 220, 255), (180, 200, 240), (150, 180, 220), (100, 150, 200), (50, 100, 180)],
}


def _palette_color(
    palette_name: str, t: float
) -> Tuple[int, int, int]:
    """Interpolate a palette at position t ∈ [0, 1]."""
    colors = PALETTES.get(palette_name, PALETTES["neon_fire"])
    n = len(colors)
    scaled = t * (n - 1)
    lo = int(scaled) % n
    hi = (lo + 1) % n
    frac = scaled - int(scaled)
    b = int(colors[lo][0] * (1 - frac) + colors[hi][0] * frac)
    g = int(colors[lo][1] * (1 - frac) + colors[hi][1] * frac)
    r = int(colors[lo][2] * (1 - frac) + colors[hi][2] * frac)
    return (b, g, r)


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

PRESETS: List[Dict[str, Any]] = [
    {
        "name": "mandala_pulse",
        "symmetry": 8,
        "feedback_decay": 0.88,
        "zoom": 1.012,
        "rotation_speed": 0.15,
        "waveform_style": "ring",
        "shape_style": "orbit",
        "edge_gain": 1.2,
        "palette": "neon_fire",
        "transition_seconds": 18.0,
    },
    {
        "name": "wave_tunnel",
        "symmetry": 2,
        "feedback_decay": 0.90,
        "zoom": 1.020,
        "rotation_speed": 0.08,
        "waveform_style": "spokes",
        "shape_style": "circles",
        "edge_gain": 1.0,
        "palette": "cyber_blue",
        "transition_seconds": 15.0,
    },
    {
        "name": "starburst_echo",
        "symmetry": 6,
        "feedback_decay": 0.85,
        "zoom": 1.008,
        "rotation_speed": 0.25,
        "waveform_style": "horizontal",
        "shape_style": "star",
        "edge_gain": 1.4,
        "palette": "acid_green",
        "transition_seconds": 12.0,
    },
    {
        "name": "mirror_bloom",
        "symmetry": 4,
        "feedback_decay": 0.92,
        "zoom": 1.005,
        "rotation_speed": -0.10,
        "waveform_style": "ring",
        "shape_style": "circles",
        "edge_gain": 1.1,
        "palette": "sunset_glow",
        "transition_seconds": 20.0,
    },
    {
        "name": "bass_spiral",
        "symmetry": 3,
        "feedback_decay": 0.86,
        "zoom": 1.018,
        "rotation_speed": 0.40,
        "waveform_style": "spokes",
        "shape_style": "orbit",
        "edge_gain": 1.3,
        "palette": "monochrome_ice",
        "transition_seconds": 14.0,
    },
    {
        "name": "ghost_shapes",
        "symmetry": 1,
        "feedback_decay": 0.94,
        "zoom": 1.003,
        "rotation_speed": 0.05,
        "waveform_style": "horizontal",
        "shape_style": "star",
        "edge_gain": 0.9,
        "palette": "cyber_blue",
        "transition_seconds": 16.0,
    },
]

PRESET_NAMES = [p["name"] for p in PRESETS]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# MilkDrop mode
# ---------------------------------------------------------------------------


class MilkDropMode(VisualMode):
    """MilkDrop-inspired preset-driven layered visual mode."""

    name = "MILKDROP"

    def __init__(
        self,
        auto_cycle: bool = True,
        cycle_seconds: float = 15.0,
        beat_transition: bool = True,
        allow_face_modulation: bool = True,
        allow_audio_modulation: bool = True,
    ):
        self.auto_cycle = auto_cycle
        self.cycle_seconds = cycle_seconds
        self.beat_transition = beat_transition
        self.allow_face_modulation = allow_face_modulation
        self.allow_audio_modulation = allow_audio_modulation

        self._preset_idx: int = 0
        self._next_idx: int = 1
        self._interp_t: float = 1.0      # 0 = current, 1 = fully in next
        self._interp_active: bool = False
        self._time_since_last: float = 0.0
        self._phase: float = 0.0         # global animation phase
        self._rotation: float = 0.0      # accumulated feedback rotation

        # Feedback buffer
        self._buffer: Optional[np.ndarray] = None

        # Beat gate — avoid cascading transitions on sustained beats
        self._beat_gate: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def preset(self) -> Dict[str, Any]:
        return PRESETS[self._preset_idx]

    @property
    def next_preset(self) -> Dict[str, Any]:
        return PRESETS[self._next_idx]

    @property
    def subtitle(self) -> str:  # type: ignore[override]
        return f"PRESET: {self.preset['name'].upper()}"

    def current_preset_name(self) -> str:
        return PRESETS[self._preset_idx]["name"]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._buffer = None
        self._phase = 0.0
        self._rotation = 0.0
        self._time_since_last = 0.0
        self._interp_t = 1.0
        self._interp_active = False
        self._beat_gate = 0.0

    def cycle_preset(self, direction: int = 1) -> None:
        """Advance to the next (or previous) preset manually."""
        self._next_idx = (self._preset_idx + direction) % len(PRESETS)
        self._interp_t = 0.0
        self._interp_active = True

    def update(self, dt: float, signals: Dict[str, Any]) -> None:
        beat = float(self._sig(signals, "beat_pulse", 0.0))
        treble = float(self._sig(signals, "audio_treble", 0.0))
        motion = float(self._sig(signals, "motion", 0.0))

        self._phase += dt * (1.0 + treble * 2.0)

        # Decay beat gate
        self._beat_gate = max(0.0, self._beat_gate - dt * 2.0)

        # Advance time-based preset cycling
        self._time_since_last += dt
        trigger_secs = min(
            self.preset.get("transition_seconds", self.cycle_seconds),
            self.cycle_seconds,
        )

        should_transition = False
        if self.auto_cycle and self._time_since_last >= trigger_secs:
            should_transition = True
        if (
            self.beat_transition
            and beat > 0.75
            and self._beat_gate <= 0.0
            and self._time_since_last >= 5.0  # minimum dwell
        ):
            should_transition = True

        if should_transition and not self._interp_active:
            self._next_idx = (self._preset_idx + 1) % len(PRESETS)
            self._interp_t = 0.0
            self._interp_active = True
            self._beat_gate = 2.0  # suppress further triggers for 2 s

        # Advance interpolation
        if self._interp_active:
            interp_speed = 0.8 + motion * 0.4  # faster on motion
            self._interp_t = min(1.0, self._interp_t + dt * interp_speed)
            if self._interp_t >= 1.0:
                self._preset_idx = self._next_idx
                self._interp_t = 1.0
                self._interp_active = False
                self._time_since_last = 0.0

    # ------------------------------------------------------------------
    # Parameter interpolation
    # ------------------------------------------------------------------

    def _param(self, key: str, default: Any = 0.0) -> Any:
        """Return an interpolated parameter value between current and next preset."""
        cur_val = self.preset.get(key, default)
        if not self._interp_active:
            return cur_val
        nxt_val = self.next_preset.get(key, default)
        t = self._interp_t
        if isinstance(cur_val, (int, float)):
            return _lerp(float(cur_val), float(nxt_val), t)
        # Non-numeric: use current until halfway, then snap to next
        return nxt_val if t >= 0.5 else cur_val

    # ------------------------------------------------------------------
    # Visual sub-routines
    # ------------------------------------------------------------------

    def _feedback_pass(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        zoom: float,
        rot_deg: float,
        decay: float,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._buffer is None or self._buffer.shape[:2] != (h, w):
            self._buffer = frame.astype(np.float32)

        M = cv2.getRotationMatrix2D((cx, cy), rot_deg, zoom)
        warped_buf = cv2.warpAffine(
            self._buffer, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        blended = (
            frame.astype(np.float32) * (1.0 - decay) + warped_buf * decay
        )
        blended = np.clip(blended, 0, 255)
        self._buffer = blended
        return blended.astype(np.uint8)

    def _waveform_overlay(
        self,
        canvas: np.ndarray,
        style: str,
        mid: float,
        bass: float,
        phase: float,
        palette: str,
        fps: float,
    ) -> np.ndarray:
        """Draw an audio-reactive waveform overlay on *canvas* (in-place copy)."""
        out = canvas.copy()
        h, w = out.shape[:2]
        cx, cy = w // 2, h // 2
        amplitude = int((10 + mid * 40 + bass * 20))
        n_points = max(16, w // 4)
        color = _palette_color(palette, (phase * 0.1) % 1.0)

        if style == "ring":
            radius = min(w, h) // 4 + int(bass * 20)
            pts = []
            for i in range(n_points + 1):
                t_angle = 2 * math.pi * i / n_points
                r_wobble = radius + int(amplitude * math.sin(t_angle * 3 + phase))
                px = int(cx + r_wobble * math.cos(t_angle))
                py = int(cy + r_wobble * math.sin(t_angle))
                pts.append([px, py])
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(out, [pts_arr], isClosed=True, color=color, thickness=1,
                          lineType=cv2.LINE_AA)

        elif style == "spokes":
            n_spokes = 8
            max_len = min(w, h) // 3
            for i in range(n_spokes):
                t_angle = 2 * math.pi * i / n_spokes + phase * 0.2
                spoke_len = max_len + int(amplitude * math.sin(phase + i))
                ex = int(cx + spoke_len * math.cos(t_angle))
                ey = int(cy + spoke_len * math.sin(t_angle))
                cv2.line(out, (cx, cy), (ex, ey), color, 1, cv2.LINE_AA)

        else:  # "horizontal"
            y_base = cy
            pts = []
            for i in range(n_points):
                x = int(i * w / n_points)
                y = int(y_base + amplitude * math.sin(i * 0.4 + phase))
                pts.append([x, y])
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(out, [pts_arr], isClosed=False, color=color, thickness=1,
                          lineType=cv2.LINE_AA)

        return out

    def _shape_overlay(
        self,
        canvas: np.ndarray,
        style: str,
        beat: float,
        bass: float,
        phase: float,
        palette: str,
        face_center: Tuple[float, float],
        face_size: float,
    ) -> np.ndarray:
        """Draw beat-reactive geometric shapes on *canvas*."""
        out = canvas.copy()
        h, w = out.shape[:2]
        cx, cy = int(face_center[0]), int(face_center[1])
        base_r = int(min(w, h) * 0.12 + bass * 20 + beat * 15)

        color_t = (phase * 0.07) % 1.0
        color = _palette_color(palette, color_t)

        if style == "circles":
            for k in range(1, 4):
                r = base_r * k
                alpha = max(0, 1.0 - beat * 0.5 - k * 0.1)
                scaled_color = tuple(int(c * alpha) for c in color)
                cv2.circle(out, (cx, cy), r, scaled_color, 1, cv2.LINE_AA)

        elif style == "orbit":
            n_dots = 6
            orbit_r = base_r + int(bass * 15)
            for i in range(n_dots):
                angle = 2 * math.pi * i / n_dots + phase * 0.5
                dot_x = int(cx + orbit_r * math.cos(angle))
                dot_y = int(cy + orbit_r * math.sin(angle))
                dot_r = max(2, int(4 + beat * 6))
                cv2.circle(out, (dot_x, dot_y), dot_r, color, -1, cv2.LINE_AA)

        elif style == "star":
            n_points = 5
            outer_r = base_r + int(beat * 20)
            inner_r = outer_r // 2
            pts = []
            for i in range(n_points * 2):
                angle = math.pi * i / n_points - math.pi / 2 + phase * 0.1
                r = outer_r if i % 2 == 0 else inner_r
                px = int(cx + r * math.cos(angle))
                py = int(cy + r * math.sin(angle))
                pts.append([px, py])
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(out, [pts_arr], isClosed=True, color=color, thickness=1,
                          lineType=cv2.LINE_AA)

        return out

    def _apply_symmetry(self, frame: np.ndarray, n: int) -> np.ndarray:
        if n < 2:
            return frame
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xs - cx
        dy = ys - cy
        angle = np.arctan2(dy, dx)
        radius = np.hypot(dx, dy)
        sector = 2.0 * math.pi / n
        angle_folded = np.mod(angle, sector)
        mirror_mask = angle_folded > sector / 2.0
        angle_folded[mirror_mask] = sector - angle_folded[mirror_mask]
        src_x = (cx + radius * np.cos(angle_folded)).astype(np.float32)
        src_y = (cy + radius * np.sin(angle_folded)).astype(np.float32)
        return cv2.remap(
            frame, src_x, src_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _edge_enhance(self, frame: np.ndarray, gain: float) -> np.ndarray:
        if gain <= 1.0:
            return frame
        blurred = cv2.GaussianBlur(frame, (0, 0), 2)
        sharpened = cv2.addWeighted(frame, gain, blurred, -(gain - 1.0), 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render(
        self, frame: np.ndarray, signals: Dict[str, Any]
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        cx_default, cy_default = w / 2.0, h / 2.0

        # Extract signals
        audio_level = float(self._sig(signals, "audio_level", 0.0))
        bass        = float(self._sig(signals, "audio_bass", 0.0))
        mid         = float(self._sig(signals, "audio_mid", 0.0))
        treble      = float(self._sig(signals, "audio_treble", 0.0))
        beat        = float(self._sig(signals, "beat_pulse", 0.0))
        face_count  = int(self._sig(signals, "face_count", 0))
        fc          = self._sig(signals, "face_center", (cx_default, cy_default))
        face_size   = float(self._sig(signals, "face_size", 0.0))
        fps         = float(self._sig(signals, "fps", 25.0))

        face_cx = float(fc[0]) if fc else cx_default
        face_cy = float(fc[1]) if fc else cy_default
        if face_count == 0:
            face_cx, face_cy = cx_default, cy_default

        # Preset parameters (interpolated)
        decay        = float(self._param("feedback_decay", 0.88))
        zoom         = float(self._param("zoom", 1.01))
        rot_spd      = float(self._param("rotation_speed", 0.15))
        symmetry     = max(1, int(round(self._param("symmetry", 4))))
        edge_gain    = float(self._param("edge_gain", 1.0))
        palette      = str(self._param("palette", "neon_fire"))
        wave_style   = str(self._param("waveform_style", "ring"))
        shape_style  = str(self._param("shape_style", "orbit"))

        # Audio modulation of preset params
        if self.allow_audio_modulation:
            zoom  += bass * 0.025
            decay += audio_level * 0.02
            decay  = float(np.clip(decay, 0.5, 0.97))
            zoom   = float(np.clip(zoom, 0.99, 1.06))

        # Accumulate rotation
        self._rotation += rot_spd + bass * 0.3

        # ---- 1. Feedback compositor ----
        result = self._feedback_pass(
            frame,
            cx=face_cx if face_count > 0 else cx_default,
            cy=face_cy if face_count > 0 else cy_default,
            zoom=zoom,
            rot_deg=self._rotation % 360.0,
            decay=decay,
        )

        # ---- 2. Edge enhancement ----
        if edge_gain > 1.0:
            effective_gain = edge_gain + treble * 0.3
            result = self._edge_enhance(result, effective_gain)

        # ---- 3. Symmetry ----
        if symmetry >= 2:
            result = self._apply_symmetry(result, symmetry)

        # ---- 4. Waveform overlay ----
        result = self._waveform_overlay(
            result, wave_style, mid, bass, self._phase, palette, fps
        )

        # ---- 5. Shape overlay ----
        result = self._shape_overlay(
            result, shape_style, beat, bass, self._phase,
            palette, (face_cx, face_cy), face_size,
        )

        return result
