"""
modes/geiss_mode.py — Geiss-inspired visual mode for DogDayDiffuser.

Captures the feel of Ryan Geiss' classic CPU-era Winamp visualizer:
  - hypnotic, fluid, tunnel/plasma/warp-oriented
  - feedback distortion with zoom/rotation/drift
  - radial tunnel warp around face or frame centre
  - plasma-style colour cycling via sine waves
  - optional rotational symmetry
  - motion smearing via frame differencing

Performance:
  All operations are vectorised NumPy / OpenCV — no per-pixel Python loops.
  Target: usable on Raspberry Pi 3 at 320×240.

Signal mapping:
    audio_bass    → zoom pulse / tunnel depth
    audio_mid     → swirl/tunnel warp amount
    audio_treble  → plasma shimmer / colour cycling speed
    beat_pulse    → momentary flash / feedback decay boost
    motion        → feedback intensity
    face_center   → warp/tunnel centre
    face_size     → distortion radius
"""

from __future__ import annotations

import math
import time
from typing import Dict, Any, Optional

import cv2
import numpy as np

from .base_mode import VisualMode


class GeissMode(VisualMode):
    """Geiss-style software visualizer mode."""

    name = "GEISS"
    subtitle = ""

    def __init__(
        self,
        use_symmetry: bool = True,
        plasma_overlay: bool = True,
        symmetry_count: int = 4,
        feedback_decay: float = 0.88,
        base_zoom: float = 1.015,
    ):
        self.use_symmetry = use_symmetry
        self.plasma_overlay = plasma_overlay
        self.symmetry_count = symmetry_count
        self.feedback_decay = feedback_decay
        self.base_zoom = base_zoom

        # Internal state
        self._buffer: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._phase: float = 0.0         # plasma colour phase
        self._time: float = 0.0          # total elapsed time

        # Cached warp maps (rebuilt when params change)
        self._warp_map_x: Optional[np.ndarray] = None
        self._warp_map_y: Optional[np.ndarray] = None
        self._warp_params: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._buffer = None
        self._prev_frame = None
        self._phase = 0.0
        self._time = 0.0
        self._warp_map_x = None
        self._warp_map_y = None
        self._warp_params = None

    def update(self, dt: float, signals: Dict[str, Any]) -> None:
        treble = float(self._sig(signals, "audio_treble", 0.0))
        # Advance phase faster with treble energy
        self._phase += dt * (1.2 + treble * 3.0)
        self._time += dt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tunnel_maps(
        self, h: int, w: int, cx: float, cy: float, strength: float
    ) -> None:
        """Compute radial tunnel / swirl warp displacement maps."""
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xs - cx
        dy = ys - cy
        dist = np.hypot(dx, dy)
        max_r = float(math.hypot(w, h)) / 2.0

        norm_dist = dist / max(max_r, 1.0)

        # Tunnel pull: pixels near centre are pulled inward
        tunnel_strength = strength * 0.35
        pull = tunnel_strength * np.clip(1.0 - norm_dist, 0.0, 1.0) ** 2
        src_x = xs - dx * pull
        src_y = ys - dy * pull

        # Add sine-driven swirl on top
        angle = np.arctan2(dy, dx)
        swirl_falloff = np.clip(1.0 - norm_dist, 0.0, 1.0)
        swirl_angle = strength * 0.4 * swirl_falloff ** 1.5
        cos_s = np.cos(swirl_angle)
        sin_s = np.sin(swirl_angle)
        rdx = dx * cos_s - dy * sin_s
        rdy = dx * sin_s + dy * cos_s
        src_x = cx + rdx - (rdx - (src_x - cx)) * 0.0
        src_y = cy + rdy - (rdy - (src_y - cy)) * 0.0

        # Ripple
        ripple = np.sin(norm_dist * math.pi * 5 + self._phase) * strength * 8.0
        src_x += ripple * (dx / (dist + 1e-6))
        src_y += ripple * (dy / (dist + 1e-6))

        self._warp_map_x = src_x.astype(np.float32)
        self._warp_map_y = src_y.astype(np.float32)
        self._warp_params = (h, w, cx, cy, strength)

    def _apply_tunnel_warp(
        self, frame: np.ndarray, cx: float, cy: float, strength: float
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        params = (h, w, cx, cy, strength)
        if self._warp_params != params:
            self._build_tunnel_maps(h, w, cx, cy, strength)
        return cv2.remap(
            frame,
            self._warp_map_x,
            self._warp_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _plasma_overlay(
        self, h: int, w: int, phase: float, treble: float
    ) -> np.ndarray:
        """Generate a cheap sine-based plasma colour layer."""
        xs = np.linspace(0.0, math.pi * 4, w, dtype=np.float32)
        ys = np.linspace(0.0, math.pi * 4, h, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        p = phase + treble * 0.5
        r = np.sin(xx + p) * 0.5 + 0.5
        g = np.sin(yy * 0.7 + p * 1.3) * 0.5 + 0.5
        b = np.sin((xx + yy) * 0.5 + p * 0.8) * 0.5 + 0.5

        plasma = np.stack([b, g, r], axis=2)  # BGR
        return (plasma * 255).astype(np.uint8)

    def _apply_symmetry(
        self, frame: np.ndarray, n: int
    ) -> np.ndarray:
        """Apply n-fold rotational symmetry (kaleidoscope-lite)."""
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

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render(
        self, frame: np.ndarray, signals: Dict[str, Any]
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        cx_default, cy_default = w / 2.0, h / 2.0

        # Extract signals
        bass     = float(self._sig(signals, "audio_bass", 0.0))
        mid      = float(self._sig(signals, "audio_mid", 0.0))
        treble   = float(self._sig(signals, "audio_treble", 0.0))
        beat     = float(self._sig(signals, "beat_pulse", 0.0))
        motion   = float(self._sig(signals, "motion", 0.0))
        fc       = self._sig(signals, "face_center", (cx_default, cy_default))
        fsize    = float(self._sig(signals, "face_size", 0.0))

        face_cx = float(fc[0]) if fc else cx_default
        face_cy = float(fc[1]) if fc else cy_default

        # Warp centre: lerp between frame centre and face centre
        face_count = int(self._sig(signals, "face_count", 0))
        if face_count > 0:
            cx = cx_default + (face_cx - cx_default) * 0.5
            cy = cy_default + (face_cy - cy_default) * 0.5
        else:
            cx, cy = cx_default, cy_default

        # ---- 1. Motion smear ----
        if self._prev_frame is not None and self._prev_frame.shape == frame.shape:
            diff = cv2.absdiff(frame, self._prev_frame)
            motion_mask = (diff.astype(np.float32) / 255.0) * (0.3 + motion * 0.4)
            smeared = cv2.addWeighted(frame, 1.0, diff, 0.3 + motion * 0.4, 0)
        else:
            smeared = frame.copy()
        self._prev_frame = frame.copy()

        # ---- 2. Tunnel warp ----
        warp_strength = 0.4 + mid * 0.6 + bass * 0.3
        warped = self._apply_tunnel_warp(smeared, cx, cy, warp_strength)

        # ---- 3. Feedback buffer blend ----
        if self._buffer is None or self._buffer.shape[:2] != (h, w):
            self._buffer = warped.astype(np.float32)

        zoom = self.base_zoom + bass * 0.04
        rot_deg = 0.3 + mid * 0.5
        decay = self.feedback_decay + motion * 0.05 - beat * 0.1
        decay = float(np.clip(decay, 0.5, 0.97))

        M = cv2.getRotationMatrix2D((cx, cy), rot_deg, zoom)
        buf_warped = cv2.warpAffine(
            self._buffer, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        blended = warped.astype(np.float32) * (1.0 - decay) + buf_warped * decay
        blended = np.clip(blended, 0, 255)
        self._buffer = blended

        result = blended.astype(np.uint8)

        # ---- 4. Plasma overlay ----
        if self.plasma_overlay:
            plasma = self._plasma_overlay(h, w, self._phase, treble)
            overlay_strength = 0.12 + treble * 0.18 + beat * 0.1
            overlay_strength = float(np.clip(overlay_strength, 0.0, 0.4))
            result = cv2.addWeighted(
                result, 1.0 - overlay_strength,
                plasma, overlay_strength,
                0,
            )

        # ---- 5. Symmetry ----
        if self.use_symmetry and self.symmetry_count >= 2:
            result = self._apply_symmetry(result, self.symmetry_count)

        return result
