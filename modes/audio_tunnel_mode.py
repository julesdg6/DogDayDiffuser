"""
modes/audio_tunnel_mode.py — Audio-reactive tunnel mode for DogDayDiffuser.

Renders a forward-moving neon perspective tunnel with audio-reactive obstacle
blocks.  Inspired by music tunnel runners (e.g. AudioSurf aesthetic).

Features:
  - Perspective tunnel recedes into the distance and scrolls toward the viewer
  - Bass pulses the tunnel walls and spawns large centre blocks
  - Mids spawn side-lane obstacles
  - Treble adds sparkle particles and colour flicker
  - Beats trigger block spawns and momentary brightness flashes
  - Short-term audio history smooths spawning to feel intentional

Performance:
  All geometry is drawn with OpenCV line/rectangle primitives.
  No per-pixel Python loops.  Block list capped at MAX_BLOCKS.
  Target: usable on Raspberry Pi 3 at 320×240.

Signal mapping:
    audio_bass    → large centre blocks, tunnel pulse / zoom
    audio_mid     → side obstacles, tunnel width sway
    audio_treble  → sparkle particles, colour phase shift
    beat_pulse    → block spawn trigger, momentary flash accent
    audio_level   → base tunnel scroll speed

Config params:
    tunnel_speed      float  Base forward scroll speed (world-units/s)
    obstacle_density  float  Spawn rate multiplier (0.5–2.0)
    lane_count        int    Number of obstacle lanes (1–5)
    audio_sensitivity float  Multiplier applied to all audio signals
    glow_strength     float  Additive-blend glow intensity (0–1)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base_mode import VisualMode

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_NEAR_Z: float = 0.4       # blocks/rings removed when z < this
_FAR_Z: float = 8.0        # new blocks spawned at this depth
_N_RINGS: int = 12         # how many tunnel rings to draw per frame
_FOCAL_SCALE: float = 0.7  # focal_length = min(w,h) * _FOCAL_SCALE
_TUNNEL_HW: float = 0.6    # tunnel half-width in world units
_TUNNEL_HH: float = 0.45   # tunnel half-height in world units
_MAX_BLOCKS: int = 40      # hard cap on live blocks (Pi-friendly)
_MAX_SPARKLES: int = 60    # hard cap on live sparkle particles
_AUDIO_HIST_SIZE: int = 8  # number of audio history samples to smooth over
_TRAIL_DECAY: float = 0.65 # frame-to-frame darkening for trail effect


# ---------------------------------------------------------------------------
# Colour palettes  (BGR tuples for OpenCV)
# ---------------------------------------------------------------------------

_NEON_PALETTE: List[Tuple[int, int, int]] = [
    (255, 0, 200),    # neon magenta
    (0, 255, 255),    # cyan
    (0, 200, 255),    # electric blue
    (50, 255, 50),    # lime green
    (0, 100, 255),    # orange-amber
    (200, 0, 255),    # violet
]


def _neon_color(t: float) -> Tuple[int, int, int]:
    """Interpolate the neon palette at position t ∈ [0, 1]."""
    n = len(_NEON_PALETTE)
    scaled = t * (n - 1)
    lo = int(scaled) % n
    hi = (lo + 1) % n
    frac = scaled - int(scaled)
    b = int(_NEON_PALETTE[lo][0] * (1 - frac) + _NEON_PALETTE[hi][0] * frac)
    g = int(_NEON_PALETTE[lo][1] * (1 - frac) + _NEON_PALETTE[hi][1] * frac)
    r = int(_NEON_PALETTE[lo][2] * (1 - frac) + _NEON_PALETTE[hi][2] * frac)
    return (b, g, r)


# ---------------------------------------------------------------------------
# AudioTunnelMode
# ---------------------------------------------------------------------------


class AudioTunnelMode(VisualMode):
    """Audio-reactive perspective tunnel with obstacle blocks."""

    name = "AUDIO TUNNEL"
    subtitle = ""

    def __init__(
        self,
        tunnel_speed: float = 1.5,
        obstacle_density: float = 1.0,
        lane_count: int = 3,
        audio_sensitivity: float = 1.0,
        glow_strength: float = 0.5,
    ) -> None:
        # Config
        self.tunnel_speed = float(tunnel_speed)
        self.obstacle_density = float(obstacle_density)
        self.lane_count = max(1, min(5, int(lane_count)))
        self.audio_sensitivity = float(audio_sensitivity)
        self.glow_strength = float(np.clip(glow_strength, 0.0, 1.0))

        # Internal state (reset on activation)
        self._time: float = 0.0
        self._color_phase: float = 0.0
        self._beat_gate: float = 0.0
        self._flash: float = 0.0           # momentary beat flash 0-1

        # Ring scroll: _ring_z_offset increases every frame; rings whose
        # effective depth drops below _NEAR_Z are re-queued at _FAR_Z.
        self._ring_offsets: List[float] = []

        # Obstacle blocks: each is a dict with keys:
        #   wx, wy  — world-space centre x/y
        #   z       — current depth (decreases as block moves toward viewer)
        #   ww, wh  — world-space half-width and half-height
        #   color   — BGR tuple
        #   btype   — 'bass' | 'mid' | 'treble'
        self._blocks: List[Dict[str, Any]] = []

        # Sparkle particles: lightweight dots
        # Each: {wx, wy, z, life, color}
        self._sparkles: List[Dict[str, Any]] = []

        # Audio history buffer for smoothed spawn decisions
        self._bass_hist: List[float] = [0.0] * _AUDIO_HIST_SIZE
        self._mid_hist: List[float] = [0.0] * _AUDIO_HIST_SIZE
        self._treble_hist: List[float] = [0.0] * _AUDIO_HIST_SIZE

        # Persistent dark canvas for trail/glow effect
        self._canvas: Optional[np.ndarray] = None

        self._rng = random.Random(0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._time = 0.0
        self._color_phase = 0.0
        self._beat_gate = 0.0
        self._flash = 0.0
        self._ring_offsets = []
        self._blocks = []
        self._sparkles = []
        self._bass_hist = [0.0] * _AUDIO_HIST_SIZE
        self._mid_hist = [0.0] * _AUDIO_HIST_SIZE
        self._treble_hist = [0.0] * _AUDIO_HIST_SIZE
        self._canvas = None
        self._rng = random.Random(0)

    def update(self, dt: float, signals: Dict[str, Any]) -> None:
        bass   = float(self._sig(signals, "audio_bass",   0.0)) * self.audio_sensitivity
        mid    = float(self._sig(signals, "audio_mid",    0.0)) * self.audio_sensitivity
        treble = float(self._sig(signals, "audio_treble", 0.0)) * self.audio_sensitivity
        beat   = float(self._sig(signals, "beat_pulse",   0.0))
        level  = float(self._sig(signals, "audio_level",  0.0)) * self.audio_sensitivity

        self._time += dt
        # Colour phase drifts with treble
        self._color_phase += dt * (0.3 + treble * 1.5)

        # Decay beat gate and flash
        self._beat_gate = max(0.0, self._beat_gate - dt * 3.0)
        self._flash = max(0.0, self._flash - dt * 6.0)

        # Push audio into history buffers
        self._bass_hist.pop(0)
        self._bass_hist.append(min(1.0, bass))
        self._mid_hist.pop(0)
        self._mid_hist.append(min(1.0, mid))
        self._treble_hist.pop(0)
        self._treble_hist.append(min(1.0, treble))

        # Effective speed: base + audio level contribution
        speed = self.tunnel_speed * (1.0 + level * 0.8)

        # Scroll ring offsets
        self._scroll_rings(dt, speed)

        # Scroll existing blocks toward viewer
        move = dt * speed
        surviving = []
        for b in self._blocks:
            b["z"] -= move
            if b["z"] >= _NEAR_Z:
                surviving.append(b)
        self._blocks = surviving

        # Scroll / age sparkles
        surviving_sp = []
        for sp in self._sparkles:
            sp["z"] -= move * 1.2  # sparkles move slightly faster
            sp["life"] -= dt * 2.0
            if sp["z"] >= _NEAR_Z and sp["life"] > 0.0:
                surviving_sp.append(sp)
        self._sparkles = surviving_sp

        # Spawn new content based on audio
        self._spawn_blocks(dt, bass, mid, treble, beat)
        self._spawn_sparkles(dt, treble, beat)

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def _scroll_rings(self, dt: float, speed: float) -> None:
        """Initialise ring offsets if needed, then advance them."""
        if not self._ring_offsets:
            # Evenly distributed across the tunnel depth
            interval = (_FAR_Z - _NEAR_Z) / _N_RINGS
            self._ring_offsets = [
                _NEAR_Z + i * interval for i in range(_N_RINGS)
            ]

        move = dt * speed
        new_offsets = []
        for z in self._ring_offsets:
            z -= move
            if z < _NEAR_Z:
                z += (_FAR_Z - _NEAR_Z)  # wrap to back
            new_offsets.append(z)
        self._ring_offsets = new_offsets

    def _spawn_blocks(
        self,
        dt: float,
        bass: float,
        mid: float,
        treble: float,
        beat: float,
    ) -> None:
        if len(self._blocks) >= _MAX_BLOCKS:
            return

        # Smoothed averages
        avg_bass   = sum(self._bass_hist)   / len(self._bass_hist)
        avg_mid    = sum(self._mid_hist)    / len(self._mid_hist)
        avg_treble = sum(self._treble_hist) / len(self._treble_hist)

        # Spawn rate: base_rate + audio contribution
        base_rate = 0.5 * self.obstacle_density  # blocks/s baseline
        spawn_rate = base_rate + avg_bass * 2.0 + avg_mid * 1.5 + avg_treble * 1.0

        # Beat trigger: instant spawn
        if beat > 0.7 and self._beat_gate <= 0.0:
            self._flash = 1.0
            self._beat_gate = 0.4
            self._try_spawn_block("bass", bass, mid)
            return

        # Probabilistic spawn
        prob = spawn_rate * dt
        if self._rng.random() < prob:
            # Choose block type by audio band
            r = self._rng.random()
            if r < 0.4:
                btype = "bass"
            elif r < 0.75:
                btype = "mid"
            else:
                btype = "treble"
            self._try_spawn_block(btype, bass, mid)

    def _try_spawn_block(
        self, btype: str, bass: float, mid: float
    ) -> None:
        lane_positions = self._lane_positions()

        if btype == "bass":
            # Large block in a random lane, biased toward centre
            wx = self._rng.choice(lane_positions)
            wy = 0.0
            ww = _TUNNEL_HW * (0.35 + bass * 0.35)
            wh = _TUNNEL_HH * (0.30 + bass * 0.30)
            color = _neon_color((self._color_phase * 0.1) % 1.0)
        elif btype == "mid":
            # Smaller blocks in outer lanes
            outer = [p for p in lane_positions if abs(p) > 0.05]
            if not outer:
                outer = lane_positions
            wx = self._rng.choice(outer)
            wy = self._rng.uniform(-_TUNNEL_HH * 0.5, _TUNNEL_HH * 0.5)
            ww = _TUNNEL_HW * (0.15 + mid * 0.15)
            wh = _TUNNEL_HH * (0.15 + mid * 0.15)
            color = _neon_color(((self._color_phase + 0.3) * 0.1) % 1.0)
        else:  # treble
            # Tiny blocks scattered across lanes
            wx = self._rng.choice(lane_positions)
            wy = self._rng.uniform(-_TUNNEL_HH * 0.6, _TUNNEL_HH * 0.6)
            ww = _TUNNEL_HW * 0.08
            wh = _TUNNEL_HH * 0.08
            color = _neon_color(((self._color_phase + 0.6) * 0.1) % 1.0)

        self._blocks.append({
            "wx": wx,
            "wy": wy,
            "z": _FAR_Z,
            "ww": ww,
            "wh": wh,
            "color": color,
            "btype": btype,
        })

    def _spawn_sparkles(self, dt: float, treble: float, beat: float) -> None:
        if len(self._sparkles) >= _MAX_SPARKLES:
            return

        avg_treble = sum(self._treble_hist) / len(self._treble_hist)
        rate = avg_treble * 8.0 * self.obstacle_density
        if beat > 0.5:
            rate += 4.0

        n = int(rate * dt * 3)
        for _ in range(n):
            if len(self._sparkles) >= _MAX_SPARKLES:
                break
            wx = self._rng.uniform(-_TUNNEL_HW * 0.9, _TUNNEL_HW * 0.9)
            wy = self._rng.uniform(-_TUNNEL_HH * 0.9, _TUNNEL_HH * 0.9)
            color = _neon_color(((self._color_phase + 0.5) * 0.1) % 1.0)
            self._sparkles.append({
                "wx": wx,
                "wy": wy,
                "z": _FAR_Z * self._rng.uniform(0.5, 1.0),
                "life": 1.0,
                "color": color,
            })

    def _lane_positions(self) -> List[float]:
        """Return lane centre x-positions in world units."""
        n = self.lane_count
        if n == 1:
            return [0.0]
        hw = _TUNNEL_HW * 0.7
        return [
            -hw + i * (2 * hw / (n - 1))
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _project(
        self,
        wx: float,
        wy: float,
        wz: float,
        cx: float,
        cy: float,
        focal: float,
    ) -> Tuple[int, int, float]:
        """Project a world-space point to screen space.

        Returns (sx, sy, scale) or (-1, -1, 0) if behind the viewer.
        """
        if wz <= 0.0:
            return (-1, -1, 0.0)
        scale = focal / wz
        sx = int(cx + wx * scale)
        sy = int(cy - wy * scale)   # y-up in world → y-down on screen
        return (sx, sy, scale)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _draw_tunnel_ring(
        self,
        canvas: np.ndarray,
        z: float,
        cx: float,
        cy: float,
        focal: float,
        hw: float,
        hh: float,
        color: Tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        """Draw one rectangular tunnel ring at depth z."""
        corners_world = [
            (-hw, -hh),
            ( hw, -hh),
            ( hw,  hh),
            (-hw,  hh),
        ]
        pts = []
        for wx, wy in corners_world:
            sx, sy, _ = self._project(wx, wy, z, cx, cy, focal)
            pts.append((sx, sy))

        for i in range(4):
            cv2.line(
                canvas,
                pts[i],
                pts[(i + 1) % 4],
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _draw_tunnel_walls(
        self,
        canvas: np.ndarray,
        z1: float,
        z2: float,
        cx: float,
        cy: float,
        focal: float,
        hw: float,
        hh: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw the connecting wall lines between two adjacent rings."""
        corners_world = [
            (-hw, -hh),
            ( hw, -hh),
            ( hw,  hh),
            (-hw,  hh),
        ]
        for wx, wy in corners_world:
            sx1, sy1, _ = self._project(wx, wy, z1, cx, cy, focal)
            sx2, sy2, _ = self._project(wx, wy, z2, cx, cy, focal)
            cv2.line(canvas, (sx1, sy1), (sx2, sy2), color, 1, cv2.LINE_AA)

    def _draw_block(
        self,
        canvas: np.ndarray,
        block: Dict[str, Any],
        cx: float,
        cy: float,
        focal: float,
        flash: float,
    ) -> None:
        """Draw a single obstacle block projected into screen space."""
        z = block["z"]
        if z <= _NEAR_Z:
            return
        wx, wy = block["wx"], block["wy"]
        ww, wh = block["ww"], block["wh"]
        color = block["color"]

        # Brighten on flash
        if flash > 0.0:
            color = tuple(min(255, int(c * (1.0 + flash * 0.5))) for c in color)

        sx1, sy1, _ = self._project(wx - ww, wy + wh, z, cx, cy, focal)
        sx2, sy2, _ = self._project(wx + ww, wy - wh, z, cx, cy, focal)

        x1, x2 = min(sx1, sx2), max(sx1, sx2)
        y1, y2 = min(sy1, sy2), max(sy1, sy2)

        # Clamp to canvas
        h, w = canvas.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        # Bright outline
        outline = tuple(min(255, c + 60) for c in color)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), outline, 1, cv2.LINE_AA)

    def _draw_sparkle(
        self,
        canvas: np.ndarray,
        sp: Dict[str, Any],
        cx: float,
        cy: float,
        focal: float,
    ) -> None:
        z = sp["z"]
        if z <= _NEAR_Z:
            return
        sx, sy, scale = self._project(sp["wx"], sp["wy"], z, cx, cy, focal)
        h, w = canvas.shape[:2]
        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            return
        alpha = sp["life"]
        color = tuple(int(c * alpha) for c in sp["color"])
        r = max(1, int(scale * 0.06))
        cv2.circle(canvas, (sx, sy), r, color, -1, cv2.LINE_AA)

    def _draw_lane_markers(
        self,
        canvas: np.ndarray,
        cx: float,
        cy: float,
        focal: float,
        hw: float,
        near_z: float,
        far_z: float,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw optional lane dividers along the tunnel floor."""
        lanes = self._lane_positions()
        for lx in lanes[1:]:   # skip the first (wall)
            sx_near, sy_near, _ = self._project(lx, -hw, near_z, cx, cy, focal)
            sx_far,  sy_far,  _ = self._project(lx, -hw, far_z,  cx, cy, focal)
            cv2.line(canvas, (sx_near, sy_near), (sx_far, sy_far),
                     color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Main render
    # ------------------------------------------------------------------

    def render(
        self, frame: np.ndarray, signals: Dict[str, Any]
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        cx, cy = w * 0.5, h * 0.5
        focal = min(w, h) * _FOCAL_SCALE

        bass   = float(self._sig(signals, "audio_bass",   0.0)) * self.audio_sensitivity
        mid    = float(self._sig(signals, "audio_mid",    0.0)) * self.audio_sensitivity
        treble = float(self._sig(signals, "audio_treble", 0.0)) * self.audio_sensitivity

        # Pulse tunnel dimensions with bass
        hw = _TUNNEL_HW * (1.0 + bass * 0.25)
        hh = _TUNNEL_HH * (1.0 + bass * 0.20)

        # ---- 1. Trail decay: darken existing canvas ----
        if self._canvas is None or self._canvas.shape[:2] != (h, w):
            self._canvas = np.zeros((h, w, 3), dtype=np.uint8)

        decay = _TRAIL_DECAY + self.glow_strength * 0.15
        decay = float(np.clip(decay, 0.4, 0.90))
        self._canvas = (self._canvas.astype(np.float32) * decay).astype(np.uint8)

        # ---- 2. Draw tunnel rings ----
        sorted_rings = sorted(self._ring_offsets, reverse=True)  # far to near
        for i, z in enumerate(sorted_rings):
            if z < _NEAR_Z:
                continue

            # Brightness based on proximity (near=bright, far=dim)
            brightness = float(np.clip(1.0 - (z / _FAR_Z) * 0.75, 0.25, 1.0))
            # Colour cycles along the tunnel with audio
            t = ((self._color_phase + z * 0.15) * 0.05) % 1.0
            ring_color = _neon_color(t)
            ring_color = tuple(int(c * brightness) for c in ring_color)

            thickness = 1 if z > _FAR_Z * 0.4 else 2
            self._draw_tunnel_ring(
                self._canvas, z, cx, cy, focal, hw, hh,
                ring_color, thickness=thickness,
            )

            # Connect to next ring with wall lines
            if i + 1 < len(sorted_rings):
                z_next = sorted_rings[i + 1]
                if z_next >= _NEAR_Z:
                    self._draw_tunnel_walls(
                        self._canvas, z_next, z, cx, cy, focal, hw, hh, ring_color
                    )

        # ---- 3. Optional lane markers ----
        if self.lane_count > 1:
            marker_t = (self._color_phase * 0.03) % 1.0
            marker_color = _neon_color(marker_t)
            marker_color = tuple(int(c * 0.35) for c in marker_color)
            self._draw_lane_markers(
                self._canvas, cx, cy, focal, hh,
                _NEAR_Z * 1.5, _FAR_Z, marker_color,
            )

        # ---- 4. Draw blocks (far to near for correct overlap) ----
        blocks_sorted = sorted(self._blocks, key=lambda b: b["z"], reverse=True)
        for block in blocks_sorted:
            self._draw_block(self._canvas, block, cx, cy, focal, self._flash)

        # ---- 5. Draw sparkles ----
        for sp in self._sparkles:
            self._draw_sparkle(self._canvas, sp, cx, cy, focal)

        # ---- 6. Beat flash: brief white overlay ----
        if self._flash > 0.0:
            overlay = np.full((h, w, 3), 255, dtype=np.uint8)
            alpha = float(np.clip(self._flash * 0.25, 0.0, 0.25))
            result = cv2.addWeighted(
                self._canvas, 1.0 - alpha, overlay, alpha, 0
            )
        else:
            result = self._canvas.copy()

        # ---- 7. Glow: soft blur blended back ----
        if self.glow_strength > 0.0:
            blurred = cv2.GaussianBlur(result, (0, 0), 3)
            glow_alpha = float(np.clip(self.glow_strength * 0.4, 0.0, 0.5))
            result = cv2.addWeighted(result, 1.0, blurred, glow_alpha, 0)
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result
