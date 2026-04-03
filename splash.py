"""
splash.py — Animated startup splash screen for DogDayDiffuser.

Screen layout (at internal resolution, e.g. 320 × 240):

  ┌────────────────────────────────────────┐  ┐
  │  Palette-animated Julia set fractal    │  │ top 2/3
  │  (King Charles Cavalier colour scheme) │  │
  ├────────────────────────────────────────┤  ┘
  │  Hardware detection console            │  ┐ bottom 1/3
  │  OS: …   Camera: …   Audio: …         │  │
  └────────────────────────────────────────┘  ┘

Displayed for SPLASH_DURATION seconds (default 10 s) before the main
animation loop starts.  Press SPACE to skip or Q / Esc to quit.
"""

from __future__ import annotations

import logging
import platform
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

SPLASH_DURATION: float = 10.0  # seconds before the main loop begins

# ---------------------------------------------------------------------------
# King Charles Cavalier colour palette
# ---------------------------------------------------------------------------
# Warm stops: dark mahogany → ruby-red → chestnut → caramel tan → cream white
# BGR tuples (OpenCV convention).
_CAVALIER_STOPS: List[Tuple[int, Tuple[int, int, int]]] = [
    (0,   (8,   5,  18)),   # near-black / dark maroon
    (28,  (20,  30, 110)),  # deep ruby-red
    (56,  (25,  65, 175)),  # ruby-red
    (84,  (40,  95, 200)),  # medium chestnut
    (112, (60, 130, 215)),  # warm chestnut
    (140, (110, 175, 230)), # caramel tan
    (168, (200, 220, 245)), # cream / white highlight
    (196, (120, 155, 210)), # warm tan
    (224, (30,  75, 155)),  # chestnut shadow
    (255, (8,   5,  18)),   # back to near-black
]


def _build_lut(stops: List[Tuple[int, Tuple[int, int, int]]]) -> np.ndarray:
    """Interpolate a 256-entry BGR LUT from (index, BGR) stops."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for k in range(len(stops) - 1):
        i0, c0 = stops[k]
        i1, c1 = stops[k + 1]
        for i in range(i0, min(i1 + 1, 256)):
            t = (i - i0) / max(i1 - i0, 1)
            lut[i] = [int(c0[j] + t * (c1[j] - c0[j])) for j in range(3)]
    return lut


_CAVALIER_LUT: np.ndarray = _build_lut(_CAVALIER_STOPS)

# ---------------------------------------------------------------------------
# Console colour constants (BGR)
# ---------------------------------------------------------------------------
_C_BG   = (10,  6,   2)    # dark warm background
_C_BORD = (40, 100, 180)   # chestnut border / accent
_C_HEAD = (200, 220, 255)  # header title (warm white)
_C_KEY  = (180, 200, 80)   # key label (cyan-ish)
_C_OK   = (80,  200, 80)   # green
_C_WARN = (60,  160, 220)  # amber / orange
_C_ERR  = (60,   60, 220)  # red
_C_VAL  = (210, 210, 210)  # neutral value text
_C_DIM  = (100, 100, 100)  # dim hint text

# ---------------------------------------------------------------------------
# Julia-set fractal (computed once, palette-cycled each frame)
# ---------------------------------------------------------------------------

# c = -0.7269 + 0.1889i  produces a richly-detailed spiral that evokes
# flowing fur — a good visual metaphor for a Cavalier's coat.
_JULIA_C: complex = complex(-0.7269, 0.1889)


def _compute_julia(
    w: int,
    h: int,
    c: complex = _JULIA_C,
    max_iter: int = 70,
) -> np.ndarray:
    """Return a ``(h, w)`` uint8 iteration-count array for the Julia set.

    Uses vectorised NumPy real/imaginary arrays (no Python pixel loop) so the
    computation typically completes in under a second even at 320×160.
    The returned array is the *static* base; animation is achieved by rotating
    the colour LUT on each frame — no recomputation is needed.

    Args:
        w:        Frame width in pixels.
        h:        Frame height in pixels.
        c:        Julia-set constant (complex).
        max_iter: Iteration depth.  Higher values give more detail but take
                  longer to compute.

    Returns:
        uint8 array of shape ``(h, w)`` with values 0–255.
    """
    x = np.linspace(-1.55, 1.55, w, dtype=np.float32)
    y = np.linspace(-1.05, 1.05, h, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    Zr = X.copy()
    Zi = Y.copy()
    M = np.zeros((h, w), dtype=np.uint8)
    alive = np.ones((h, w), dtype=bool)
    cr = float(c.real)
    ci = float(c.imag)

    for i in range(1, max_iter):
        Zr2 = Zr * Zr
        Zi2 = Zi * Zi
        # Escape check before update
        escaped = alive & (Zr2 + Zi2 > 4.0)
        M[escaped] = int(i * 255 / max_iter)
        alive &= ~escaped
        # Z = Z² + c  (update all pixels; masking is slower for large arrays)
        Zi_new = 2.0 * Zr * Zi + ci
        Zr[:] = Zr2 - Zi2 + cr
        Zi[:] = Zi_new
        # Reset escaped pixels so they stay bounded and don't overflow next iteration
        Zr[~alive] = 0.0
        Zi[~alive] = 0.0

    return M


# ---------------------------------------------------------------------------
# Hardware info type
# ---------------------------------------------------------------------------

#: Each entry is (label, value, status) where status ∈ {"ok","warn","err","val"}
HwEntry = Tuple[str, str, str]


def _probe_system() -> List[HwEntry]:
    """Return lightweight system-info rows (no hardware devices opened)."""
    rows: List[HwEntry] = []
    try:
        os_str = f"{platform.system()} {platform.release()}"
        rows.append(("OS", os_str[:42], "val"))
    except Exception:
        rows.append(("OS", "unknown", "warn"))
    try:
        cpu = (platform.processor() or platform.machine() or "unknown")[:42]
        rows.append(("CPU", cpu, "val"))
    except Exception:
        rows.append(("CPU", "unknown", "warn"))
    return rows


# ---------------------------------------------------------------------------
# SplashScreen
# ---------------------------------------------------------------------------


class SplashScreen:
    """Animated splash frame generator.

    Usage::

        hw = [("Camera", "device 0  640×480", "ok"), ...]
        splash = SplashScreen(width=320, height=240, hw_entries=hw)

        start = time.monotonic()
        while True:
            t = time.monotonic() - start
            if t >= SplashScreen.DURATION:
                break
            frame = splash.render(t)
            renderer.show(frame, fps=0.0, effect_name="")
            key = renderer.poll_key()
            if key in (ord("q"), 27):
                quit()
            if key == ord(" "):
                break  # skip to main loop

    The Julia set is pre-computed once at construction time; palette rotation
    is achieved with ``np.roll`` on each ``render()`` call, so the per-frame
    cost is very low.

    Args:
        width:      Frame width in pixels (matches ``AppConfig.width``).
        height:     Frame height in pixels (matches ``AppConfig.height``).
        hw_entries: Hardware detection rows supplied by the caller.  If
                    ``None`` a lightweight system-only probe is run.
        max_iter:   Julia set iteration depth.
    """

    DURATION: float = SPLASH_DURATION

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        hw_entries: Optional[List[HwEntry]] = None,
        max_iter: int = 70,
    ) -> None:
        self.width = width
        self.height = height

        # Vertical split: top 2/3 fractal, bottom 1/3 console
        self._frac_h = max(1, height * 2 // 3)
        self._con_h = max(1, height - self._frac_h)

        # Pre-compute Julia set (once at startup)
        logger.info(
            "Splash: computing Julia fractal %d×%d …", width, self._frac_h
        )
        self._iters = _compute_julia(width, self._frac_h, max_iter=max_iter)
        logger.info("Splash: fractal ready")

        # Hardware info rows
        self._hw: List[HwEntry] = hw_entries if hw_entries is not None else _probe_system()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, t: float) -> np.ndarray:
        """Return a ``(height × width × 3)`` BGR uint8 splash frame.

        Args:
            t: Elapsed time in seconds since the splash started.
        """
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[: self._frac_h] = self._render_fractal(t)
        frame[self._frac_h :] = self._render_console(t)
        return frame

    # ------------------------------------------------------------------
    # Fractal panel (top 2/3)
    # ------------------------------------------------------------------

    def _render_fractal(self, t: float) -> np.ndarray:
        """Palette-animate the pre-computed Julia iteration array."""
        # Rotate the LUT by ~25 steps per second for smooth animation
        shift = int(t * 25.0) % 256
        lut = np.roll(_CAVALIER_LUT, shift, axis=0)
        panel = lut[self._iters]  # (frac_h, width, 3) BGR

        # Title overlay
        cv2.putText(
            panel, "DogDayDiffuser",
            (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return panel

    # ------------------------------------------------------------------
    # Console panel (bottom 1/3)
    # ------------------------------------------------------------------

    def _render_console(self, t: float) -> np.ndarray:
        """Render a colourful hardware-detection status panel."""
        h, w = self._con_h, self.width
        canvas = np.full((h, w, 3), _C_BG, dtype=np.uint8)

        # Outer border
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), _C_BORD, 1)

        # Header bar
        bar_btm = 14
        cv2.rectangle(canvas, (1, 1), (w - 2, bar_btm), _C_BORD, cv2.FILLED)
        cv2.putText(
            canvas, " Hardware Detection",
            (2, 11), cv2.FONT_HERSHEY_SIMPLEX,
            0.33, _C_HEAD, 1, cv2.LINE_AA,
        )

        # Hardware rows — revealed progressively (one every 0.55 s)
        color_map = {
            "ok":   _C_OK,
            "warn": _C_WARN,
            "err":  _C_ERR,
            "val":  _C_VAL,
        }
        line_h = 11
        y0 = bar_btm + 5
        # Reserve space for the bottom hint + progress bar
        max_y = h - 14
        visible = min(int(t / 0.55) + 1, len(self._hw))

        for i in range(visible):
            label, value, status = self._hw[i]
            y = y0 + i * line_h
            if y + line_h > max_y:
                break
            col = color_map.get(status, _C_VAL)
            label_str = f"{label}:"
            cv2.putText(
                canvas, label_str,
                (3, y + 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.28, _C_KEY, 1, cv2.LINE_AA,
            )
            key_w = min(88, len(label_str) * 5 + 6)
            cv2.putText(
                canvas, value,
                (key_w, y + 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.28, col, 1, cv2.LINE_AA,
            )

        # Skip hint
        cv2.putText(
            canvas, "SPACE / Q to skip",
            (3, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.25, _C_DIM, 1, cv2.LINE_AA,
        )

        # Countdown progress bar
        bar_y = h - 4
        remaining = max(0.0, SPLASH_DURATION - t)
        bar_w = max(0, int((remaining / SPLASH_DURATION) * (w - 4)))
        cv2.rectangle(canvas, (2, bar_y), (w - 3, h - 2), (30, 20, 10), cv2.FILLED)
        if bar_w > 0:
            cv2.rectangle(
                canvas, (2, bar_y), (2 + bar_w, h - 2), _C_BORD, cv2.FILLED
            )

        return canvas
