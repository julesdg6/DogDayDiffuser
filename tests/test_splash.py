"""
tests/test_splash.py — Unit tests for the splash screen module.

All tests run without a display, camera, or audio device.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_splash(width: int = 320, height: int = 240, hw=None, max_iter: int = 10):
    """Construct a SplashScreen quickly with a low iteration count."""
    from splash import SplashScreen
    return SplashScreen(width=width, height=height, hw_entries=hw, max_iter=max_iter)


# ---------------------------------------------------------------------------
# _build_lut
# ---------------------------------------------------------------------------


class TestBuildLut:
    def test_lut_shape(self) -> None:
        from splash import _build_lut, _CAVALIER_STOPS
        lut = _build_lut(_CAVALIER_STOPS)
        assert lut.shape == (256, 3)
        assert lut.dtype == np.uint8

    def test_cavalier_lut_is_prebuilt(self) -> None:
        from splash import _CAVALIER_LUT
        assert _CAVALIER_LUT.shape == (256, 3)
        assert _CAVALIER_LUT.dtype == np.uint8

    def test_lut_values_in_range(self) -> None:
        from splash import _CAVALIER_LUT
        assert int(_CAVALIER_LUT.min()) >= 0
        assert int(_CAVALIER_LUT.max()) <= 255


# ---------------------------------------------------------------------------
# _compute_julia
# ---------------------------------------------------------------------------


class TestComputeJulia:
    def test_output_shape(self) -> None:
        from splash import _compute_julia
        M = _compute_julia(32, 20, max_iter=5)
        assert M.shape == (20, 32)

    def test_output_dtype(self) -> None:
        from splash import _compute_julia
        M = _compute_julia(32, 20, max_iter=5)
        assert M.dtype == np.uint8

    def test_values_in_range(self) -> None:
        from splash import _compute_julia
        M = _compute_julia(32, 20, max_iter=5)
        assert int(M.min()) >= 0
        assert int(M.max()) <= 255

    def test_not_all_zero(self) -> None:
        from splash import _compute_julia
        M = _compute_julia(64, 40, max_iter=10)
        assert M.max() > 0, "Julia set should produce non-zero iteration counts"

    def test_custom_constant(self) -> None:
        from splash import _compute_julia
        M = _compute_julia(32, 20, c=complex(-0.4, 0.6), max_iter=5)
        assert M.shape == (20, 32)


# ---------------------------------------------------------------------------
# _probe_system
# ---------------------------------------------------------------------------


class TestProbeSystem:
    def test_returns_list(self) -> None:
        from splash import _probe_system
        rows = _probe_system()
        assert isinstance(rows, list)
        assert len(rows) >= 1

    def test_each_entry_is_triple(self) -> None:
        from splash import _probe_system
        for label, value, status in _probe_system():
            assert isinstance(label, str)
            assert isinstance(value, str)
            assert status in ("ok", "warn", "err", "val")


# ---------------------------------------------------------------------------
# SplashScreen construction
# ---------------------------------------------------------------------------


class TestSplashScreenInit:
    def test_construction_no_hw(self) -> None:
        splash = _make_splash()
        assert splash.width == 320
        assert splash.height == 240

    def test_construction_with_hw(self) -> None:
        hw = [("Camera", "device 0  640×480", "ok"), ("Audio", "disabled", "warn")]
        splash = _make_splash(hw=hw)
        assert splash._hw == hw

    def test_frac_h_is_two_thirds(self) -> None:
        splash = _make_splash(height=240)
        assert splash._frac_h == 160  # 240 * 2 // 3

    def test_con_h_is_remainder(self) -> None:
        splash = _make_splash(height=240)
        assert splash._con_h == 80  # 240 - 160

    def test_frac_h_plus_con_h_equals_height(self) -> None:
        for h in (180, 240, 480):
            splash = _make_splash(height=h)
            assert splash._frac_h + splash._con_h == h

    def test_non_standard_resolution(self) -> None:
        splash = _make_splash(width=160, height=120)
        assert splash.width == 160
        assert splash.height == 120

    def test_iters_shape(self) -> None:
        splash = _make_splash(width=64, height=48)
        assert splash._iters.shape == (splash._frac_h, 64)

    def test_duration_constant(self) -> None:
        from splash import SPLASH_DURATION, SplashScreen
        assert SplashScreen.DURATION == SPLASH_DURATION
        assert SplashScreen.DURATION == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# SplashScreen.render
# ---------------------------------------------------------------------------


class TestSplashScreenRender:
    def test_render_returns_correct_shape(self) -> None:
        splash = _make_splash(width=320, height=240)
        frame = splash.render(0.0)
        assert frame.shape == (240, 320, 3)

    def test_render_dtype_is_uint8(self) -> None:
        splash = _make_splash()
        frame = splash.render(0.0)
        assert frame.dtype == np.uint8

    def test_render_multiple_times_stable(self) -> None:
        splash = _make_splash()
        for t in [0.0, 1.0, 5.0, 9.9]:
            frame = splash.render(t)
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8

    def test_render_non_standard_resolution(self) -> None:
        splash = _make_splash(width=160, height=120)
        frame = splash.render(0.5)
        assert frame.shape == (120, 160, 3)

    def test_render_at_t_zero(self) -> None:
        splash = _make_splash()
        frame = splash.render(0.0)
        # Frame should not be entirely black (fractal has colours)
        assert frame.max() > 0

    def test_palette_animation_changes_frame(self) -> None:
        """Frames at different times should differ (palette rotation)."""
        splash = _make_splash()
        f0 = splash.render(0.0).copy()
        f5 = splash.render(5.0).copy()
        # At least some pixels should differ due to palette rotation
        assert not np.array_equal(f0, f5), \
            "Frames at t=0 and t=5 should differ due to palette animation"

    def test_render_with_many_hw_entries(self) -> None:
        hw = [(f"Dev{i}", f"value{i}", "ok") for i in range(20)]
        splash = _make_splash(hw=hw)
        frame = splash.render(15.0)  # all entries revealed
        assert frame.shape == (240, 320, 3)

    def test_render_with_empty_hw(self) -> None:
        splash = _make_splash(hw=[])
        frame = splash.render(1.0)
        assert frame.shape == (240, 320, 3)


# ---------------------------------------------------------------------------
# Fractal panel
# ---------------------------------------------------------------------------


class TestRenderFractal:
    def test_fractal_panel_shape(self) -> None:
        splash = _make_splash(width=320, height=240)
        panel = splash._render_fractal(0.0)
        assert panel.shape == (splash._frac_h, 320, 3)

    def test_fractal_panel_dtype(self) -> None:
        splash = _make_splash()
        panel = splash._render_fractal(0.0)
        assert panel.dtype == np.uint8

    def test_fractal_varies_with_time(self) -> None:
        splash = _make_splash()
        p0 = splash._render_fractal(0.0).copy()
        p5 = splash._render_fractal(5.0).copy()
        assert not np.array_equal(p0, p5)


# ---------------------------------------------------------------------------
# Console panel
# ---------------------------------------------------------------------------


class TestRenderConsole:
    def test_console_panel_shape(self) -> None:
        splash = _make_splash(width=320, height=240)
        panel = splash._render_console(0.0)
        assert panel.shape == (splash._con_h, 320, 3)

    def test_console_panel_dtype(self) -> None:
        splash = _make_splash()
        panel = splash._render_console(0.0)
        assert panel.dtype == np.uint8

    def test_console_with_all_status_types(self) -> None:
        hw = [
            ("OK item",   "all good",   "ok"),
            ("Warn item", "might fail", "warn"),
            ("Err item",  "broken",     "err"),
            ("Val item",  "info only",  "val"),
        ]
        splash = _make_splash(hw=hw)
        panel = splash._render_console(10.0)  # all revealed
        assert panel.shape == (splash._con_h, 320, 3)

    def test_console_progressive_reveal(self) -> None:
        """Console panel at t=0 should differ from t=5 (more rows revealed)."""
        hw = [(f"Item{i}", f"val{i}", "ok") for i in range(8)]
        splash = _make_splash(hw=hw)
        p0 = splash._render_console(0.0).copy()
        p5 = splash._render_console(5.0).copy()
        assert not np.array_equal(p0, p5)

    def test_countdown_bar_changes(self) -> None:
        """Progress bar should shrink as time advances."""
        splash = _make_splash(hw=[])
        p0 = splash._render_console(0.0).copy()
        p9 = splash._render_console(9.0).copy()
        assert not np.array_equal(p0, p9)
