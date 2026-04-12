"""
tests/test_modes.py — Unit tests for the Geiss and MilkDrop visual modes.

These tests do not require any hardware (no webcam, no audio device).  They
use dummy NumPy frames and mock signal dicts so they can run in CI.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a solid mid-grey BGR frame."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


_RNG_SEED = 42


def _noise_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Return a random-noise BGR frame (exercises warp paths more thoroughly)."""
    rng = np.random.default_rng(_RNG_SEED)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _empty_signals(frame_w: int = 320, frame_h: int = 240) -> Dict[str, Any]:
    return {
        "audio_level":  0.0,
        "audio_bass":   0.0,
        "audio_mid":    0.0,
        "audio_treble": 0.0,
        "beat_pulse":   0.0,
        "motion":       0.0,
        "face_count":   0,
        "face_center":  (frame_w / 2.0, frame_h / 2.0),
        "face_size":    0.0,
        "fps":          25.0,
        "source_name":  "test",
    }


def _audio_signals(**overrides) -> Dict[str, Any]:
    sig = _empty_signals()
    sig.update(overrides)
    return sig


# ---------------------------------------------------------------------------
# modes/__init__.py — registry and build_signals
# ---------------------------------------------------------------------------


class TestModeRegistry:
    def test_modes_dict_contains_geiss_and_milkdrop(self) -> None:
        from modes import MODES
        assert "geiss" in MODES
        assert "milkdrop" in MODES

    def test_modes_dict_contains_audio_tunnel(self) -> None:
        from modes import MODES
        assert "audio_tunnel" in MODES

    def test_mode_names_list(self) -> None:
        from modes import MODE_NAMES
        assert "geiss" in MODE_NAMES
        assert "milkdrop" in MODE_NAMES
        assert "audio_tunnel" in MODE_NAMES

    def test_preset_names_list(self) -> None:
        from modes import PRESET_NAMES
        assert len(PRESET_NAMES) > 0
        assert "mandala_pulse" in PRESET_NAMES

    def test_build_signals_defaults(self) -> None:
        from modes import build_signals
        sig = build_signals()
        assert sig["audio_level"] == 0.0
        assert sig["face_count"] == 0
        assert sig["motion"] == 0.0

    def test_build_signals_with_audio(self) -> None:
        from modes import build_signals
        from unittest.mock import MagicMock
        audio = MagicMock()
        audio.volume = 0.5
        audio.bass = 0.7
        audio.mid = 0.3
        audio.treble = 0.2
        audio.beat = 0.9
        sig = build_signals(audio=audio)
        assert sig["audio_level"] == pytest.approx(0.5)
        assert sig["audio_bass"] == pytest.approx(0.7)
        assert sig["beat_pulse"] == pytest.approx(0.9)

    def test_build_signals_with_face(self) -> None:
        from modes import build_signals
        from unittest.mock import MagicMock
        face = MagicMock()
        face.detected = True
        face.cx_norm = 0.6
        face.cy_norm = 0.4
        face.w = 60
        face.h = 70
        sig = build_signals(face=face, frame_w=320, frame_h=240)
        assert sig["face_count"] == 1
        assert sig["face_center"] == pytest.approx((0.6 * 320, 0.4 * 240))
        assert sig["face_size"] == 70

    def test_build_signals_undetected_face_ignored(self) -> None:
        from modes import build_signals
        from unittest.mock import MagicMock
        face = MagicMock()
        face.detected = False
        sig = build_signals(face=face)
        assert sig["face_count"] == 0

    def test_build_signals_motion(self) -> None:
        from modes import build_signals
        sig = build_signals(motion=0.42)
        assert sig["motion"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# VisualMode base class
# ---------------------------------------------------------------------------


class TestVisualModeBase:
    def test_base_mode_is_abstract(self) -> None:
        from modes.base_mode import VisualMode
        with pytest.raises(TypeError):
            VisualMode()  # type: ignore[abstract]

    def test_sig_helper_default(self) -> None:
        from modes.base_mode import VisualMode

        class _Stub(VisualMode):
            name = "stub"
            def render(self, frame, signals):
                return frame

        stub = _Stub()
        assert stub._sig({}, "missing_key", 99) == 99
        assert stub._sig({"foo": 42}, "foo", 0) == 42


# ---------------------------------------------------------------------------
# GeissMode
# ---------------------------------------------------------------------------


class TestGeissMode:
    def test_render_returns_same_shape(self) -> None:
        from modes.geiss_mode import GeissMode
        mode = GeissMode()
        frame = _blank_frame()
        out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_render_multiple_frames_stable(self) -> None:
        """Render several frames and ensure output shape stays constant."""
        from modes.geiss_mode import GeissMode
        mode = GeissMode()
        frame = _noise_frame()
        for _ in range(5):
            out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_reset_clears_buffer(self) -> None:
        from modes.geiss_mode import GeissMode
        mode = GeissMode()
        frame = _blank_frame()
        mode.render(frame, _empty_signals())
        assert mode._buffer is not None
        mode.reset()
        assert mode._buffer is None

    def test_update_advances_phase(self) -> None:
        from modes.geiss_mode import GeissMode
        mode = GeissMode()
        phase_before = mode._phase
        mode.update(1.0 / 30.0, _empty_signals())
        assert mode._phase > phase_before

    def test_audio_signals_do_not_crash(self) -> None:
        from modes.geiss_mode import GeissMode
        mode = GeissMode()
        frame = _noise_frame()
        sig = _audio_signals(audio_bass=1.0, audio_mid=1.0,
                             audio_treble=1.0, beat_pulse=1.0)
        out = mode.render(frame, sig)
        assert out.shape == frame.shape

    def test_face_signals_do_not_crash(self) -> None:
        from modes.geiss_mode import GeissMode
        mode = GeissMode()
        frame = _noise_frame()
        sig = _empty_signals()
        sig["face_count"] = 1
        sig["face_center"] = (80.0, 60.0)
        sig["face_size"] = 50.0
        out = mode.render(frame, sig)
        assert out.shape == frame.shape

    def test_symmetry_disabled(self) -> None:
        from modes.geiss_mode import GeissMode
        mode = GeissMode(use_symmetry=False)
        frame = _noise_frame()
        out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape

    def test_plasma_disabled(self) -> None:
        from modes.geiss_mode import GeissMode
        mode = GeissMode(plasma_overlay=False)
        frame = _noise_frame()
        out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape

    def test_high_symmetry_count(self) -> None:
        from modes.geiss_mode import GeissMode
        mode = GeissMode(symmetry_count=8)
        frame = _noise_frame()
        out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape

    def test_non_standard_resolution(self) -> None:
        from modes.geiss_mode import GeissMode
        mode = GeissMode()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        out = mode.render(frame, _empty_signals(frame_w=160, frame_h=120))
        assert out.shape == frame.shape


# ---------------------------------------------------------------------------
# MilkDropMode
# ---------------------------------------------------------------------------


class TestMilkDropMode:
    def test_render_returns_same_shape(self) -> None:
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode()
        frame = _blank_frame()
        out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_render_multiple_frames_stable(self) -> None:
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode()
        frame = _noise_frame()
        for _ in range(5):
            out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape

    def test_reset_clears_buffer(self) -> None:
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode()
        frame = _blank_frame()
        mode.render(frame, _empty_signals())
        assert mode._buffer is not None
        mode.reset()
        assert mode._buffer is None

    def test_preset_cycling_forward(self) -> None:
        from modes.milkdrop_mode import MilkDropMode, PRESETS
        mode = MilkDropMode(auto_cycle=False)
        start_idx = mode._preset_idx
        mode.cycle_preset(direction=1)
        assert mode._next_idx == (start_idx + 1) % len(PRESETS)
        assert mode._interp_active is True

    def test_preset_cycling_backward(self) -> None:
        from modes.milkdrop_mode import MilkDropMode, PRESETS
        mode = MilkDropMode(auto_cycle=False)
        mode._preset_idx = 2
        mode.cycle_preset(direction=-1)
        assert mode._next_idx == 1
        assert mode._interp_active is True

    def test_preset_cycling_wraps_backward(self) -> None:
        from modes.milkdrop_mode import MilkDropMode, PRESETS
        mode = MilkDropMode(auto_cycle=False)
        mode._preset_idx = 0
        mode.cycle_preset(direction=-1)
        assert mode._next_idx == len(PRESETS) - 1

    def test_preset_cycling_wraps_forward(self) -> None:
        from modes.milkdrop_mode import MilkDropMode, PRESETS
        mode = MilkDropMode(auto_cycle=False)
        mode._preset_idx = len(PRESETS) - 1
        mode.cycle_preset(direction=1)
        assert mode._next_idx == 0

    def test_subtitle_contains_preset_name(self) -> None:
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode()
        assert mode.preset["name"].upper() in mode.subtitle

    def test_auto_cycle_advances_preset(self) -> None:
        """After enough time elapses, the preset should advance."""
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode(auto_cycle=True, cycle_seconds=1.0,
                            beat_transition=False)
        start_idx = mode._preset_idx
        # A single large dt step exceeds cycle_seconds (1 s) AND gives the
        # interpolator (speed 0.8/s) enough time to finish (1/0.8 ≈ 1.25 s).
        # dt=3 s is well beyond both, so the transition completes in one call.
        mode.update(3.0, _empty_signals())
        assert mode._preset_idx != start_idx, "Preset should have advanced after transition"
        assert mode._interp_active is False, "Interpolation should be complete"

    def test_beat_transition_fires(self) -> None:
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode(auto_cycle=False, beat_transition=True,
                            cycle_seconds=999.0)
        # Advance time past the 5 s minimum dwell
        mode._time_since_last = 6.0
        start_idx = mode._preset_idx
        sig = _audio_signals(beat_pulse=1.0)
        mode.update(1.0 / 30.0, sig)
        assert mode._interp_active is True

    def test_no_beat_transition_when_disabled(self) -> None:
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode(auto_cycle=False, beat_transition=False)
        mode._time_since_last = 6.0
        sig = _audio_signals(beat_pulse=1.0)
        mode.update(1.0 / 30.0, sig)
        assert mode._interp_active is False

    def test_all_presets_renderable(self) -> None:
        from modes.milkdrop_mode import MilkDropMode, PRESETS
        mode = MilkDropMode(auto_cycle=False)
        frame = _noise_frame()
        for i in range(len(PRESETS)):
            mode._preset_idx = i
            out = mode.render(frame, _empty_signals())
            assert out.shape == frame.shape, f"Preset {i} ({PRESETS[i]['name']}) failed"

    def test_audio_signals_do_not_crash(self) -> None:
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode()
        frame = _noise_frame()
        sig = _audio_signals(audio_bass=1.0, audio_mid=1.0,
                             audio_treble=1.0, beat_pulse=1.0,
                             audio_level=1.0)
        out = mode.render(frame, sig)
        assert out.shape == frame.shape

    def test_face_signals_do_not_crash(self) -> None:
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode()
        frame = _noise_frame()
        sig = _empty_signals()
        sig["face_count"] = 1
        sig["face_center"] = (100.0, 80.0)
        sig["face_size"] = 60.0
        out = mode.render(frame, sig)
        assert out.shape == frame.shape

    def test_non_standard_resolution(self) -> None:
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        out = mode.render(frame, _empty_signals(frame_w=160, frame_h=120))
        assert out.shape == frame.shape

    def test_parameter_interpolation(self) -> None:
        from modes.milkdrop_mode import MilkDropMode
        mode = MilkDropMode(auto_cycle=False)
        mode._preset_idx = 0
        mode._next_idx = 1
        mode._interp_active = True
        mode._interp_t = 0.5
        # At t=0.5, param should be between current and next
        val = mode._param("feedback_decay", 0.88)
        cur = mode.preset["feedback_decay"]
        nxt = mode.next_preset["feedback_decay"]
        assert min(cur, nxt) <= val <= max(cur, nxt)


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------


class TestPaletteHelpers:
    def test_palette_color_returns_bgr_tuple(self) -> None:
        from modes.milkdrop_mode import _palette_color
        color = _palette_color("neon_fire", 0.5)
        assert len(color) == 3
        for c in color:
            assert 0 <= c <= 255

    def test_palette_color_endpoints(self) -> None:
        from modes.milkdrop_mode import _palette_color
        c0 = _palette_color("cyber_blue", 0.0)
        c1 = _palette_color("cyber_blue", 1.0)
        assert len(c0) == 3
        assert len(c1) == 3

    def test_unknown_palette_falls_back(self) -> None:
        from modes.milkdrop_mode import _palette_color
        color = _palette_color("nonexistent_palette", 0.5)
        assert len(color) == 3

    def test_all_palettes_accessible(self) -> None:
        from modes.milkdrop_mode import PALETTES, _palette_color
        for name in PALETTES:
            color = _palette_color(name, 0.5)
            assert len(color) == 3


# ---------------------------------------------------------------------------
# AudioTunnelMode
# ---------------------------------------------------------------------------


class TestAudioTunnelMode:
    def test_render_returns_same_shape(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode()
        frame = _blank_frame()
        out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_render_multiple_frames_stable(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode()
        frame = _noise_frame()
        for _ in range(5):
            out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_reset_clears_state(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode()
        frame = _blank_frame()
        mode.render(frame, _empty_signals())
        assert mode._canvas is not None
        mode.reset()
        assert mode._canvas is None
        assert mode._blocks == []
        assert mode._sparkles == []
        assert mode._time == 0.0

    def test_update_advances_time(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode()
        mode.update(1.0 / 30.0, _empty_signals())
        assert mode._time > 0.0

    def test_audio_signals_do_not_crash(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode()
        frame = _noise_frame()
        sig = _audio_signals(audio_bass=1.0, audio_mid=1.0,
                             audio_treble=1.0, beat_pulse=1.0,
                             audio_level=1.0)
        out = mode.render(frame, sig)
        assert out.shape == frame.shape

    def test_beat_pulse_triggers_flash(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode()
        sig = _audio_signals(beat_pulse=1.0)
        mode.update(1.0 / 30.0, sig)
        assert mode._flash > 0.0

    def test_blocks_spawned_with_audio(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode(obstacle_density=2.0)
        sig = _audio_signals(audio_bass=1.0, audio_mid=1.0,
                             audio_treble=1.0, beat_pulse=1.0)
        # Run many updates to ensure blocks are spawned
        for _ in range(60):
            mode.update(1.0 / 30.0, sig)
        assert len(mode._blocks) > 0

    def test_block_count_capped(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode, _MAX_BLOCKS
        mode = AudioTunnelMode(obstacle_density=5.0)
        sig = _audio_signals(audio_bass=1.0, audio_mid=1.0,
                             audio_treble=1.0, beat_pulse=1.0)
        for _ in range(300):
            mode.update(1.0 / 30.0, sig)
        assert len(mode._blocks) <= _MAX_BLOCKS

    def test_non_standard_resolution(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        out = mode.render(frame, _empty_signals(frame_w=160, frame_h=120))
        assert out.shape == frame.shape

    def test_single_lane(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode(lane_count=1)
        frame = _noise_frame()
        out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape

    def test_max_lane_count(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode(lane_count=5)
        frame = _noise_frame()
        out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape

    def test_glow_disabled(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode(glow_strength=0.0)
        frame = _noise_frame()
        out = mode.render(frame, _empty_signals())
        assert out.shape == frame.shape

    def test_high_sensitivity(self) -> None:
        from modes.audio_tunnel_mode import AudioTunnelMode
        mode = AudioTunnelMode(audio_sensitivity=3.0)
        frame = _noise_frame()
        sig = _audio_signals(audio_bass=0.5, audio_mid=0.5, audio_treble=0.5)
        out = mode.render(frame, sig)
        assert out.shape == frame.shape

    def test_neon_color_helper(self) -> None:
        from modes.audio_tunnel_mode import _neon_color
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            color = _neon_color(t)
            assert len(color) == 3
            for c in color:
                assert 0 <= c <= 255

    def test_in_registry(self) -> None:
        from modes import MODES
        assert "audio_tunnel" in MODES
        from modes.audio_tunnel_mode import AudioTunnelMode
        assert MODES["audio_tunnel"] is AudioTunnelMode


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestModeConfig:
    def test_parse_args_default_mode_none(self) -> None:
        from config import parse_args
        cfg = parse_args([])
        assert cfg.default_mode is None

    def test_parse_args_mode_geiss(self) -> None:
        from config import parse_args
        cfg = parse_args(["--mode", "geiss"])
        assert cfg.default_mode == "geiss"

    def test_parse_args_mode_milkdrop(self) -> None:
        from config import parse_args
        cfg = parse_args(["--mode", "milkdrop"])
        assert cfg.default_mode == "milkdrop"

    def test_parse_args_milkdrop_cycle_seconds(self) -> None:
        from config import parse_args
        cfg = parse_args(["--milkdrop-cycle-seconds", "30"])
        assert cfg.milkdrop_cycle_seconds == pytest.approx(30.0)

    def test_parse_args_no_milkdrop_auto_cycle(self) -> None:
        from config import parse_args
        cfg = parse_args(["--no-milkdrop-auto-cycle"])
        assert cfg.milkdrop_auto_cycle is False

    def test_parse_args_no_milkdrop_beat_transition(self) -> None:
        from config import parse_args
        cfg = parse_args(["--no-milkdrop-beat-transition"])
        assert cfg.milkdrop_beat_transition is False

    def test_parse_args_no_geiss_symmetry(self) -> None:
        from config import parse_args
        cfg = parse_args(["--no-geiss-symmetry"])
        assert cfg.geiss_use_symmetry is False

    def test_parse_args_no_geiss_plasma(self) -> None:
        from config import parse_args
        cfg = parse_args(["--no-geiss-plasma"])
        assert cfg.geiss_plasma_overlay is False

    def test_parse_args_mode_audio_tunnel(self) -> None:
        from config import parse_args
        cfg = parse_args(["--mode", "audio_tunnel"])
        assert cfg.default_mode == "audio_tunnel"

    def test_parse_args_tunnel_speed(self) -> None:
        from config import parse_args
        cfg = parse_args(["--tunnel-speed", "2.5"])
        assert cfg.tunnel_speed == pytest.approx(2.5)

    def test_parse_args_tunnel_lane_count(self) -> None:
        from config import parse_args
        cfg = parse_args(["--tunnel-lane-count", "5"])
        assert cfg.tunnel_lane_count == 5

    def test_parse_args_tunnel_glow_strength(self) -> None:
        from config import parse_args
        cfg = parse_args(["--tunnel-glow-strength", "0.8"])
        assert cfg.tunnel_glow_strength == pytest.approx(0.8)

    def test_json_config_mode_settings(self, tmp_path) -> None:
        import json
        from config import parse_args
        cfg_path = tmp_path / "test_cfg.json"
        cfg_path.write_text(json.dumps({
            "default_mode": "milkdrop",
            "milkdrop_auto_cycle": False,
            "milkdrop_cycle_seconds": 20.0,
            "geiss_use_symmetry": False,
            "geiss_plasma_overlay": False,
        }))
        cfg = parse_args(["--config", str(cfg_path)])
        assert cfg.default_mode == "milkdrop"
        assert cfg.milkdrop_auto_cycle is False
        assert cfg.milkdrop_cycle_seconds == pytest.approx(20.0)
        assert cfg.geiss_use_symmetry is False
        assert cfg.geiss_plasma_overlay is False

    def test_json_config_tunnel_settings(self, tmp_path) -> None:
        import json
        from config import parse_args
        cfg_path = tmp_path / "test_tunnel_cfg.json"
        cfg_path.write_text(json.dumps({
            "default_mode": "audio_tunnel",
            "tunnel_speed": 2.0,
            "tunnel_obstacle_density": 1.5,
            "tunnel_lane_count": 4,
            "tunnel_audio_sensitivity": 1.2,
            "tunnel_glow_strength": 0.7,
        }))
        cfg = parse_args(["--config", str(cfg_path)])
        assert cfg.default_mode == "audio_tunnel"
        assert cfg.tunnel_speed == pytest.approx(2.0)
        assert cfg.tunnel_obstacle_density == pytest.approx(1.5)
        assert cfg.tunnel_lane_count == 4
        assert cfg.tunnel_audio_sensitivity == pytest.approx(1.2)
        assert cfg.tunnel_glow_strength == pytest.approx(0.7)
