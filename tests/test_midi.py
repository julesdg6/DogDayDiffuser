"""
tests/test_midi.py — Unit tests for the DogDayDiffuser MIDI subsystem.

These tests do not require hardware or the mido/rtmidi packages.
They mock the mido interface to validate:
  - CC-to-parameter mapping
  - Transport CC handling
  - Solo/Mute/Rec button toggling
  - Note On/Off handling
  - Device auto-detection
  - Graceful degradation when mido is not installed
  - MidiState snapshot correctness
  - Config integration
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cc(control: int, value: int) -> MagicMock:
    """Create a fake mido CC message."""
    msg = MagicMock()
    msg.type = "control_change"
    msg.control = control
    msg.value = value
    return msg


def _make_note(note: int, velocity: int = 127, msg_type: str = "note_on") -> MagicMock:
    """Create a fake mido Note message."""
    msg = MagicMock()
    msg.type = msg_type
    msg.note = note
    msg.velocity = velocity
    return msg


# ---------------------------------------------------------------------------
# midi.nanokontrol2 constants
# ---------------------------------------------------------------------------


class TestNanoKontrol2Constants:
    def test_fader_cc_count(self) -> None:
        from midi.nanokontrol2 import FADER_CC
        assert len(FADER_CC) == 8

    def test_knob_cc_count(self) -> None:
        from midi.nanokontrol2 import KNOB_CC
        assert len(KNOB_CC) == 8

    def test_button_cc_counts(self) -> None:
        from midi.nanokontrol2 import SOLO_CC, MUTE_CC, REC_CC
        assert len(SOLO_CC) == 8
        assert len(MUTE_CC) == 8
        assert len(REC_CC) == 8

    def test_no_cc_collisions(self) -> None:
        """All CC numbers must be unique across faders, knobs, and buttons."""
        from midi.nanokontrol2 import (
            FADER_CC, KNOB_CC, SOLO_CC, MUTE_CC, REC_CC,
            TRANSPORT_PLAY_CC, TRANSPORT_STOP_CC, TRANSPORT_REC_CC,
            TRANSPORT_BACK_CC, TRANSPORT_FWD_CC, TRANSPORT_LOOP_CC,
        )
        all_ccs: List[int] = (
            FADER_CC + KNOB_CC + SOLO_CC + MUTE_CC + REC_CC
            + [TRANSPORT_PLAY_CC, TRANSPORT_STOP_CC, TRANSPORT_REC_CC,
               TRANSPORT_BACK_CC, TRANSPORT_FWD_CC, TRANSPORT_LOOP_CC]
        )
        assert len(all_ccs) == len(set(all_ccs)), "Duplicate CC numbers found"


# ---------------------------------------------------------------------------
# midi.mapping — DEFAULT_CC_MAP and MidiState
# ---------------------------------------------------------------------------


class TestDefaultCCMap:
    def test_fader_ccs_mapped(self) -> None:
        from midi.mapping import DEFAULT_CC_MAP
        from midi.nanokontrol2 import FADER_CC
        for cc in FADER_CC:
            assert cc in DEFAULT_CC_MAP, f"Fader CC {cc} not in DEFAULT_CC_MAP"

    def test_knob_ccs_mapped(self) -> None:
        from midi.mapping import DEFAULT_CC_MAP
        from midi.nanokontrol2 import KNOB_CC
        for cc in KNOB_CC:
            assert cc in DEFAULT_CC_MAP, f"Knob CC {cc} not in DEFAULT_CC_MAP"

    def test_fader_1_maps_to_master_intensity(self) -> None:
        from midi.mapping import DEFAULT_CC_MAP
        from midi.nanokontrol2 import FADER_CC
        assert DEFAULT_CC_MAP[FADER_CC[0]] == "master_intensity"

    def test_fader_2_maps_to_warp_amount(self) -> None:
        from midi.mapping import DEFAULT_CC_MAP
        from midi.nanokontrol2 import FADER_CC
        assert DEFAULT_CC_MAP[FADER_CC[1]] == "warp_amount"

    def test_fader_3_maps_to_feedback_decay(self) -> None:
        from midi.mapping import DEFAULT_CC_MAP
        from midi.nanokontrol2 import FADER_CC
        assert DEFAULT_CC_MAP[FADER_CC[2]] == "feedback_decay"

    def test_knob_1_maps_to_warp_speed(self) -> None:
        from midi.mapping import DEFAULT_CC_MAP
        from midi.nanokontrol2 import KNOB_CC
        assert DEFAULT_CC_MAP[KNOB_CC[0]] == "warp_speed"

    def test_all_mapped_params_exist_in_midi_state(self) -> None:
        from midi.mapping import DEFAULT_CC_MAP
        from midi.mapping import MidiState
        state = MidiState()
        for cc, param in DEFAULT_CC_MAP.items():
            assert hasattr(state, param), (
                f"CC {cc} maps to '{param}' but MidiState has no such attribute"
            )


class TestMidiState:
    def test_default_values_in_range(self) -> None:
        from midi.mapping import MidiState
        state = MidiState()
        for key, val in state.as_dict().items():
            if isinstance(val, float):
                assert 0.0 <= val <= 1.0, f"{key} default {val} out of [0, 1]"

    def test_as_dict_keys(self) -> None:
        from midi.mapping import MidiState
        d = MidiState().as_dict()
        assert "master_intensity" in d
        assert "warp_amount" in d
        assert "feedback_decay" in d
        assert "brightness" in d


# ---------------------------------------------------------------------------
# midi.midi_manager — MidiManager (mocked mido)
# ---------------------------------------------------------------------------


class TestMidiManagerHandleCC:
    """Test _handle_cc without a real MIDI port."""

    def _get_manager(self):
        from midi.midi_manager import MidiManager
        return MidiManager()

    def test_fader_1_updates_master_intensity(self) -> None:
        mgr = self._get_manager()
        mgr._handle_cc(0, 127)  # Fader 1 full
        assert abs(mgr._state.master_intensity - 1.0) < 1e-6

    def test_fader_1_half_value(self) -> None:
        mgr = self._get_manager()
        mgr._handle_cc(0, 64)
        assert abs(mgr._state.master_intensity - 64 / 127.0) < 1e-4

    def test_fader_2_updates_warp_amount(self) -> None:
        mgr = self._get_manager()
        mgr._handle_cc(1, 0)
        assert abs(mgr._state.warp_amount - 0.0) < 1e-6

    def test_knob_1_updates_warp_speed(self) -> None:
        mgr = self._get_manager()
        mgr._handle_cc(16, 127)
        assert abs(mgr._state.warp_speed - 1.0) < 1e-6

    def test_knob_8_updates_brightness(self) -> None:
        mgr = self._get_manager()
        mgr._handle_cc(23, 0)
        assert abs(mgr._state.brightness - 0.0) < 1e-6

    def test_solo_button_toggles(self) -> None:
        mgr = self._get_manager()
        from midi.nanokontrol2 import SOLO_CC
        mgr._handle_cc(SOLO_CC[0], 127)
        assert mgr._state.solo_active.get(0) is True
        mgr._handle_cc(SOLO_CC[0], 127)
        assert mgr._state.solo_active.get(0) is False

    def test_mute_button_toggles(self) -> None:
        mgr = self._get_manager()
        from midi.nanokontrol2 import MUTE_CC
        mgr._handle_cc(MUTE_CC[3], 127)
        assert mgr._state.mute_active.get(3) is True

    def test_rec_button_toggles(self) -> None:
        mgr = self._get_manager()
        from midi.nanokontrol2 import REC_CC
        mgr._handle_cc(REC_CC[7], 127)
        assert mgr._state.rec_active.get(7) is True

    def test_button_low_value_does_not_toggle(self) -> None:
        """Buttons only act on value 127 (press), not on value 0 (release)."""
        mgr = self._get_manager()
        from midi.nanokontrol2 import SOLO_CC
        mgr._handle_cc(SOLO_CC[0], 0)
        assert mgr._state.solo_active.get(0, False) is False

    def test_unknown_cc_is_ignored(self) -> None:
        mgr = self._get_manager()
        # CC 99 is not assigned; should not raise
        mgr._handle_cc(99, 64)


class TestMidiManagerTransport:
    def _get_manager(self):
        from midi.midi_manager import MidiManager
        return MidiManager()

    def test_play_sets_playing_true(self) -> None:
        from midi.nanokontrol2 import TRANSPORT_PLAY_CC
        mgr = self._get_manager()
        mgr._handle_cc(TRANSPORT_PLAY_CC, 127)
        assert mgr._state.playing is True

    def test_stop_sets_playing_false(self) -> None:
        from midi.nanokontrol2 import TRANSPORT_STOP_CC
        mgr = self._get_manager()
        mgr._state.playing = True
        mgr._handle_cc(TRANSPORT_STOP_CC, 127)
        assert mgr._state.playing is False

    def test_rec_toggles_recording(self) -> None:
        from midi.nanokontrol2 import TRANSPORT_REC_CC
        mgr = self._get_manager()
        mgr._handle_cc(TRANSPORT_REC_CC, 127)
        assert mgr._state.recording is True
        mgr._handle_cc(TRANSPORT_REC_CC, 127)
        assert mgr._state.recording is False

    def test_transport_low_value_is_ignored(self) -> None:
        from midi.nanokontrol2 import TRANSPORT_PLAY_CC
        mgr = self._get_manager()
        mgr._handle_cc(TRANSPORT_PLAY_CC, 0)
        assert mgr._state.playing is False


class TestMidiManagerHandleMessage:
    def _get_manager(self):
        from midi.midi_manager import MidiManager
        return MidiManager()

    def test_cc_message_dispatched(self) -> None:
        mgr = self._get_manager()
        msg = _make_cc(0, 127)
        mgr._handle_message(msg)
        assert abs(mgr._state.master_intensity - 1.0) < 1e-6

    def test_note_on_does_not_raise(self) -> None:
        mgr = self._get_manager()
        msg = _make_note(60, 127, "note_on")
        mgr._handle_message(msg)  # should not raise

    def test_note_off_does_not_raise(self) -> None:
        mgr = self._get_manager()
        msg = _make_note(60, 0, "note_off")
        mgr._handle_message(msg)


class TestMidiManagerGetState:
    def _get_manager(self):
        from midi.midi_manager import MidiManager
        return MidiManager()

    def test_get_state_returns_copy(self) -> None:
        mgr = self._get_manager()
        s1 = mgr.get_state()
        mgr._handle_cc(0, 127)  # mutate internal state
        s2 = mgr.get_state()
        # s1 was a snapshot; its value should be unchanged
        assert s1.master_intensity != s2.master_intensity

    def test_get_state_contains_expected_fields(self) -> None:
        mgr = self._get_manager()
        state = mgr.get_state()
        assert hasattr(state, "master_intensity")
        assert hasattr(state, "warp_amount")
        assert hasattr(state, "feedback_decay")
        assert hasattr(state, "playing")
        assert hasattr(state, "recording")


class TestMidiManagerDeviceDetection:
    def test_find_device_returns_matching_port(self) -> None:
        from midi.midi_manager import MidiManager
        mgr = MidiManager(device_name="nanoKONTROL2")
        with patch("mido.get_input_names", return_value=["nanoKONTROL2 MIDI 1"]):
            port = mgr.find_device()
        assert port == "nanoKONTROL2 MIDI 1"

    def test_find_device_case_insensitive(self) -> None:
        from midi.midi_manager import MidiManager
        mgr = MidiManager(device_name="nanokontrol2")
        with patch("mido.get_input_names", return_value=["nanoKONTROL2 MIDI 1"]):
            port = mgr.find_device()
        assert port is not None

    def test_find_device_returns_none_when_absent(self) -> None:
        from midi.midi_manager import MidiManager
        mgr = MidiManager(device_name="nanoKONTROL2")
        with patch("mido.get_input_names", return_value=["Some Other Device"]):
            port = mgr.find_device()
        assert port is None

    def test_list_ports_delegates_to_mido(self) -> None:
        from midi.midi_manager import MidiManager
        expected = ["Port A", "Port B"]
        with patch("mido.get_input_names", return_value=expected):
            ports = MidiManager.list_ports()
        assert ports == expected


class TestMidiManagerOpenClose:
    def test_open_raises_when_no_device_found(self) -> None:
        from midi.midi_manager import MidiManager
        mgr = MidiManager()
        with (
            patch("mido.get_input_names", return_value=[]),
            pytest.raises(RuntimeError, match="No MIDI device"),
        ):
            mgr.open()

    def test_open_starts_reader_thread(self) -> None:
        from midi.midi_manager import MidiManager
        import threading

        # Use an event to block the reader thread until we've checked the port
        ready = threading.Event()
        mock_port = MagicMock()
        mock_port.__iter__ = MagicMock(return_value=iter([]))
        # close() unblocks the reader loop via _running = False; mock it
        mock_port.close = MagicMock()

        with (
            patch("mido.get_input_names", return_value=["nanoKONTROL2 MIDI 1"]),
            patch("mido.open_input", return_value=mock_port),
        ):
            mgr = MidiManager()
            mgr.open()
            # Thread may finish quickly (empty iterator), but port was opened
            assert mgr._thread is not None
            mgr.close()
            assert mgr._port is None

    def test_close_is_idempotent(self) -> None:
        from midi.midi_manager import MidiManager
        mgr = MidiManager()
        mgr.close()  # should not raise even when never opened
        mgr.close()

    def test_open_raises_import_error_without_mido(self) -> None:
        """If mido is not installed, open() should propagate ImportError."""
        import builtins
        original_import = builtins.__import__

        def _block_mido(name, *args, **kwargs):
            if name == "mido":
                raise ImportError("No module named 'mido'")
            return original_import(name, *args, **kwargs)

        from midi.midi_manager import MidiManager
        mgr = MidiManager()
        with (
            patch("builtins.__import__", side_effect=_block_mido),
            pytest.raises(ImportError),
        ):
            mgr.open()


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigMidiFields:
    def test_midi_disabled_by_default(self) -> None:
        from config import parse_args
        cfg = parse_args([])
        assert cfg.midi_enabled is False

    def test_midi_flag_enables_midi(self) -> None:
        from config import parse_args
        cfg = parse_args(["--midi"])
        assert cfg.midi_enabled is True

    def test_midi_device_default(self) -> None:
        from config import parse_args
        cfg = parse_args([])
        assert cfg.midi_device == "nanoKONTROL2"

    def test_midi_device_override(self) -> None:
        from config import parse_args
        cfg = parse_args(["--midi-device", "MyController"])
        assert cfg.midi_device == "MyController"

    def test_mapping_profile_default(self) -> None:
        from config import parse_args
        cfg = parse_args([])
        assert cfg.mapping_profile == "default_vj"

    def test_mapping_profile_override(self) -> None:
        from config import parse_args
        cfg = parse_args(["--mapping-profile", "custom"])
        assert cfg.mapping_profile == "custom"

    def test_json_config_can_set_midi_enabled(self, tmp_path) -> None:
        import json
        from config import parse_args
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({"midi_enabled": True}))
        cfg = parse_args(["--config", str(cfg_file)])
        assert cfg.midi_enabled is True
