"""
tests/test_audio.py — Unit tests for the audio package.

Tests cover:
  - device_detection: USB input and HDMI output enumeration (mocked sounddevice).
  - AudioPassthrough: duplex stream management (mocked sounddevice).
  - AudioManager: device resolution and stream selection (mocked sounddevice).
  - config: new audio config fields.

No hardware is required — all sounddevice calls are mocked.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sd_mock(devices: List[Dict[str, Any]]) -> MagicMock:
    """Return a sounddevice mock whose query_devices() returns *devices*."""
    sd = MagicMock()
    sd.query_devices.return_value = devices
    return sd


def _device(name: str, inputs: int = 2, outputs: int = 0) -> Dict[str, Any]:
    """Build a minimal device dict like sounddevice returns."""
    return {
        "name": name,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


# ---------------------------------------------------------------------------
# audio.device_detection — find_usb_input_device
# ---------------------------------------------------------------------------


class TestFindUsbInputDevice:
    def test_returns_none_when_no_devices(self) -> None:
        from audio.device_detection import find_usb_input_device

        sd = _make_sd_mock([])
        assert find_usb_input_device(sd) is None

    def test_finds_usb_device_by_name(self) -> None:
        from audio.device_detection import find_usb_input_device

        devices = [
            _device("Built-in Microphone", inputs=1),
            _device("USB Audio Device", inputs=2),
        ]
        sd = _make_sd_mock(devices)
        assert find_usb_input_device(sd) == 1

    def test_finds_focusrite_by_name(self) -> None:
        from audio.device_detection import find_usb_input_device

        devices = [
            _device("HDMI Output", inputs=0, outputs=2),
            _device("Focusrite Scarlett Solo USB", inputs=2),
        ]
        sd = _make_sd_mock(devices)
        assert find_usb_input_device(sd) == 1

    def test_skips_device_with_no_input_channels(self) -> None:
        from audio.device_detection import find_usb_input_device

        devices = [
            _device("USB Output Only", inputs=0, outputs=2),
            _device("Regular Mic", inputs=1),
        ]
        sd = _make_sd_mock(devices)
        # "USB Output Only" has 0 inputs so should be skipped;
        # "Regular Mic" has no USB keyword → None.
        assert find_usb_input_device(sd) is None

    def test_returns_first_match(self) -> None:
        from audio.device_detection import find_usb_input_device

        devices = [
            _device("USB Mic A", inputs=1),
            _device("USB Mic B", inputs=1),
        ]
        sd = _make_sd_mock(devices)
        assert find_usb_input_device(sd) == 0

    def test_returns_none_when_no_usb_device(self) -> None:
        from audio.device_detection import find_usb_input_device

        devices = [
            _device("Built-in Microphone", inputs=1),
            _device("Internal Speaker", inputs=0, outputs=2),
        ]
        sd = _make_sd_mock(devices)
        assert find_usb_input_device(sd) is None

    def test_returns_none_on_query_failure(self) -> None:
        from audio.device_detection import find_usb_input_device

        sd = MagicMock()
        sd.query_devices.side_effect = RuntimeError("no audio")
        assert find_usb_input_device(sd) is None

    def test_case_insensitive_match(self) -> None:
        from audio.device_detection import find_usb_input_device

        devices = [_device("USB AUDIO DEVICE", inputs=1)]
        sd = _make_sd_mock(devices)
        assert find_usb_input_device(sd) == 0


# ---------------------------------------------------------------------------
# audio.device_detection — find_hdmi_output_device
# ---------------------------------------------------------------------------


class TestFindHdmiOutputDevice:
    def test_returns_none_when_no_devices(self) -> None:
        from audio.device_detection import find_hdmi_output_device

        sd = _make_sd_mock([])
        assert find_hdmi_output_device(sd) is None

    def test_finds_hdmi_by_name(self) -> None:
        from audio.device_detection import find_hdmi_output_device

        devices = [
            _device("Built-in Speakers", inputs=0, outputs=2),
            _device("HDMI Output", inputs=0, outputs=2),
        ]
        sd = _make_sd_mock(devices)
        assert find_hdmi_output_device(sd) == 1

    def test_skips_hdmi_device_with_no_outputs(self) -> None:
        from audio.device_detection import find_hdmi_output_device

        devices = [
            _device("HDMI Input", inputs=2, outputs=0),
            _device("Regular Output", inputs=0, outputs=2),
        ]
        sd = _make_sd_mock(devices)
        # Only input HDMI → skipped; "Regular Output" has no HDMI keyword → None.
        assert find_hdmi_output_device(sd) is None

    def test_returns_first_hdmi_output(self) -> None:
        from audio.device_detection import find_hdmi_output_device

        devices = [
            _device("HDMI-A", inputs=0, outputs=2),
            _device("HDMI-B", inputs=0, outputs=2),
        ]
        sd = _make_sd_mock(devices)
        assert find_hdmi_output_device(sd) == 0

    def test_returns_none_when_no_hdmi(self) -> None:
        from audio.device_detection import find_hdmi_output_device

        devices = [_device("Built-in Speakers", inputs=0, outputs=2)]
        sd = _make_sd_mock(devices)
        assert find_hdmi_output_device(sd) is None

    def test_returns_none_on_query_failure(self) -> None:
        from audio.device_detection import find_hdmi_output_device

        sd = MagicMock()
        sd.query_devices.side_effect = RuntimeError("error")
        assert find_hdmi_output_device(sd) is None

    def test_displayport_keyword(self) -> None:
        from audio.device_detection import find_hdmi_output_device

        devices = [_device("DisplayPort Audio", inputs=0, outputs=2)]
        sd = _make_sd_mock(devices)
        assert find_hdmi_output_device(sd) == 0


# ---------------------------------------------------------------------------
# audio.device_detection — get_device_label
# ---------------------------------------------------------------------------


class TestGetDeviceLabel:
    def test_returns_default_when_index_is_none(self) -> None:
        from audio.device_detection import get_device_label

        sd = _make_sd_mock([])
        assert get_device_label(sd, None) == "default"

    def test_returns_custom_default(self) -> None:
        from audio.device_detection import get_device_label

        sd = _make_sd_mock([])
        assert get_device_label(sd, None, "system default") == "system default"

    def test_returns_device_name(self) -> None:
        from audio.device_detection import get_device_label

        devices = [_device("USB Mic"), _device("HDMI Out", inputs=0, outputs=2)]
        sd = _make_sd_mock(devices)
        assert get_device_label(sd, 0) == "USB Mic"
        assert get_device_label(sd, 1) == "HDMI Out"

    def test_returns_fallback_on_invalid_index(self) -> None:
        from audio.device_detection import get_device_label

        devices = [_device("USB Mic")]
        sd = _make_sd_mock(devices)
        assert get_device_label(sd, 99) == "device 99"

    def test_returns_fallback_on_query_error(self) -> None:
        from audio.device_detection import get_device_label

        sd = MagicMock()
        sd.query_devices.side_effect = RuntimeError("error")
        assert get_device_label(sd, 0) == "device 0"


# ---------------------------------------------------------------------------
# audio.passthrough — AudioPassthrough
# ---------------------------------------------------------------------------


class TestAudioPassthrough:
    def _make_passthrough(self, analysis_callback=None):
        from audio.passthrough import AudioPassthrough

        sd = MagicMock()
        mock_stream = MagicMock()
        sd.Stream.return_value = mock_stream

        pt = AudioPassthrough(
            sd=sd,
            input_device=1,
            output_device=2,
            sample_rate=44100,
            block_size=512,
            analysis_callback=analysis_callback,
        )
        return pt, sd, mock_stream

    def test_stream_opened_on_init(self) -> None:
        pt, sd, mock_stream = self._make_passthrough()
        sd.Stream.assert_called_once()
        mock_stream.start.assert_called_once()

    def test_close_stops_stream(self) -> None:
        pt, sd, mock_stream = self._make_passthrough()
        pt.close()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert pt._stream is None

    def test_close_is_idempotent(self) -> None:
        pt, sd, mock_stream = self._make_passthrough()
        pt.close()
        pt.close()  # should not raise
        assert mock_stream.stop.call_count == 1

    def test_callback_copies_input_to_output(self) -> None:
        pt, _, _ = self._make_passthrough()

        indata = np.ones((512, 1), dtype=np.float32) * 0.5
        outdata = np.zeros((512, 1), dtype=np.float32)
        pt._callback(indata, outdata, 512, None, None)

        np.testing.assert_array_equal(outdata, indata)

    def test_callback_calls_analysis(self) -> None:
        received = []

        def on_audio(samples):
            received.append(samples.copy())

        pt, _, _ = self._make_passthrough(analysis_callback=on_audio)
        indata = np.ones((512, 1), dtype=np.float32) * 0.3
        outdata = np.zeros((512, 1), dtype=np.float32)
        pt._callback(indata, outdata, 512, None, None)

        assert len(received) == 1
        assert received[0].shape == (512,)
        assert float(received[0][0]) == pytest.approx(0.3)

    def test_callback_no_analysis_callback(self) -> None:
        """Callback should not raise when no analysis callback is set."""
        pt, _, _ = self._make_passthrough(analysis_callback=None)
        indata = np.zeros((512, 1), dtype=np.float32)
        outdata = np.zeros((512, 1), dtype=np.float32)
        pt._callback(indata, outdata, 512, None, None)  # should not raise

    def test_raises_on_stream_open_failure(self) -> None:
        from audio.passthrough import AudioPassthrough

        sd = MagicMock()
        sd.Stream.side_effect = RuntimeError("no device")

        with pytest.raises(RuntimeError, match="Could not open audio passthrough"):
            AudioPassthrough(sd=sd, input_device=99)

    def test_stream_configured_with_correct_params(self) -> None:
        pt, sd, _ = self._make_passthrough()
        _, kwargs = sd.Stream.call_args
        assert kwargs["samplerate"] == 44100
        assert kwargs["blocksize"] == 512
        assert kwargs["dtype"] == "float32"
        assert kwargs["device"] == (1, 2)


# ---------------------------------------------------------------------------
# audio.audio_manager — AudioManager
# ---------------------------------------------------------------------------


def _mock_sounddevice(devices: List[Dict[str, Any]]) -> MagicMock:
    sd = MagicMock()
    sd.query_devices.return_value = devices
    mock_stream = MagicMock()
    sd.InputStream.return_value = mock_stream
    sd.Stream.return_value = mock_stream
    return sd


class TestAudioManagerDeviceResolution:
    def _make_manager_resolve_input(self, devices, **kwargs):
        """Directly test _resolve_input_device by setting up a partial manager."""
        from audio.audio_manager import AudioManager

        sd_mock = _mock_sounddevice(devices)
        reactor_mock = MagicMock()

        with patch.dict("sys.modules", {"sounddevice": sd_mock}), \
             patch("audio.audio_manager.AudioReactor", return_value=reactor_mock):
            mgr = AudioManager(
                audio_device=kwargs.get("audio_device"),
                auto_usb=kwargs.get("auto_usb", True),
                prefer_usb=kwargs.get("prefer_usb", True),
                enable_passthrough=False,
            )
        return mgr

    def test_usb_device_selected_when_auto_usb(self) -> None:
        devices = [
            _device("Built-in Mic", inputs=1),
            _device("USB Audio Device", inputs=2),
        ]
        mgr = self._make_manager_resolve_input(devices, auto_usb=True)
        assert mgr._input_device == 1

    def test_explicit_device_used_when_no_usb_found(self) -> None:
        devices = [_device("Built-in Mic", inputs=1)]
        mgr = self._make_manager_resolve_input(devices, audio_device=0, auto_usb=True)
        assert mgr._input_device == 0

    def test_usb_preferred_over_explicit(self) -> None:
        devices = [
            _device("Explicit Device", inputs=1),
            _device("USB Sound Card", inputs=2),
        ]
        mgr = self._make_manager_resolve_input(
            devices, audio_device=0, auto_usb=True, prefer_usb=True
        )
        assert mgr._input_device == 1

    def test_explicit_used_when_prefer_usb_false(self) -> None:
        devices = [
            _device("Explicit Device", inputs=1),
            _device("USB Sound Card", inputs=2),
        ]
        mgr = self._make_manager_resolve_input(
            devices, audio_device=0, auto_usb=True, prefer_usb=False
        )
        # prefer_usb=False: explicit device wins even if USB is found.
        assert mgr._input_device == 0

    def test_none_device_when_nothing_found(self) -> None:
        devices = [_device("Built-in Mic", inputs=1)]
        mgr = self._make_manager_resolve_input(devices, auto_usb=True, audio_device=None)
        # No USB device found and no explicit device → system default (None).
        assert mgr._input_device is None


class TestAudioManagerInit:
    def _patch_and_create(self, devices, **kwargs):
        """Create an AudioManager with sounddevice mocked at import time."""
        from audio.audio_manager import AudioManager

        sd_mock = _mock_sounddevice(devices)
        reactor_mock = MagicMock()
        reactor_mock.get_features.return_value = MagicMock(
            volume=0.0, bass=0.0, mid=0.0, treble=0.0, beat=0.0
        )

        with patch.dict("sys.modules", {"sounddevice": sd_mock}), \
             patch("audio.audio_manager.AudioReactor", return_value=reactor_mock), \
             patch("audio.audio_manager.AudioPassthrough") as mock_pt:
            mock_pt_instance = MagicMock()
            mock_pt.return_value = mock_pt_instance
            mgr = AudioManager(**kwargs)
            return mgr, sd_mock, reactor_mock, mock_pt, mock_pt_instance

    def test_uses_reactor_when_passthrough_disabled(self) -> None:
        devices = [_device("USB Mic", inputs=1)]
        mgr, _, reactor_mock, mock_pt, _ = self._patch_and_create(
            devices, enable_passthrough=False
        )
        assert mgr._reactor is reactor_mock
        assert mgr._passthrough is None
        mock_pt.assert_not_called()

    def test_uses_passthrough_when_enabled(self) -> None:
        devices = [_device("USB Mic", inputs=1)]
        mgr, _, reactor_mock, mock_pt, mock_pt_instance = self._patch_and_create(
            devices, enable_passthrough=True
        )
        assert mgr._passthrough is mock_pt_instance
        assert mgr._reactor is None
        mock_pt.assert_called_once()

    def test_get_features_delegates_to_reactor(self) -> None:
        from audio_reactivity import AudioFeatures

        devices = [_device("USB Mic", inputs=1)]
        mgr, _, reactor_mock, _, _ = self._patch_and_create(
            devices, enable_passthrough=False
        )
        expected = AudioFeatures(volume=0.5, bass=0.3)
        reactor_mock.get_features.return_value = expected
        result = mgr.get_features()
        assert result is expected

    def test_close_calls_reactor_close(self) -> None:
        devices = [_device("USB Mic", inputs=1)]
        mgr, _, reactor_mock, _, _ = self._patch_and_create(
            devices, enable_passthrough=False
        )
        mgr.close()
        reactor_mock.close.assert_called_once()
        assert mgr._reactor is None

    def test_close_calls_passthrough_close(self) -> None:
        devices = [_device("USB Mic", inputs=1)]
        mgr, _, _, _, mock_pt_instance = self._patch_and_create(
            devices, enable_passthrough=True
        )
        mgr.close()
        mock_pt_instance.close.assert_called_once()
        assert mgr._passthrough is None

    def test_raises_import_error_without_sounddevice(self) -> None:
        from audio.audio_manager import AudioManager

        with patch.dict("sys.modules", {"sounddevice": None}):
            with pytest.raises(ImportError):
                AudioManager()

    def test_input_label_set(self) -> None:
        devices = [_device("USB Audio CODEC", inputs=2)]
        mgr, _, _, _, _ = self._patch_and_create(devices)
        assert "USB Audio CODEC" in mgr.input_label

    def test_output_label_disabled_when_no_passthrough(self) -> None:
        devices = [_device("USB Mic", inputs=1)]
        mgr, _, _, _, _ = self._patch_and_create(devices, enable_passthrough=False)
        assert mgr.output_label == "disabled"

    def test_hdmi_detection_skipped_when_prefer_hdmi_false(self) -> None:
        devices = [
            _device("USB Mic", inputs=1),
            _device("HDMI Output", inputs=0, outputs=2),
        ]
        mgr, sd_mock, _, _, _ = self._patch_and_create(
            devices, enable_passthrough=True, output_prefer_hdmi=False
        )
        # output_prefer_hdmi=False: no HDMI detection attempted.
        assert mgr._output_device is None


# ---------------------------------------------------------------------------
# audio.audio_manager — _analyse (feature extraction in passthrough mode)
# ---------------------------------------------------------------------------


class TestAudioManagerAnalyse:
    def _create_manager_for_analyse(self):
        from audio.audio_manager import AudioManager

        sd_mock = _mock_sounddevice([])
        reactor_mock = MagicMock()

        with patch.dict("sys.modules", {"sounddevice": sd_mock}), \
             patch("audio.audio_manager.AudioReactor", return_value=reactor_mock):
            mgr = AudioManager(enable_passthrough=False)

        # Swap in a passthrough mock so get_features uses self._lock path.
        mgr._reactor = None
        return mgr

    def test_analyse_updates_features(self) -> None:
        mgr = self._create_manager_for_analyse()
        samples = np.sin(
            2 * np.pi * 440 * np.arange(512) / 44100
        ).astype(np.float32)
        mgr._analyse(samples)
        f = mgr.get_features()
        assert 0.0 <= f.volume <= 1.0
        assert 0.0 <= f.bass <= 1.0
        assert 0.0 <= f.treble <= 1.0

    def test_beat_pulse_decays(self) -> None:
        mgr = self._create_manager_for_analyse()
        # Force a beat
        mgr._beat_pulse = 1.0
        silence = np.zeros(512, dtype=np.float32)
        mgr._analyse(silence)
        assert mgr._beat_pulse < 1.0

    def test_analyse_silent_frame(self) -> None:
        mgr = self._create_manager_for_analyse()
        silence = np.zeros(512, dtype=np.float32)
        mgr._analyse(silence)
        f = mgr.get_features()
        assert f.volume == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# config — new audio fields
# ---------------------------------------------------------------------------


class TestAudioConfig:
    def test_defaults(self) -> None:
        from config import AppConfig

        cfg = AppConfig()
        assert cfg.audio_auto_usb is True
        assert cfg.audio_prefer_usb is True
        assert cfg.audio_output_prefer_hdmi is True
        assert cfg.audio_buffer_size == 512
        assert cfg.audio_sample_rate == 44100
        assert cfg.audio_enable_passthrough is False

    def test_parse_args_no_audio_auto_usb(self) -> None:
        from config import parse_args

        cfg = parse_args(["--no-audio-auto-usb"])
        assert cfg.audio_auto_usb is False

    def test_parse_args_no_audio_hdmi_out(self) -> None:
        from config import parse_args

        cfg = parse_args(["--no-audio-hdmi-out"])
        assert cfg.audio_output_prefer_hdmi is False

    def test_parse_args_audio_passthrough(self) -> None:
        from config import parse_args

        cfg = parse_args(["--audio-passthrough"])
        assert cfg.audio_enable_passthrough is True

    def test_parse_args_audio_buffer_size(self) -> None:
        from config import parse_args

        cfg = parse_args(["--audio-buffer-size", "1024"])
        assert cfg.audio_buffer_size == 1024

    def test_parse_args_audio_sample_rate(self) -> None:
        from config import parse_args

        cfg = parse_args(["--audio-sample-rate", "48000"])
        assert cfg.audio_sample_rate == 48000

    def test_json_config_audio_fields(self, tmp_path) -> None:
        import json
        from config import parse_args

        cfg_path = tmp_path / "audio_cfg.json"
        cfg_path.write_text(json.dumps({
            "audio_auto_usb": False,
            "audio_prefer_usb": False,
            "audio_output_prefer_hdmi": False,
            "audio_buffer_size": 256,
            "audio_sample_rate": 48000,
            "audio_enable_passthrough": True,
        }))
        cfg = parse_args(["--config", str(cfg_path)])
        assert cfg.audio_auto_usb is False
        assert cfg.audio_prefer_usb is False
        assert cfg.audio_output_prefer_hdmi is False
        assert cfg.audio_buffer_size == 256
        assert cfg.audio_sample_rate == 48000
        assert cfg.audio_enable_passthrough is True
