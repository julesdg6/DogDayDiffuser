"""
audio/audio_manager.py — Unified audio manager for DogDayDiffuser.

Responsibilities:
  - Auto-detect USB audio input devices via ``audio.device_detection``.
  - Auto-detect HDMI audio output devices when passthrough is requested.
  - Open the appropriate stream:
      * duplex ``AudioPassthrough`` (input → analysis + HDMI output) when
        ``enable_passthrough=True``;
      * input-only ``AudioReactor`` otherwise.
  - Expose the same ``get_features()`` / ``close()`` API as ``AudioReactor``
    so ``main.py`` can use ``AudioManager`` as a drop-in replacement.

Startup log example::

    [INFO] Audio IN: USB Audio Device (hw:1,0)
    [INFO] Audio OUT: HDMI (hw:0,3)

If USB detection is enabled and a device is unplugged at runtime, the stream
callback will start receiving silence / status warnings; the manager logs these
and the main loop continues with the last known features.
"""

import logging
import threading
from typing import Optional

import numpy as np

from audio_reactivity import AudioFeatures, AudioReactor, _band_energy
from audio.device_detection import (
    find_usb_input_device,
    find_hdmi_output_device,
    get_device_label,
)
from audio.passthrough import AudioPassthrough

logger = logging.getLogger(__name__)


class AudioManager:
    """Manages audio input, optional HDMI passthrough, and feature extraction.

    Provides the same interface as :class:`~audio_reactivity.AudioReactor` so
    it can be used as a drop-in replacement in ``main.py``.

    Args:
        audio_device:       Explicit audio input device index.  When
                            ``auto_usb=True`` the USB device is preferred
                            over this value unless it is ``None``.
        auto_usb:           Auto-detect and use a connected USB sound card as
                            the audio input source.
        prefer_usb:         When ``True`` and a USB device is found, use it
                            even if *audio_device* was explicitly provided.
        output_prefer_hdmi: Auto-detect HDMI output for passthrough.
        enable_passthrough: Route input audio to the output device in real time.
        sample_rate:        Audio sampling rate in Hz.
        buffer_size:        Frames per processing block.
        smooth:             EMA smoothing factor (0 < smooth < 1) for features.
    """

    def __init__(
        self,
        audio_device: Optional[int] = None,
        auto_usb: bool = True,
        prefer_usb: bool = True,
        output_prefer_hdmi: bool = True,
        enable_passthrough: bool = False,
        sample_rate: int = 44100,
        buffer_size: int = 512,
        smooth: float = 0.3,
    ) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            raise ImportError(
                "sounddevice is not installed.  "
                "Install it with: pip install sounddevice\n"
                "Running without audio reactivity."
            )

        self._sd = sd
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._smooth = smooth
        self._enable_passthrough = enable_passthrough

        # Resolve input / output devices.
        self._input_device = self._resolve_input_device(
            audio_device, auto_usb, prefer_usb
        )
        self._output_device = (
            self._resolve_output_device(output_prefer_hdmi) if enable_passthrough else None
        )

        # Human-readable labels for overlay / log display.
        self.input_label: str = get_device_label(sd, self._input_device, "default")
        self.output_label: str = (
            get_device_label(sd, self._output_device, "default")
            if enable_passthrough
            else "disabled"
        )

        logger.info("Audio IN: %s", self.input_label)
        if enable_passthrough:
            logger.info("Audio OUT: %s", self.output_label)

        # Shared state for feature extraction (used in passthrough mode only).
        self._lock = threading.Lock()
        self._features = AudioFeatures()
        self._prev_energy: float = 0.0
        self._beat_pulse: float = 0.0
        self._freqs = np.fft.rfftfreq(buffer_size, d=1.0 / sample_rate)

        # Active streams (only one of these will be non-None at a time).
        self._passthrough: Optional[AudioPassthrough] = None
        self._reactor: Optional[AudioReactor] = None

        self._open()

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def _resolve_input_device(
        self,
        explicit: Optional[int],
        auto_usb: bool,
        prefer_usb: bool,
    ) -> Optional[int]:
        """Return the input device index to use."""
        if auto_usb:
            usb_idx = find_usb_input_device(self._sd)
            if usb_idx is not None and (prefer_usb or explicit is None):
                logger.info("Auto-selected USB audio input device index: %d", usb_idx)
                return usb_idx

        if explicit is not None:
            logger.info("Using explicit audio input device index: %d", explicit)
        else:
            logger.info("Using system default audio input device")
        return explicit  # may be None → system default

    def _resolve_output_device(self, prefer_hdmi: bool) -> Optional[int]:
        """Return the HDMI output device index (or None for system default).

        When *prefer_hdmi* is ``False``, HDMI detection is skipped and the
        system default output device will be used for passthrough.
        """
        if not prefer_hdmi:
            logger.info("HDMI output preference disabled; using system default output")
            return None
        hdmi_idx = find_hdmi_output_device(self._sd)
        if hdmi_idx is not None:
            return hdmi_idx
        logger.info(
            "No HDMI output device found; passthrough will use system default output"
        )
        return None  # system default

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the appropriate audio stream."""
        if self._enable_passthrough:
            self._passthrough = AudioPassthrough(
                sd=self._sd,
                input_device=self._input_device,
                output_device=self._output_device,
                sample_rate=self._sample_rate,
                block_size=self._buffer_size,
                analysis_callback=self._analyse,
            )
        else:
            self._reactor = AudioReactor(
                device=self._input_device,
                sample_rate=self._sample_rate,
                block_size=self._buffer_size,
                smooth=self._smooth,
            )

    # ------------------------------------------------------------------
    # Feature extraction (used when passthrough is active)
    # ------------------------------------------------------------------

    def _analyse(self, samples: np.ndarray) -> None:
        """FFT-based feature extraction — called from the audio thread.

        Mirrors the logic in ``AudioReactor._analyse`` so that passthrough mode
        produces identical ``AudioFeatures`` to input-only mode.
        """
        window = np.hanning(len(samples))
        spectrum = np.abs(np.fft.rfft(samples * window))
        spectrum /= max(len(samples), 1)

        rms = float(np.sqrt(np.mean(samples ** 2)))
        rms = min(rms * 4.0, 1.0)

        # Scale factors mirror AudioReactor._analyse to keep consistent feature levels
        # across both input-only and passthrough modes.
        bass   = min(_band_energy(spectrum, self._freqs, 80,   300)  * 6.0, 1.0)
        mid    = min(_band_energy(spectrum, self._freqs, 300,  3000) * 4.0, 1.0)
        treble = min(_band_energy(spectrum, self._freqs, 3000, 8000) * 3.0, 1.0)

        energy = bass
        delta = max(energy - self._prev_energy, 0.0)
        if delta > 0.25:
            self._beat_pulse = 1.0
        else:
            self._beat_pulse *= 0.85
        self._prev_energy = energy

        a = self._smooth
        with self._lock:
            f = self._features
            f.volume = a * rms    + (1 - a) * f.volume
            f.bass   = a * bass   + (1 - a) * f.bass
            f.mid    = a * mid    + (1 - a) * f.mid
            f.treble = a * treble + (1 - a) * f.treble
            f.beat   = self._beat_pulse

    # ------------------------------------------------------------------
    # Public API — mirrors AudioReactor
    # ------------------------------------------------------------------

    def get_features(self) -> AudioFeatures:
        """Return a snapshot of the latest audio features (thread-safe)."""
        if self._reactor is not None:
            return self._reactor.get_features()

        with self._lock:
            f = self._features
            return AudioFeatures(
                volume=f.volume,
                bass=f.bass,
                mid=f.mid,
                treble=f.treble,
                beat=f.beat,
            )

    def close(self) -> None:
        """Stop and close all audio streams."""
        if self._passthrough is not None:
            self._passthrough.close()
            self._passthrough = None
        if self._reactor is not None:
            self._reactor.close()
            self._reactor = None
        logger.info("AudioManager closed")
