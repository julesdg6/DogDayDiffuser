"""
audio_reactivity.py — Audio capture and FFT-based feature extraction.

Provides:
  - AudioFeatures  dataclass holding extracted per-frame audio info
  - AudioReactor   captures live audio and computes features

Audio reactivity is entirely optional.  If ``sounddevice`` is not installed
or the audio device is unavailable the reactor raises an ImportError /
RuntimeError at construction time.  The main loop catches this and continues
without audio.

Feature extraction uses NumPy FFT only (no librosa / aubio dependency).

Frequency band boundaries (approximate, at 44100 Hz):
  sub-bass:  20 –  80 Hz
  bass:      80 – 300 Hz
  mid:      300 – 3000 Hz
  treble:  3000 – 8000 Hz
"""

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Audio capture settings
_SAMPLE_RATE = 44100
_BLOCK_SIZE = 1024   # frames per block (~23 ms at 44100 Hz)
_CHANNELS = 1


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AudioFeatures:
    """Extracted audio features for one analysis frame."""

    volume: float = 0.0       # RMS amplitude 0–1
    bass: float = 0.0         # Low-band energy 0–1
    mid: float = 0.0          # Mid-band energy 0–1
    treble: float = 0.0       # High-band energy 0–1
    beat: float = 0.0         # Beat / onset pulse 0–1 (decays over time)

    def as_dict(self) -> dict:
        return {
            "volume": self.volume,
            "bass": self.bass,
            "mid": self.mid,
            "treble": self.treble,
            "beat": self.beat,
        }


# ---------------------------------------------------------------------------
# Band energy helper
# ---------------------------------------------------------------------------


def _band_energy(magnitudes: np.ndarray, freqs: np.ndarray,
                 f_low: float, f_high: float) -> float:
    """Return RMS energy in the specified frequency band (normalised 0–1)."""
    mask = (freqs >= f_low) & (freqs < f_high)
    band = magnitudes[mask]
    if len(band) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(band ** 2)))
    return min(rms, 1.0)


# ---------------------------------------------------------------------------
# Main reactor class
# ---------------------------------------------------------------------------


class AudioReactor:
    """Captures audio from a microphone or line-in and extracts features.

    Requires the ``sounddevice`` package::

        pip install sounddevice

    Args:
        device:      Input device index (None = system default).
        sample_rate: Sampling rate in Hz.
        block_size:  Number of samples per analysis block.
        smooth:      EMA smoothing factor for all features (0 < smooth < 1).
    """

    def __init__(self, device=None, sample_rate: int = _SAMPLE_RATE,
                 block_size: int = _BLOCK_SIZE, smooth: float = 0.3):
        try:
            import sounddevice as sd  # type: ignore
            self._sd = sd
        except ImportError:
            raise ImportError(
                "sounddevice is not installed.  "
                "Install it with: pip install sounddevice\n"
                "Running without audio reactivity."
            )

        self._device = device
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._smooth = smooth

        # Frequency axis for the FFT output
        self._freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)

        # Shared state updated by the audio callback
        self._lock = threading.Lock()
        self._buffer: np.ndarray = np.zeros(block_size, dtype=np.float32)
        self._features = AudioFeatures()

        # Beat detection state
        self._prev_energy: float = 0.0
        self._beat_pulse: float = 0.0

        self._stream = None
        self._open_stream()

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    def _open_stream(self) -> None:
        """Open the sounddevice input stream."""
        try:
            self._stream = self._sd.InputStream(
                device=self._device,
                channels=_CHANNELS,
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            logger.info(
                "Audio stream opened — device=%s, rate=%d, block=%d",
                self._device, self._sample_rate, self._block_size,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not open audio stream: {exc}") from exc

    def _callback(self, indata: np.ndarray, frames: int,
                  time_info, status) -> None:
        """sounddevice callback — runs in an audio thread."""
        if status:
            logger.debug("Audio stream status: %s", status)

        mono = indata[:, 0].copy()
        with self._lock:
            self._buffer = mono
        self._analyse(mono)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _analyse(self, samples: np.ndarray) -> None:
        """Compute audio features from *samples* and store them thread-safely."""
        # Windowed FFT
        window = np.hanning(len(samples))
        spectrum = np.abs(np.fft.rfft(samples * window))
        # Normalise by number of samples
        spectrum /= max(len(samples), 1)

        # RMS volume
        rms = float(np.sqrt(np.mean(samples ** 2)))
        rms = min(rms * 4.0, 1.0)  # scale up for typical microphone levels

        bass   = _band_energy(spectrum, self._freqs, 80, 300) * 6.0
        mid    = _band_energy(spectrum, self._freqs, 300, 3000) * 4.0
        treble = _band_energy(spectrum, self._freqs, 3000, 8000) * 3.0

        bass   = min(bass, 1.0)
        mid    = min(mid, 1.0)
        treble = min(treble, 1.0)

        # Simple onset/beat detection: energy spike in bass band
        energy = bass
        delta = max(energy - self._prev_energy, 0.0)
        if delta > 0.25:
            self._beat_pulse = 1.0
        else:
            self._beat_pulse *= 0.85  # decay
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
    # Public API
    # ------------------------------------------------------------------

    def get_features(self) -> AudioFeatures:
        """Return a snapshot of the latest audio features (thread-safe copy)."""
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
        """Stop and close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio stream closed")
