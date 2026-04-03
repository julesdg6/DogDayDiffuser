"""
audio/passthrough.py — Low-latency audio passthrough via a sounddevice duplex stream.

Routes USB audio input simultaneously to:
  1. An *analysis callback* (non-blocking copy, for visual reactivity).
  2. An HDMI (or default) audio **output** (live monitoring / playback).

The duplex ``sounddevice.Stream`` is used instead of a plain ``InputStream``
so that both directions share a single low-latency callback.

Example usage::

    def on_audio(mono_samples):
        features = analyse(mono_samples)

    pt = AudioPassthrough(sd, input_device=2, output_device=5,
                          analysis_callback=on_audio)
    # … run main loop …
    pt.close()
"""

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioPassthrough:
    """Duplex audio stream that passes input straight to output.

    The *analysis_callback* is called with a mono ``float32`` NumPy array in
    the audio thread.  It **must not block** — copy the data and process it
    asynchronously if needed.

    Args:
        sd:                The ``sounddevice`` module.
        input_device:      Input device index (``None`` = system default).
        output_device:     Output device index (``None`` = system default).
        sample_rate:       Sampling rate in Hz (default 44100).
        block_size:        Frames per callback block (default 512).
        analysis_callback: Optional callable ``(samples: np.ndarray) -> None``
                           invoked for every input block.
    """

    def __init__(
        self,
        sd,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        sample_rate: int = 44100,
        block_size: int = 512,
        analysis_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self._sd = sd
        self._input_device = input_device
        self._output_device = output_device
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._analysis_callback = analysis_callback
        self._stream = None
        self._open()

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the sounddevice duplex stream."""
        try:
            self._stream = self._sd.Stream(
                device=(self._input_device, self._output_device),
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                dtype="float32",
                channels=1,
                callback=self._callback,
            )
            self._stream.start()
            logger.info(
                "Audio passthrough stream opened — "
                "in=%s, out=%s, rate=%d, block=%d",
                self._input_device,
                self._output_device,
                self._sample_rate,
                self._block_size,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not open audio passthrough stream: {exc}"
            ) from exc

    def _callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """sounddevice duplex callback — runs in the audio thread.

        Copies input directly to output (live passthrough) and forwards a
        mono copy to the analysis callback without blocking the audio thread.
        """
        if status:
            logger.debug("Passthrough stream status: %s", status)

        # Live passthrough: copy input to output.
        outdata[:] = indata

        # Non-blocking analysis: hand off a copy so analysis cannot stall playback.
        if self._analysis_callback is not None:
            mono = indata.reshape(-1).copy()  # safe for both (frames,) and (frames,1)
            self._analysis_callback(mono)

    def close(self) -> None:
        """Stop and close the duplex stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio passthrough stream closed")
