"""
audio/analysis.py — Audio feature extraction utilities for the audio package.

Re-exports ``AudioFeatures`` and the ``_band_energy`` helper from
``audio_reactivity`` so that modules within this package can import analysis
tools from a single, consistent location without duplicating FFT logic.
"""

from audio_reactivity import AudioFeatures, _band_energy  # noqa: F401

__all__ = ["AudioFeatures", "_band_energy"]
