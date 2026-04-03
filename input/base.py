"""
input/base.py — Abstract base class for all DogDayDiffuser frame sources.

Any concrete source (webcam, USB video, network stream, …) must subclass
FrameSource and implement the three abstract methods below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class FrameSource(ABC):
    """Common interface for all frame providers.

    The main processing pipeline only ever calls ``open``, ``read``, and
    ``release`` on a ``FrameSource``, so it doesn't need to know whether
    frames come from a webcam, a file, or anything else.
    """

    @abstractmethod
    def open(self) -> bool:
        """Open / initialise the source.

        Returns:
            ``True`` if the source was opened successfully, ``False``
            otherwise.  Implementations should *not* raise on failure —
            return ``False`` instead so the caller can decide what to do.
        """

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the next frame.

        Returns:
            A ``(ok, frame)`` pair where *ok* is ``True`` when a valid BGR
            uint8 NumPy array is returned in *frame*.  Both values are
            ``False``/``None`` when no frame is available.
        """

    @abstractmethod
    def release(self) -> None:
        """Release all resources held by this source."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable identifier shown in the status overlay."""

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """``True`` for real-time sources (webcam), ``False`` for file playback."""
