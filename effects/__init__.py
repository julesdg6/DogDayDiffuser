"""
effects/__init__.py — Effect registry for DogDayDiffuser.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseEffect
from .kaleidoscope import KaleidoscopeEffect
from .feedback import FeedbackEffect
from .warp import WarpEffect
from .color_fx import ColorFXEffect

# Mapping from CLI name to effect class
EFFECTS: Dict[str, Type[BaseEffect]] = {
    "kaleidoscope": KaleidoscopeEffect,
    "feedback": FeedbackEffect,
    "warp": WarpEffect,
    "color": ColorFXEffect,
}

EFFECT_ORDER = list(EFFECTS.keys())

__all__ = [
    "BaseEffect",
    "KaleidoscopeEffect",
    "FeedbackEffect",
    "WarpEffect",
    "ColorFXEffect",
    "EFFECTS",
    "EFFECT_ORDER",
]
