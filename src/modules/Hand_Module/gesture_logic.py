"""
Deprecated location.

Gesture/action matching now lives in `jarvis/src/modules/Vision_Module/gesture_logic.py`.
This module re-exports it to avoid breaking old imports.
"""

from ..Vision_Module.gesture_logic import GestureProcessor

__all__ = ["GestureProcessor"]