from __future__ import annotations

"""CHM configuration helpers.

Sensor-specific scientific defaults remain in ``fastgc.sensors`` for backward
compatibility. This module is the future home for CHM-only presets and typed
configuration translation; it must not duplicate sensor defaults.
"""

DEFAULT_GRID_RES = 0.5
DEFAULT_SMOOTH_METHOD = "none"
DEFAULT_PERCENTILE = 99.0

__all__ = [
    "DEFAULT_GRID_RES",
    "DEFAULT_SMOOTH_METHOD",
    "DEFAULT_PERCENTILE",
]
