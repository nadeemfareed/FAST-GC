from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SurfaceResult:
    """Canonical in-memory result returned by future CHM surface kernels."""
    array: np.ndarray
    xmin: float
    ymax: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CHMQuality:
    """Optional quality layers attached to a CHM product."""
    confidence: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    support_count: np.ndarray | None = None


__all__ = ["SurfaceResult", "CHMQuality"]
